"""Share D baseline models.

Run the D1 floors from the repository root::

    python -m levellens.run data=arc model=majority device=cpu owner=D
    python -m levellens.run data=arc model=stratified_random device=cpu owner=D
    python -m levellens.run data=math model=subject_prior device=cpu owner=D

Run majority and stratified random on each of race, arc and math. The topic
prior uses MATH's meta["type"] and applies only to math.

Run the Share D tests from the repository root::

    python -m pytest tests/test_classic.py tests/test_tfidf.py

Run the default D2 logistic regression on the toy dataset::

    python -m levellens.run data=toy model=tfidf_logreg device=cpu owner=D

For formal runs, first run ``python -m levellens.data.prepare race arc math``
to obtain the team's splits, then replace ``data=toy`` with each dataset.
For five-fold search, add ``model.params.tune=true variant=cv5`` and run
each of ``model=tfidf_logreg``, ``model=tfidf_svm``, and ``model=tfidf_cnb``.
The runner saves the selected parameters under ``config.model.cv_selection``.

Run the D3 R2DE-style regression baseline on the same team splits::

    python -m levellens.run data=arc model=r2de device=cpu owner=D
    python -m pytest tests/test_r2de.py

Replace ``data=arc`` with ``data=math`` or ``data=race`` for the other runs.
D3 uses word-unigram TF-IDF and a fixed CPU-sized forest. It uses the shared
question-and-options text and full ARC training split. See the source model
at https://github.com/lucabenedetto/r2de-nlp-to-estimating-irt-parameters.

Install the optional D4 dependencies and run the surface-feature model::

    python -m pip install "lightgbm>=4.6,<5" "cmudict>=1,<2" "pyphen>=0.17,<1" "wordfreq>=3.1,<4"
    python -m levellens.run data=arc model=linguistic_lightgbm device=cpu owner=D save_model=true
    python -m pytest tests/test_linguistic.py

Replace ``data=arc`` with ``data=math`` or ``data=race`` for the other runs.
The feature cache lives under ``LEVELLENS_CACHE/share-d/linguistic``. Saved
models can be loaded by the shared Predictor and used for D5.
"""

from collections import Counter
import json
import os
import random

from joblib import Memory, parallel_config
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from levellens.features.tfidf import make_r2de_tfidf, make_tfidf
from levellens.models.base import Model, texts
from levellens.paths import cache_dir


def _training_levels(train, num_levels):
    """Return the training levels as an array, after checking that they are integers in the valid range."""
    levels = np.array([item["level"] for item in train])
    if not levels.size or levels.dtype.kind not in "iu" or np.any((levels < 0) | (levels >= num_levels)):
        raise ValueError("Training needs at least one integer level in [0, num_levels)")
    return levels


class StratifiedRandomBaseline(Model):
    """Draw one level per question from the training set's label frequencies."""

    def fit(self, train, dev, config):
        """Count the levels of the training set and keep the seed."""
        levels = _training_levels(train, self.num_levels)
        self.weights_ = np.bincount(levels, minlength=self.num_levels).tolist()
        self.seed_ = int(config.seed)

    def predict_proba(self, items):
        """Return a one-hot row per question. The draw depends only on the seed, the dataset and the question id."""
        # Each item has its own draw so reordering, batching or evaluating test_1k
        # directly cannot change the prediction for a question.
        proba = np.zeros((len(items), self.num_levels))
        for index, item in enumerate(items):
            rng = random.Random(f"{self.seed_}:{item['dataset']}:{item['id']}")
            level = rng.choices(range(self.num_levels), weights=self.weights_)[0]
            proba[index, level] = 1.0
        return proba


class SubjectPriorBaseline(Model):
    """Predict the most frequent training level for each MATH topic."""

    def fit(self, train, dev, config):
        """Find the most frequent level of each MATH problem type, and the overall majority level as the fallback."""
        if config.data.name != "math" or any(item["dataset"] != "math" for item in train):
            raise ValueError("The subject prior applies only to MATH")
        levels = _training_levels(train, self.num_levels)
        counts = {}
        for item, level in zip(train, levels):
            topic = item.get("meta", {}).get("type")
            if topic:
                counts.setdefault(topic, Counter())[level] += 1
        self.topic_modes_ = {topic: max(values, key=lambda level: (values[level], level))
                             for topic, values in counts.items()}
        overall = Counter(levels)
        # Match the shared majority model's higher-level tie break.
        self.majority_ = max(overall, key=lambda level: (overall[level], level))

    def predict_proba(self, items):
        """Return a one-hot row per question: the level of its problem type, or the majority level for an unknown type."""
        proba = np.zeros((len(items), self.num_levels))
        for index, item in enumerate(items):
            if item["dataset"] != "math":
                raise ValueError("The subject prior applies only to MATH")
            topic = item.get("meta", {}).get("type")
            proba[index, self.topic_modes_.get(topic, self.majority_)] = 1.0
        return proba


def _training_folds(train, labels, seed):
    """Five training folds, keeping all questions from a RACE passage together."""
    if any(count < 5 for count in Counter(labels).values()):
        raise ValueError("Five-fold CV needs at least five training examples per observed level")
    if train[0].get("dataset") == "race":
        group_ids = {}
        groups = []
        for item in train:
            meta = item["meta"]
            key = (meta["source"], meta["passage_id"])
            groups.append(group_ids.setdefault(key, len(group_ids)))
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(splitter.split(labels, labels, groups))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(splitter.split(labels, labels))
    levels = set(labels)
    if any(set(labels[fit]) != levels or set(labels[held_out]) != levels for fit, held_out in folds):
        raise ValueError("Five-fold CV requires every observed level in every fold")
    return folds


class _TfidfClassifier(Model):
    """Fit TF-IDF inside each fold and score held-out training rows by QWK."""

    calibrate = False

    def fit(self, train, dev, config):
        """Fit TF-IDF and the classifier on the training set.

        With `tune`, the setting is chosen first by five-fold QWK on the training set. With `calibrate`,
        a sigmoid calibration is fitted on out-of-fold scores. `dev` is not used.
        """
        x = texts(train)
        y = np.array([item["level"] for item in train])
        if not len(y) or len(np.unique(y)) < 2:
            raise ValueError("TF-IDF classification requires at least two training levels")
        if np.any((y < 0) | (y >= self.num_levels)):
            raise ValueError("Training levels must be within the dataset's label range")
        seed = int(config.seed)
        tune = self.params.get("tune", False)
        folds = _training_folds(train, y, seed) if tune or self.calibrate else None
        estimator = self._estimator(seed)
        # Reuse the independently fitted features for each fold across candidates
        # and across D2 classifiers. A different set of training rows has its own key.
        memory = None
        if tune:
            location = str(cache_dir() / "d2-tfidf")
            if os.name == "nt":
                # joblib adds nested function names, hashes and temporary filenames.
                location = (r"\\?\UNC" + location[1:] if location.startswith("\\\\")
                            else "\\\\?\\" + location)
            memory = Memory(location, verbose=0)
        jobs = self.params.get("jobs", 3 if tune else 1)
        pipeline = Pipeline([
            ("tfidf", make_tfidf(
                min_df=self.params.get("min_df", 2),
                word_limit=self.params.get("word_limit", 50_000),
                char_limit=self.params.get("char_limit", 100_000),
            )),
            ("classifier", estimator),
        ], memory=memory)
        # Share the input between folds and avoid a BLAS thread pool per worker.
        with threadpool_limits(limits=1), parallel_config(backend="threading"):
            if tune:
                scorer = make_scorer(cohen_kappa_score, weights="quadratic",
                                     labels=list(range(self.num_levels)))
                search = GridSearchCV(
                    pipeline, {f"classifier__{self.parameter}": self.params["grid"]},
                    scoring=scorer, cv=folds, refit=False, error_score="raise",
                    n_jobs=jobs, pre_dispatch=jobs,
                )
                search.fit(x, y)
                pipeline.set_params(**search.best_params_)
                selected_value = search.best_params_[f"classifier__{self.parameter}"]
                config.model.cv_selection = {
                    "metric": "qwk", "folds": 5, "mean_qwk": float(search.best_score_),
                    "params": {self.parameter: float(selected_value)},
                }
            if self.calibrate:
                # Generate out-of-fold decision scores, then refit TF-IDF and SVM on all training rows.
                pipeline = CalibratedClassifierCV(
                    pipeline, method="sigmoid", cv=folds, ensemble=False, n_jobs=jobs,
                )
            self.model_ = pipeline.fit(x, y)

    def predict_proba(self, items):
        """Return the class probabilities, with a zero column for a level that the training set lacks."""
        proba = np.zeros((len(items), self.num_levels))
        if len(items):
            proba[:, self.model_.classes_] = self.model_.predict_proba(texts(items))
            proba /= proba.sum(axis=1, keepdims=True)
        return proba


class TfidfLogReg(_TfidfClassifier):
    """TF-IDF with logistic regression."""

    parameter = "C"

    def _estimator(self, seed):
        """Return the logistic regression with the configured C."""
        return LogisticRegression(C=self.params.get("C", 1.0), max_iter=1000, random_state=seed)


class TfidfLinearSVM(_TfidfClassifier):
    """TF-IDF with LinearSVC and training-only sigmoid calibration."""

    parameter = "C"
    calibrate = True

    def _estimator(self, seed):
        """Return the linear SVM with the configured C."""
        return LinearSVC(C=self.params.get("C", 1.0), max_iter=5000, random_state=seed)


class TfidfComplementNB(_TfidfClassifier):
    """TF-IDF with Complement Naive Bayes."""

    parameter = "alpha"

    def _estimator(self, seed):
        """Return the Complement Naive Bayes classifier with the configured alpha."""
        return ComplementNB(alpha=self.params.get("alpha", 1.0))


class R2DE(Model):
    """Estimate an ordered level with word TF-IDF and random forest regression."""

    def fit(self, train, dev, config):
        """Fit word TF-IDF and a random forest regressor on the level index."""
        vectorizer = make_r2de_tfidf(
            max_features=self.params["max_terms"], min_df=self.params["min_df"],
        )
        regressor = RandomForestRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            min_samples_leaf=self.params["min_samples_leaf"],
            max_features=self.params["max_features"],
            n_jobs=self.params["jobs"],
            random_state=int(config.seed),
        )
        self.model_ = Pipeline([("tfidf", vectorizer), ("forest", regressor)])
        self.model_.fit(texts(train), [item["level"] for item in train])

    def predict_proba(self, items):
        """Return one continuous score per item for the shared regression evaluator."""
        if not len(items):
            return np.empty(0, dtype=float)
        return self.model_.predict(texts(items))


class LinguisticLightGBM(Model):
    """Estimate the target level from Share D's English surface features."""

    def feature_matrix(self, items, *, use_cache=None):
        """Return the ordered input matrix for prediction or D5 explanations."""
        from levellens.features.linguistic import feature_matrix

        if use_cache is None:
            use_cache = self.params.get("cache_features", False)
        return feature_matrix(texts(items), use_cache=use_cache)

    def fit(self, train, dev, config):
        """Fit a LightGBM classifier on the surface features of the training questions."""
        from lightgbm import LGBMClassifier
        from levellens.features.linguistic import FEATURE_NAMES

        y = np.asarray([item["level"] for item in train])
        if np.unique(y).size < 2 or np.any((y < 0) | (y >= self.num_levels)):
            raise ValueError("LightGBM needs at least two valid training levels")
        model = LGBMClassifier(
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            num_leaves=self.params["num_leaves"],
            min_child_samples=self.params["min_child_samples"],
            n_jobs=self.params["jobs"],
            random_state=int(config.seed),
            device_type="cpu",
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(self.feature_matrix(train), y, feature_name=list(FEATURE_NAMES))
        self.booster_ = model.booster_
        self.classes_ = model.classes_
        self.jobs_ = self.params["jobs"]

    def predict_proba(self, items):
        """Return the class probabilities, with a zero column for a level that the training set lacks."""
        proba = np.zeros((len(items), self.num_levels))
        if len(items):
            observed = self.booster_.predict(self.feature_matrix(items), num_threads=self.jobs_)
            if len(self.classes_) == 2:
                observed = np.column_stack((1.0 - observed, observed))
            proba[:, self.classes_] = observed
        return proba

    def save(self, folder):
        """Write the booster and the feature schema into `folder`."""
        from levellens.features.linguistic import FEATURE_VERSION

        folder.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(folder / "lightgbm.txt"))
        state = {
            "num_levels": self.num_levels,
            "classes": self.classes_.tolist(),
            "feature_version": FEATURE_VERSION,
            "jobs": self.jobs_,
        }
        (folder / "linguistic.json").write_text(json.dumps(state), encoding="utf-8")

    @classmethod
    def load(cls, folder):
        """Return the saved model. Fail if it was trained with another feature schema."""
        from lightgbm import Booster
        from levellens.features.linguistic import FEATURE_NAMES, FEATURE_VERSION

        state = json.loads((folder / "linguistic.json").read_text(encoding="utf-8"))
        model = cls(num_levels=state["num_levels"])
        model.booster_ = Booster(model_file=str(folder / "lightgbm.txt"))
        if state["feature_version"] != FEATURE_VERSION or model.booster_.feature_name() != list(FEATURE_NAMES):
            raise ValueError("The saved model uses a different surface feature schema")
        model.classes_ = np.asarray(state["classes"], dtype=int)
        model.jobs_ = state["jobs"]
        return model
