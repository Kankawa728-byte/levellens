"""Bayesian tuning with Optuna: search on dev, then train the best setting once more as a normal run.

    python -m levellens.tune data=arc model=distilbert head=ce owner=E n_trials=12

The arguments are those of `levellens.run`, plus `n_trials` (default 12) and `study` (a name; default
`<dataset>-<model>`). What happens:

1. Search. Each trial draws one setting from SPACE, trains on the training split and reports the QWK
   on dev after every evaluation. Nothing touches the test split.
   - TPE sampler (tree-structured Parzen estimator): after a few random trials it draws new settings
     near those that scored well, instead of covering a grid.
   - Median pruner: a trial is stopped early when its dev QWK at some evaluation is below the median
     of the earlier trials at the same evaluation. Weak settings cost one or two epochs, not all.
   - The study is kept in `<LEVELLENS_CACHE>/optuna_storage/<study>.db`. Starting the same command
     again continues the study until it has `n_trials` trials.
2. Re-train. The best setting is run through `levellens.run` with `variant=optuna`: a result row in
   `runs.jsonl`, predictions, and the saved model, like any other run.
3. The table of all trials goes to `results/optuna/<study>.csv` and the best setting to `<study>.json`.
"""

from __future__ import annotations

import json
import sys

import optuna
from omegaconf import DictConfig, OmegaConf

from levellens.config import compose
from levellens.data import info, load_split, stratified_subset
from levellens.models.base import build_model
from levellens.paths import cache_dir, results_dir
from levellens.run import resolve_device, run, set_seed

STARTUP_TRIALS = 4  # random trials before TPE starts to model the scores
WARMUP_EVALS = 1    # a trial is never pruned at its first evaluation

HEADS = ("ce", "corn")


def suggest(trial: optuna.Trial) -> list[str]:
    """Draw one setting. Return it as overrides for `levellens.config.compose`."""
    return [
        f"head={trial.suggest_categorical('head', HEADS)}",
        f"model.params.lr={trial.suggest_float('lr', 1e-5, 1e-4, log=True)}",
        f"model.params.batch_size={trial.suggest_categorical('batch_size', [8, 16, 32])}",
        f"model.params.weight_decay={trial.suggest_float('weight_decay', 0.0, 0.1)}",
        f"model.params.warmup_ratio={trial.suggest_float('warmup_ratio', 0.0, 0.2)}",
        f"model.params.dropout={trial.suggest_float('dropout', 0.0, 0.3)}",
    ]


def split_arguments(argv: list[str]) -> tuple[list[str], int, str | None]:
    """Take `n_trials=` and `study=` out of the arguments. Return the rest, the number of trials and the study name."""
    overrides, n_trials, study = [], 12, None
    for argument in argv:
        key, _, value = argument.partition("=")
        if key == "n_trials":
            n_trials = int(value)
        elif key == "study":
            study = value
        else:
            overrides.append(argument)
    return overrides, n_trials, study


def tune(overrides: list[str], n_trials: int = 12, study_name: str | None = None) -> dict:
    """Run the search and the re-training. Return the result row of the re-trained model."""
    base = compose(overrides)
    dataset = info(base.data.name)
    study_name = study_name or f"{dataset.name}-{base.model.name}"
    train = load_split(dataset.name, base.data.train_split)
    dev = load_split(dataset.name, "dev")
    if base.limit_train:
        train = stratified_subset(train, base.limit_train, seed=base.seed)

    def objective(trial: optuna.Trial) -> float:
        """Train one trial with the settings that Optuna suggests. Return its best dev QWK."""
        config: DictConfig = compose([*overrides, *suggest(trial)])
        config.device = resolve_device(config.device)
        set_seed(config.seed)
        model = build_model(config, dataset.num_levels)

        def report(step: int, score: float) -> None:
            """Report the dev QWK to Optuna, and stop the trial if the pruner says so."""
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model.on_eval = report
        model.fit(train, dev, config)
        trial.set_user_attr("evaluations", len(model.history))
        return model.best_dev_qwk

    folder = cache_dir() / "optuna_storage"
    folder.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{(folder / f'{study_name}.db').as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(base.seed), n_startup_trials=STARTUP_TRIALS),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=STARTUP_TRIALS, n_warmup_steps=WARMUP_EVALS),
    )
    finished = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    if len(finished) < n_trials:
        study.optimize(objective, n_trials=n_trials - len(finished))

    best = study.best_trial
    print(f"{study_name}: best trial {best.number}, dev QWK {best.value:.4f}, {best.params}")
    out = results_dir() / "optuna"
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / f"{study_name}.csv", index=False, lineterminator="\n")
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    summary = {"study": study_name, "trials": len(study.trials), "pruned": pruned, "best_trial": best.number,
               "best_dev_qwk": best.value, "best_params": best.params, "base": overrides}
    (out / f"{study_name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    best_overrides = suggest(optuna.trial.FixedTrial(best.params))
    final = compose([*overrides, *best_overrides, "variant=optuna", "save_model=true"])
    print(f"{study_name}: re-training the best setting: {' '.join(best_overrides)}")
    return run(final)


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    overrides, n_trials, study = split_arguments(sys.argv[1:] if argv is None else argv)
    tune(overrides, n_trials, study)


if __name__ == "__main__":
    main()
