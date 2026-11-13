# Reproduce the Share D results

Run these commands from the repository root with the project dependencies installed. They use the same dataset splits and seed as the 28 Share D rows in `results/runs.jsonl`. Use a separate output directory to keep a rerun separate from the checked-in results. The examples below use Bash. In PowerShell, set the directory with `$env:LEVELLENS_RESULTS = 'results/share-d-reproduction'`.

```bash
python -m pip install -r requirements.txt
python -m levellens.data.prepare race arc math
export LEVELLENS_RESULTS=results/share-d-reproduction
python -m pytest tests/test_classic.py tests/test_tfidf.py tests/test_r2de.py tests/test_linguistic.py tests/test_global_shap.py tests/test_llm.py -q
```

## D1–D5: CPU runs

```bash
for ds in race arc math; do
    for model in majority stratified_random tfidf_logreg r2de; do
        python -m levellens.run data="$ds" model="$model" seed=0 device=cpu owner=D
    done
    for model in tfidf_logreg tfidf_svm tfidf_cnb; do
        python -m levellens.run data="$ds" model="$model" seed=0 device=cpu owner=D \
            model.params.tune=true variant=cv5
    done
    python -m levellens.run data="$ds" model=linguistic_lightgbm seed=0 device=cpu owner=D save_model=true
done
python -m levellens.run data=math model=subject_prior seed=0 device=cpu owner=D
python -m levellens.explain.global_shap --seed 0
```

The SHAP command loads the three saved LightGBM models and writes three PNGs, the Chinese captions and the numerical summary. The reviewed English captions are in [summary.md](../results/figures/share_d_shap/summary.md).

## D6: Qwen on CUDA

Install a CUDA-enabled PyTorch build following the [PyTorch installation guide](https://pytorch.org/get-started/locally/). The published runs used PyTorch 2.9.1 with CUDA 12.8, Transformers 4.57.6 and bitsandbytes 0.49.2.

```bash
python -m pip install 'transformers==4.57.6' 'bitsandbytes==0.49.2'
python -m levellens.models.llm --download
for ds in race arc math; do
    python -m levellens.run data="$ds" model=qwen_zero_shot seed=0 owner=D device=cuda \
        'eval_splits=[test_1k]' model.params.quantization=nf4 variant=nf4
done
python -m levellens.results
```

The Qwen command downloads the pinned official snapshot into the model cache. Evaluation then loads local weights and selects a prompt using the stratified validation sample before scoring the fixed 1,000-question test split. Results and predictions are written under `LEVELLENS_RESULTS`.
