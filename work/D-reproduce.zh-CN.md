# 复现分工 D 的结果

English: [D-reproduce.md](D-reproduce.md)

在仓库根目录运行下面的命令，先装好项目依赖。它们使用的数据划分和随机种子，和 `results/runs.jsonl` 里分工 D 的 28 行相同。重跑时请用单独的输出目录，不要和已提交的结果混在一起。下面的例子用 Bash。在 PowerShell 里用 `$env:LEVELLENS_RESULTS = 'results/share-d-reproduction'` 设置目录。

```bash
python -m pip install -r requirements.txt
python -m levellens.data.prepare race arc math
export LEVELLENS_RESULTS=results/share-d-reproduction
python -m pytest tests/test_classic.py tests/test_tfidf.py tests/test_r2de.py tests/test_linguistic.py tests/test_global_shap.py tests/test_llm.py -q
```

## D1–D5：CPU 上的运行

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

SHAP 这条命令会载入三个保存好的 LightGBM 模型，写出三张 PNG、中文说明和数值汇总。说明文字见 [summary.zh-CN.md](../results/figures/share_d_shap/summary.zh-CN.md)，它的英文版经过人工核对。

## D6：在 CUDA 上运行 Qwen

按照 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装支持 CUDA 的 PyTorch。已发布的结果用的是 PyTorch 2.9.1 加 CUDA 12.8，Transformers 4.57.6，bitsandbytes 0.49.2。

```bash
python -m pip install 'transformers==4.57.6' 'bitsandbytes==0.49.2'
python -m levellens.models.llm --download
for ds in race arc math; do
    python -m levellens.run data="$ds" model=qwen_zero_shot seed=0 owner=D device=cuda \
        'eval_splits=[test_1k]' model.params.quantization=nf4 variant=nf4
done
python -m levellens.results
```

Qwen 这条命令会把固定版本的官方权重下载到模型缓存里。评估时载入本地权重，先在分层抽样的验证样本上选提示词，再给固定的 1,000 题测试子集打分。结果和预测写在 `LEVELLENS_RESULTS` 指向的目录下。
