# 同一引擎的配置对照

`bench_vs_llama.py` 可以对同一个 Strata 引擎依次加载不同配置，对照环境变量或引擎参数。原有 Strata 与 llama.cpp 对照仍可使用。每轮交替调整配置顺序，测量请求通过 `/v1/chat/completions` 发送，结果写到输出目录的 `raw/`，`summary` 汇总中位数。

## 两条铁律

1. 速度和延迟只在 API 边界测量，使用接口返回的时间和吞吐量。脚本仍按原流程采样显存和内存占用，但不将它们当作速度计时。绝不设置 Strata 的计时或剖析环境变量，例如 `STRATA_PREFILL_TIMING`、`STRATA_DECODE_TIMING`、`STRATA_VERIFY_PROFILE` 和 `STRATA_MTP_PROFILE`；其他此类变量也必须清除。
2. 测试实例必须复制生产环境。`run` 和 `quality` 都带 `--require-env-file bench/prod.env`。启动前逐一核对文件中每个变量与每个配置的引擎有效环境，包括未被 `--configs` 选中的配置；任一缺失或不符就全部停止，不启动引擎。文件里的 `<file:/绝对路径>` 在运行时读取，去掉文件结尾的换行；密钥不显示。环境文件只做核对，不会自动设置变量。`--allow-diff NAME` 可重复指定，仅供明确获准的例外；放行后不能再声称该变量与生产一致。

测试需要独占 GPU。操作员须在测试前自行停止生产服务并确保设备空闲。脚本不会操作 systemd。不要对生产配置或生产输出目录直接运行：先准备隔离的 Strata 配置，将其中的 `log`、其他可写路径和工作目录指向测试目录，避免覆盖生产文件。引擎、模型和分词器路径须由操作员按实际情况填写。密钥只通过环境传递，不要写进机器配置、命令参数、记录或仓库。

机器配置的 `configs[].env` 是可选的字符串对象，会覆盖调用进程的环境。对 Strata，源配置自带的 `env` 也参与有效环境；同名项以本次 `configs[].env` 为准。`cells`、`gen` 可省略，缺省为 4k 两次、32k 一次和生成 256 token。`prompt_tokens` 是单元的标记，不会重新计算提示文件的实际 token 数；使用者须自行核实分词器和提示长度。

`run --check` 只读检查配置、守卫、提示文件、分词器文件和程序路径，并显示各配置的有效环境差异，不加载引擎或使用 GPU。密钥显示为 `<redacted>`。分词器从 `--tokenizer`、机器配置的 `tokenizer` 或 Strata 源配置的 `tokenizer` 取得，依次优先；需要 `vocab.json`、`merges.txt`、`token_type.json`。

`quality` 原本保存两个固定提示、温度为 0 的完整答案。加 `--smoke` 后，逐提示以 UTF-8 字节比较所有配置的最终答案，显示 `IDENTICAL` 或 `DIFFERS`；不同、空答案或执行错误返回非零。答案仍写在 `raw/quality.json`。温度为 0 也不保证不同实现或硬件一定产生相同答案。

## 两次对照的共同准备

在仓库根目录执行。先准备隔离配置 `/tmp/strata-ab-strata.json`，并把下面的 `TOKENIZER` 改为该模型的实际分词器目录。确认此配置使用同一个引擎和模型，且其 `env` 没有设置待测试的 `STRATA_MTP_BATCH_F16`。以下命令不停止服务。

```sh
cd /home/g0/strata-v100
export STRATA_CFG=/tmp/strata-ab-strata.json
export SERVE="$PWD/serve"
export TOKENIZER=/models/strata-data/packs/unsloth-ud-q4_k_xl/tokenizer

# 不要开启 shell 的命令回显；不显示、不打印密钥。
set +x
export STRATA_KV_PREFETCH=1 STRATA_MMQ_KQ=1 STRATA_UNBUFFERED_LOAD=1
export STRATA_PREFILL_CPU_SHARE=0 STRATA_SM70_TABLE=1
export STRATA_ADAPT_OVERLAP_RESIDENT=1 STRATA_ADAPT_OVERLAP=1 STRATA_NUMA_POOL=0
export STRATA_API_KEY="$(cat /models/strata-data/api-key.txt)"
unset STRATA_MTP_BATCH_F16 STRATA_PREFILL_TIMING STRATA_DECODE_TIMING
unset STRATA_DRAFT_TIMING STRATA_SPLIT_TIMING STRATA_COMMIT_TIMING
unset STRATA_VERIFY_PROFILE STRATA_MTP_PROFILE STRATA_DF2_PROFILE STRATA_DF2_TIMING
```

以上只是常见计时和剖析变量的清理示例；脚本不会自动检查或清除所有此类变量，使用者还须确认其他此类变量未设置。密钥文件必须存在且当前用户可读；不要回显变量或打开命令回显。示例复用仓库内的 4k 提示文件；换模型时要先核实它的实际 token 数。

## 对照 1a：基准与 MTP 半精度

```sh
OUT=/tmp/strata-ab-mtp
mkdir -p "$OUT"
cp bench/results/2026-10-06-llamacpp-0.6.0/prompt-4k.txt "$OUT/prompt-4k.txt"
cat > "$OUT/machine.json" <<EOF
{
  "strata_cfg": "$STRATA_CFG",
  "serve": "$SERVE",
  "tokenizer": "$TOKENIZER",
  "configs": [
    {"name": "baseline", "kind": "strata", "args": []},
    {"name": "mtp_f16", "kind": "strata", "args": [], "env": {"STRATA_MTP_BATCH_F16": "1"}}
  ],
  "cells": [{"name": "4k", "prompt_tokens": 4000, "reps": 2}],
  "gen": 256
}
EOF
python3 bench/bench_vs_llama.py run --machine "$OUT/machine.json" --out "$OUT" --require-env-file bench/prod.env --check
```

先确认检查输出中基准没有 `STRATA_MTP_BATCH_F16=1`，对照配置为 `1`，并确认检查通过、GPU 已独占，才执行：

```sh
python3 bench/bench_vs_llama.py quality --machine "$OUT/machine.json" --out "$OUT" --require-env-file bench/prod.env --smoke &&
python3 bench/bench_vs_llama.py run --machine "$OUT/machine.json" --out "$OUT" --require-env-file bench/prod.env --rounds 2 &&
python3 bench/bench_vs_llama.py summary --out "$OUT"
```

检查和质量对照不是速度测量；质量不同要先分析，不要当作等价的性能结果。

## 通用对照：改变引擎参数

下面以追加 `--spec 2` 为例；实际参数须由操作员选定，并确认与隔离源配置的基准不同。两项仍用同一源配置、同一引擎、同一模型。

```sh
OUT=/tmp/strata-ab-args
mkdir -p "$OUT"
cp bench/results/2026-10-06-llamacpp-0.6.0/prompt-4k.txt "$OUT/prompt-4k.txt"
cat > "$OUT/machine.json" <<EOF
{
  "strata_cfg": "$STRATA_CFG",
  "serve": "$SERVE",
  "tokenizer": "$TOKENIZER",
  "configs": [
    {"name": "baseline", "kind": "strata", "args": []},
    {"name": "spec_2", "kind": "strata", "args": ["--spec", "2"]}
  ],
  "cells": [{"name": "4k", "prompt_tokens": 4000, "reps": 2}],
  "gen": 256
}
EOF
python3 bench/bench_vs_llama.py run --machine "$OUT/machine.json" --out "$OUT" --require-env-file bench/prod.env --check
```

只在检查成功、操作员确认 GPU 独占且确实要做测试时执行：

```sh
python3 bench/bench_vs_llama.py quality --machine "$OUT/machine.json" --out "$OUT" --require-env-file bench/prod.env --smoke &&
python3 bench/bench_vs_llama.py run --machine "$OUT/machine.json" --out "$OUT" --require-env-file bench/prod.env --rounds 2 &&
python3 bench/bench_vs_llama.py summary --out "$OUT"
```

原有跨引擎配置不传新选项时，仍按原有流程运行。
