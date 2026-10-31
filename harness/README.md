# InferArena / TrainArena benchmark harnesses

Two harnesses live here: `arena_bench.py` for **inference** engines (InferArena) and
`train_bench.py` for **fine-tuning** engines (TrainArena, [below](#trainarena-train_benchpy)).

## InferArena (`arena_bench.py`)

Measures a local LLM inference engine with the **same requests for every engine**, and writes
`arena_result.json` for upload to the InferArena leaderboard.

```
pip install httpx
# put prompts_v2.py next to arena_bench.py
python arena_bench.py --base-url <engine's OpenAI-compatible URL> --model <model id> \
    --engine <slug> --engine-version <version> --quant <quant> --mode full_vram|offload
```

## Rules v2 (suite v2, the default)

These are the InferArena rules for every engine, current and future. The Lab benchmark tab uses the
same rules on runs made by our team.

**Comparisons.** Speeds are only compared on the same GPU and the same model (a *cell*). Models that fit
in GPU memory (`--mode full_vram`) and models that must be offloaded (`--mode offload`) are ranked
separately.

**Prompts.** `prompts_v2.py` holds the public set, and all three sets are scored:
- **A:** 4 short chat and code requests, up to 300 output tokens.
- **V:** 14 held-out tasks on real code, documents and data, up to 512 output tokens.
- **R:** 7 headline copy-heavy tasks (edit, rename or rewrite a whole file, log triage), up to 512 output tokens.

**Run settings.** Requests are greedy (temperature 0, top-k 1), with a fixed seed and thinking turned off,
identical for every engine. Each prompt runs twice: pass 1 in order, pass 2 reversed.

**Score.**
- Decode tok/s = tokens after the first ÷ time after the first token, as the median of the two passes per
  prompt.
- A set's score is the **geometric mean over its prompts**.
- TTFT is from pass 1.

**Exact decoding (main board).** Output identical to the engine's standard greedy decode (`--decode-class exact`,
the default). Lossless speed-ups count here: speculative or MTP drafts that are verified token by token, so the
output doesn't change. The harness checks whether the output reproduces across the two passes; a run that doesn't is flagged
`not_reproducible` and shown with a badge, but it is still ranked (offload setups often aren't bit-identical
across restarts).

**Fast class (separate board).** For a speculative or multi-token mode, first save the engine's exact run
as `exact.json`, then run the fast mode with:
```
--decode-class fast --plain-ref exact.json
```
The result records how many set-A outputs match the exact run (e.g. 1/4). Use `--out` to keep the two files apart.

**Coverage.** If an engine can't run a model on your GPU, record that instead of skipping it:
```
--record-status failed_start --reason "out of memory at start"
```
or `--record-status not_supported` with a reason. Failures stay visible, and a setup we broke ourselves is
re-run, not counted.

**Categories.** Results are ranked per category, each with its own GPUs, models and engines:
- **Overall:** every engine, models that fit in GPU memory.
- **General-purpose:** engines that load any model architecture from a standard format (GGUF or Hugging
  Face weights), as opposed to engines built for particular models.
- **One board per offload model**, e.g. 35B and 27B offload, or Flash-Next offload.

A category can score fewer prompt sets when the others were used for tuning.

**Ranking.** The Speed Index fits log(speed) = engine + cell over all verified exact runs, with Ollama =
1.0× (in a category without Ollama, the engine that ran the most cells is the baseline). Engines need at least
2 scored cells to be ranked, or as many as the category has; otherwise they're provisional. Engines within 3%
share a rank. "Runs on" = cells finished ÷ cells tried.

**Tuning is disclosed.** If an engine's settings were tuned on any of these prompts, say so in `--notes`.

**Speed records.** Each engine's best single-prompt speed (any set, any mode) is listed separately. It is the median of
at least 2 passes on that prompt, and the passes must agree within 10%.

Suite v1 (`--suite v1`: synthetic short / long / concurrent requests, described below) is still
accepted.

## Engine examples

The harness only talks to the engine's OpenAI-compatible endpoint (`/v1/chat/completions`). Start
the engine the way you normally would, then point the harness at it.

| engine | typical base URL | `--engine` |
|---|---|---|
| Ollama | `http://127.0.0.1:11434/v1` | `ollama` |
| llama.cpp server | `http://127.0.0.1:8080/v1` | `llamacpp` |
| vLLM | `http://127.0.0.1:8000/v1` | `vllm` |
| SGLang | `http://127.0.0.1:30000/v1` | `sglang` |
| AlphaLlama | your AlphaLlama server's `/v1` URL | `alphallama` |

Other engines (freetoken, ninfer, …) work the same way if they expose an OpenAI-compatible
streaming endpoint. If your engine isn't listed on the site yet, ask us to add it — engines are
data on our side, not code.

`--mode full_vram` means the whole model is in GPU memory; `--mode offload` means part of it lives in
system RAM. Put launch flags that matter (context size, batch settings, speculative decoding, …) in
`--notes` so reviewers and readers can reproduce the run.

## What it measures (suite v1)

| workload | prompt | output | concurrency |
|---|---|---|---|
| `short` | ~512 tokens | 256 tokens | 1 |
| `long` | ~4096 tokens | 256 tokens | 1 |
| `concurrent` | ~512 tokens | 256 tokens | 8 parallel requests |

Each workload runs 3 times by default (`--repeats`) and the median is reported.

- **TTFT** — time from sending the request to the first streamed token.
- **Prefill tok/s** — prompt tokens ÷ TTFT.
- **Decode tok/s** — (generated tokens − 1) ÷ time from the first to the last token.
- **Aggregate tok/s** (concurrent) — all generated tokens ÷ wall time for the batch.
- **Peak VRAM** — the highest GPU memory in use during the run, via `nvidia-smi`. This includes other
  processes on the GPU.

Token counts come from the engine's own `usage` report. If the engine doesn't report usage while
streaming, the harness counts streamed chunks instead, which is less exact.

## How it keeps the comparison fair

- **Identical requests.** Every engine gets the same standard OpenAI fields: greedy decoding
  (`temperature 0`, fixed `seed`). There are no engine-specific parameters.
- **No prefix-cache shortcuts.** Each timed request starts with a unique tag, so an engine with
  prefix caching can't skip prefill by recognising a repeated prompt.
- **Output fingerprint.** The harness records a sha256 of the first 256 characters of greedy output
  for a fixed prompt. The same engine version, model and quant should reproduce it exactly; if it
  doesn't, the run is flagged for review.
- **Warm-up excluded.** A short warm-up request runs first, so model loading isn't timed.

## What it does not measure

- **It's client-side timing.** Local HTTP overhead is included, and it's the same for every engine.
- **Quant formats differ across engines.** For example, GGUF `Q4_K_M` and AWQ 4-bit are different
  formats. The site groups them by bit width and always shows the exact quant.
- **It doesn't measure output quality.** It only measures speed.

## After the run

Upload `arena_result.json` on the InferArena page (Submit tab, sign-in required). A reviewer checks
each submission before it appears on the board. Results then earn a confidence badge:

- **verified**: reproduced within ±10% by another user
- **single source**: one submitter so far
- **lab**: measured by the AlphaChat team
- **disputed**: other users' runs disagree by more than 25%

---

## TrainArena (`train_bench.py`)

Measures a local fine-tuning engine (AlphaTrain, Unsloth, Axolotl, LLaMA-Factory, torchtune, Hugging
Face TRL, …) on the **same data with the same hyperparameters**, and writes `trainarena_result.json`.

```
pip install httpx transformers torch peft      # torch + peft only for the quality check
python train_bench.py --engine <slug> --engine-version <version> --model <HF model id>     --workload lora|qlora --mode full_vram|offload --     <your training command, using the {placeholders} below>
```

Training engines have no common API, so the harness doesn't drive them. It runs **your** command
twice and measures it from outside. Your command (usually a short wrapper script around your
engine) must:

1. train an adapter on the JSONL file at **`{data}`** (one sample per line; train on the `text`
   field, loss on all tokens, or build the same text from `instruction`/`input`/`output`);
2. use the workload's hyperparameters: `{seq_len}` max length, `{batch}` samples per optimizer step,
   `{lr}` learning rate, LoRA rank `{rank}` and alpha `{alpha}` on all linear projections,
   `{epochs}` epoch(s);
3. save the adapter in standard PEFT format (`adapter_config.json` + weights) anywhere under
   **`{out}`**.

Other placeholders: `{model}`, `{samples}`, `{workload}`, `{base_precision}`. Braces that aren't
placeholders are left untouched.

### Workloads (suite train-v1)

| workload | frozen base | seq len | batch | lr | LoRA rank / alpha | epochs |
|---|---|---|---|---|---|---|
| `lora` | bf16 | 2048 | 8 | 2e-4 | 16 / 16 | 1 |
| `qlora` | 4-bit (nf4) | 2048 | 8 | 2e-4 | 16 / 16 | 1 |

One result file holds one workload. A model too large for bf16 LoRA on your GPU can still enter as
`qlora`.

Data: [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) (CC-BY-4.0),
pinned to a revision and checked by sha256. It is shuffled with a fixed seed: 200 rows are held out
for evaluation, and the runs train on the first 1,000 and the first 2,000 of the rest.

### What it measures

- **Train tok/s**: `(tokens(2000) − tokens(1000)) ÷ (time(2000) − time(1000))`. Model load,
  compilation and warm-up are paid by both runs, so they cancel. This is steady-state speed.
  Tokens are the dataset's tokens under the model's own tokenizer, so padding doesn't count:
  an engine that packs sequences well is faster, as it really is.
- **Time to train**: the whole 2,000-sample run, start-up included.
- **Peak VRAM**: as for InferArena (`nvidia-smi`, includes other processes on the GPU).
- **Eval loss**: the 2,000-sample adapter, loaded onto the bf16 base, scored on the held-out rows,
  next to the base model's own eval loss.

### How it keeps the comparison fair

- **Same data, same hyperparameters** for every engine, delivered through the placeholders.
- **Quality guard.** Being fast by training less (fewer steps, a lower rank, skipped layers) shows
  up as a worse eval loss. Runs whose eval loss is far (>5%) from other runs on the same model and
  workload are flagged for review. A run that doesn't beat the base model is flagged too.
- **`--skip-eval`** is allowed (eval needs the model to fit for inference), but those runs are
  flagged `no_quality_check` and can never reach **verified**.
- **Compute bound.** A speed above what the GPU's spec-sheet compute allows is flagged, not refused.

### What it does not measure

- **Full fine-tuning.** It's planned for a later suite.
- **Multi-node training.**
- **Model quality beyond the eval loss on this one dataset.**

Upload `trainarena_result.json` on the TrainArena page. Review and confidence badges work exactly as
for InferArena (above).
