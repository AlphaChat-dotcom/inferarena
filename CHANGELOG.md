# Changelog

## 2026-11-02
- RTX 3090 Flash-Next published from a repeat run with no routing profile loaded (the earlier run auto-loaded one of
  unknown origin and is not used). Its plain-decoding row is not yet measured. The hold on that cell is lifted.
- **Held:** the RTX 3090 Flash-Next results, while that run is repeated. AlphaLlama's automatic start loaded a routing
  profile found next to the model, and it isn't known which prompts the profile was recorded on. The rows stay in the
  edition file under `withheld.cells`. RTX 4080 and RTX 5090 loaded no profile.
- RTX 5090 Flash-Next: the same AlphaLlama release candidate against Strata 0.2.8 at its fastest setup on that card
  (32K, one request slot, STRATA_SPEC_DEPTH=1, a tuned setting; labelled on the board).
- **Flash-Next published again.** RTX 3090 and RTX 4080: AlphaLlama release candidate 3e82a70ea and Strata 0.2.8
  (870aedd9) in one window per machine. Every earlier Flash-Next row is replaced. AlphaLlama has no exact mode with
  drafting on Flash-Next, so its exact row is plain decoding (repeated 25/25 across starts). Earlier AlphaLlama builds
  had a Flash-Next bug that could emit stray tokens in long outputs; this build has the fix.

## 2026-11-01
- RTX 4070 laptop Qwen3.6-35B-A3B: AlphaLlama fast mode re-run on commit 8e9e192. Plain decoding in the same
  window matched the published exact row within about 2%, so that row is unchanged.
- RTX 4080 Qwen3.6-35B-A3B: AlphaLlama re-run on commit 8e9e192 (built from source, not yet packaged), fast and
  plain. Restart reproducibility of the plain row was not re-checked on this build and is not claimed.

## 2026-10-31: open source
- The harness, scoring code, edition data and per-prompt results are published under the MIT license.

## 2026-10-30
- **Withheld:** Qwen3.8-Flash-Next results are withheld from the published board while new releases of AlphaLlama
  and Strata are measured. The rows stay in `data/editions/2026-10.json` under `withheld`. Strata's only results were
  on this model, so it is not on the published board for now.
- RTX 4080 Flash-Next re-run with Strata 0.2.7.1 and AlphaLlama's release build, both in the same session
  (held with the withheld rows above).

## 2026-10-29
- RTX 3090 and RTX 4080 re-runs for AlphaLlama on its newer engine backend; RTX 4070 laptop 35B re-run.
- Results-as-of date and competitor cutoff (2026-10-22) shown on the board.

## 2026-10-28
- RTX 4080 Ollama 0.40.11, vLLM 27B and FreeToken 27B runs.
- The RTX 5090 Flash-Next run was removed as outdated.

## 2026-10-27
- Records require the shared streaming client. RTX 5090 AlphaLlama and NInfer speeds timed by each engine's own tool
  are shown on charts, marked, but are not records.

## 2026-10-24
- Rules v2: exact and fast boards, the Speed Index fit, coverage, categories and failure reasons.
- Reproducibility across restarts became a badge instead of a gate.
