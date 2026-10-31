# InferArena

An open benchmark for local LLM inference engines. Every engine runs the same models on the same GPUs, is driven by
the same streaming client, and is scored by the same code. The results are published at
[alphachat.com/inferarena](https://alphachat.com/inferarena), and everything needed to check them is in this
repository.

InferArena is run by the AlphaChat team, who also make one of the engines it ranks (AlphaLlama). That is a conflict of
interest, so the rules, the scoring code, the per-prompt data and every rule change are public, and AlphaLlama is
measured exactly like everyone else. If you think a number is wrong, open an issue or submit your own run.

## What's here

| Path | What it is |
|---|---|
| `harness/` | The benchmark client (`arena_bench.py`) and the public prompt suite (`prompts_v2.py`). Python 3.10+, standard library only. |
| `inferarena/score.py` | Scoring: cell values, the Speed Index fit, ties, coverage and categories. Pure functions. |
| `inferarena/validate.py` | The checks a submitted result goes through: schema, versions and plausibility flags. |
| `data/editions/` | Each published edition: every run, its version, format, status and notes. |
| `data/raw/` | Per-prompt, per-pass decode speeds behind each edition. |
| `data/gpus.json` | Memory bandwidth per GPU, used by the plausibility checks. |
| `RULES.md` | The rules every number follows. |
| `CHANGELOG.md` | Every change to the rules, the data and what is published. |

## Run the harness against your engine

```bash
pip install -r requirements.txt
python harness/arena_bench.py --help
```

The harness talks to any OpenAI-compatible endpoint. It sends identical requests to every engine (temperature 0,
top-k 1, thinking off, a fixed seed, no engine-specific settings) and writes `arena_result.json`. See
`harness/README.md` for the full options.

## Submit a result

Upload `arena_result.json` on the Submit tab at
[alphachat.com/inferarena#submit](https://alphachat.com/inferarena#submit). A reviewer checks every run before it is
published; automatic checks flag speeds beyond the GPU's memory bandwidth, models too big for the claimed mode and
unknown hardware. See `CONTRIBUTING.md`.

## How scoring works

- Speeds are only compared on the same GPU and the same model (a *cell*).
- A cell's value is the geometric mean of decode speed over the scored prompts.
- The Speed Index fits every engine against every other on the cells they share, with the reference engine at 1.0.
  Engines within 3% share a rank; engines with fewer than 2 scored cells are provisional.
- Exact decoding (output identical to the engine's own greedy decode) and fast modes that change the output are ranked
  on separate boards.
- Failures stay on the board with their reason.

The full rules are in `RULES.md`.

## Prompt sets

The public suite in `harness/prompts_v2.py` is what anyone can run. The team's lab editions also use held-out prompt
sets of the same kinds and sizes. Some of those prompts contain source code that isn't public, so they are not
published here. Results on them are still listed per prompt in `data/raw/`, by task description.

## Run the tests

```bash
python tests/test_score.py
python tests/test_arena_bench.py
```

## License

MIT. See `LICENSE`.
