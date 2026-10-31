# Contributing

## Submitting a benchmark run
1. Run `harness/arena_bench.py` against your engine on your GPU. Use the engine's best public configuration and say
   which one in the notes.
2. Upload `arena_result.json` on the Submit tab at [alphachat.com/inferarena](https://alphachat.com/inferarena#submit).
3. A reviewer checks the run. Approved runs appear on the community leaderboard.

## Challenging a number
Open an issue with the GPU, model, engine and version, what you measured, and how. If you can, attach your
`arena_result.json`. Engine maintainers are welcome to review how their engine is configured and to propose a better
public configuration.

## Changing the rules or the scoring
Open a pull request with the change and a line for `CHANGELOG.md`. Changes to scoring need tests in
`tests/test_score.py`. Rule changes apply to every engine, including AlphaLlama.

## Code
- Python 3.10+. The harness uses the standard library only; the validator uses `pydantic`.
- Run `python tests/test_score.py` and `python tests/test_arena_bench.py` before sending a pull request.
