# InferArena rules (v2)

These rules apply to every engine, including AlphaLlama. Changes are recorded in `CHANGELOG.md`.

## Comparisons
1. Speeds are only compared on the same GPU and the same model (a *cell*).
2. Models that fit in GPU memory and models that must be partly offloaded to system memory are scored separately.
3. Categories rank only their own GPUs, models and engines: Overall (every engine), General-purpose (engines that load
   any model architecture) and one board per offload group.

## Measurement
4. Every engine gets the same prompts and the same client: temperature 0, top-k 1, thinking off, a fixed seed, no
   engine-specific settings, OpenAI-compatible streaming.
5. Decode speed is the speed you see while an answer streams: tokens after the first, divided by the time after the
   first token. A set's score is the geometric mean over its prompts, so no single prompt dominates.
6. Every prompt runs twice, the second pass in reversed order. A row whose streamed token count is under 90% of the
   count the engine reports is left out, unless the engine sends several tokens per streamed chunk: such a row counts
   when it streamed no thinking tokens and its visible text is as long as the reported count implies (at least 90% of
   the median text per token of the other replies to the same prompt).
7. Each engine runs its own best public configuration and model format for the GPU. Formats are labelled on every
   result, not equalised.

## Exact and fast decoding
8. The main board counts exact decoding: output identical to the engine's own standard greedy decode, including
   lossless speed-ups such as drafts verified token by token.
9. Whether a run reproduces bit for bit across restarts is shown as a badge; it does not exclude the run.
10. Modes that change the output are ranked on a separate fast board, labelled with how many outputs match the
    engine's exact decode.

## Scoring
11. The Speed Index fits every engine against every other on the cells they share: log(value) = engine strength + cell
    difficulty, normalised so the reference engine is 1.0.
12. Engines within 3% of each other share a rank. Engines with fewer than 2 scored cells are provisional.
13. Coverage ("runs on k/N") is separate from speed. Failures stay on the board with their reason. Cells where the
    test setup itself was broken don't count against an engine.

## Records
14. A record needs at least 2 passes within 10% of each other, measured through the shared client. Speeds timed by an
    engine's own tool may be shown on charts, marked, but are never records.

## Publication
15. A person reviews every submitted run before it is published. Automatic checks flag speeds beyond the GPU's memory
    bandwidth, models too big for the claimed mode and unknown hardware; flags inform the reviewer and don't reject a
    run on their own.
16. Anyone, including engine maintainers, can challenge a number by submitting their own run. Disagreements are shown
    on the board.
17. A model's results may be withheld from the published board while new releases of the engines on it are measured.
    Withheld rows stay in the edition file and every withholding is recorded in `CHANGELOG.md`.
