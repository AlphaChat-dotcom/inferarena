"""
InferArena scoring, rules v2 — one set of pure functions for lab editions and (later) community runs.

- A *cell* is (GPU, model). Speeds are compared only inside a cell.
- A cell's value for an engine is the geometric mean over the scored prompts (set geomeans weighted by prompt count).
- The Speed Index is a two-way fit, log(value) = engine strength + cell difficulty, over the plain-decoding runs of one
  category (categories never mix regimes: an offload cell where the reference offloads badly would shift every
  engine's index), normalised so the reference engine = 1.0; a category without it uses its best-covered engine.
- Plain decoding (no draft tokens) is ranked whether or not it reproduced bit-for-bit across restarts — that is a
  per-row badge, not a gate. Speculative / multi-token modes are a separate "fast" board.
- Coverage ("runs on k/N") is separate from speed: N counts every cell the engine was tried in, except cells where our
  own setup was broken (env_broken = not yet measured).
- Nothing here knows an engine, GPU or model name.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

# failed_run = started but didn't finish the scored prompts (e.g. every held-out prompt timed out)
STATUSES = ("ran", "failed_start", "failed_run", "not_supported", "env_broken")


def cell_key(run: dict) -> str:
    return f"{run['gpu']}/{run['model']}"


def cell_value(run: dict, sets: dict) -> Optional[float]:
    """Prompt-weighted geometric mean over the scored sets; None if any scored set is missing."""
    num = den = 0.0
    for name, spec in sets.items():
        if not spec.get("scored"):
            continue
        v = run.get(name)
        if not isinstance(v, (int, float)) or v <= 0:
            return None
        n = float(spec.get("prompts") or 1)
        num += n * math.log(v)
        den += n
    return math.exp(num / den) if den else None


def speed_index(values: dict[str, dict[str, float]], reference: str, iterations: int = 2000) -> dict[str, float]:
    """values[cell][engine] → {engine: index vs reference}. Cells with a single engine carry no comparison and are
    ignored. Engines not connected to the reference through shared cells get no index."""
    cells = {c: d for c, d in values.items() if len(d) >= 2}
    engines = sorted({e for d in cells.values() for e in d})
    if reference not in engines:
        return {}
    # Engines connected to the reference through shared cells (otherwise the fit has no anchor for them).
    linked, frontier = {reference}, [reference]
    while frontier:
        e = frontier.pop()
        for d in cells.values():
            if e in d:
                for other in d:
                    if other not in linked:
                        linked.add(other)
                        frontier.append(other)
    s = {e: 0.0 for e in linked}
    h = {c: 0.0 for c in cells}
    for _ in range(iterations):
        for c, d in cells.items():
            obs = [math.log(t) - s[e] for e, t in d.items() if e in s]
            h[c] = sum(obs) / len(obs) if obs else 0.0
        for e in s:
            obs = [math.log(d[e]) - h[c] for c, d in cells.items() if e in d]
            s[e] = sum(obs) / len(obs)
        ref = s[reference]
        s = {e: v - ref for e, v in s.items()}
    return {e: math.exp(v) for e, v in s.items()}


def head_to_head(values: dict[str, dict[str, float]]) -> list[dict]:
    """Geometric-mean speed ratio for every engine pair, on the cells both ran."""
    engines = sorted({e for d in values.values() for e in d})
    out = []
    for i, a in enumerate(engines):
        for b in engines[i + 1:]:
            shared = [c for c, d in values.items() if a in d and b in d]
            if shared:
                r = math.exp(sum(math.log(values[c][a] / values[c][b]) for c in shared) / len(shared))
                out.append({"a": a, "b": b, "ratio": round(r, 3), "cells": len(shared)})
    return out


def coverage(runs: Iterable[dict]) -> dict[str, dict]:
    """{engine: {ran, tried}} counted per cell (GPU × model), whatever the decoding class: an engine "runs on" a cell if
    any of its modes finished there. tried excludes env_broken (our setup failed; not an engine verdict)."""
    best: dict[tuple[str, str], str] = {}
    order = {"ran": 4, "failed_run": 3, "failed_start": 2, "not_supported": 1, "env_broken": 0}
    for r in runs:
        st = r.get("status")
        if st not in STATUSES:
            continue
        k = (r["engine"], cell_key(r))
        if order[st] > order.get(best.get(k, ""), -1):
            best[k] = st
    out: dict[str, dict] = defaultdict(lambda: {"ran": 0, "tried": 0})
    for (engine, _), st in best.items():
        out[engine]  # noqa: B018 — make sure the engine appears
        if st != "env_broken":
            out[engine]["tried"] += 1
        if st == "ran":
            out[engine]["ran"] += 1
    return dict(out)


def rank(index: dict[str, float], verified_cells: dict[str, int], min_cells: int, tie_within: float) -> list[dict]:
    """Sorted by index. Engines with fewer than min_cells verified cells are 'provisional' and get no rank.
    Ranked engines within tie_within of the previous rank-holder share its rank."""
    rows = sorted(index.items(), key=lambda kv: -kv[1])
    out, last_rank, last_val, position = [], None, None, 0
    for engine, val in rows:
        n = verified_cells.get(engine, 0)
        if n < min_cells:
            out.append({"engine": engine, "index": round(val, 3), "cells": n, "rank": None, "provisional": True})
            continue
        position += 1
        if last_val is not None and last_val > 0 and abs(val - last_val) / last_val <= tie_within:
            r = last_rank
        else:
            r, last_val = position, val
        last_rank = r
        out.append({"engine": engine, "index": round(val, 3), "cells": n, "rank": r, "provisional": False})
    return out


def pick_reference(preferred: str, values: dict[str, dict[str, float]]) -> Optional[str]:
    """The edition's reference engine when it ran in these cells; otherwise the engine in the most comparative cells
    (ties: the slower one, so the index reads as 'times faster than the baseline'). None without a comparison."""
    comparative = {c: d for c, d in values.items() if len(d) >= 2}
    engines = {e for d in comparative.values() for e in d}
    if not engines:
        return None
    if preferred in engines:
        return preferred
    def key(e):
        cells = [c for c, d in comparative.items() if e in d]
        speed = math.exp(sum(math.log(comparative[c][e]) for c in cells) / len(cells))
        return (-len(cells), speed, e)
    return min(engines, key=key)


def _category_filter(cat: dict, scope: dict, regimes: dict):
    models = set(cat.get("models") or [])
    want_scope = cat.get("engine_scope")

    def keep(r: dict) -> bool:
        if cat.get("regime") and regimes.get(cell_key(r)) != cat["regime"]:
            return False
        if models and r["model"] not in models:
            return False
        if want_scope and scope.get(r["engine"]) != want_scope:
            return False
        return True
    return keep


def board(runs: list[dict], klass: str, sets: dict, reference: str, min_cells: int, tie_within: float) -> dict:
    """Ranking + head-to-head for one decoding class over the given runs (plain decoding is ranked whether or not it
    reproduced across restarts; that is reported per row, not used as a gate)."""
    vals: dict[str, dict[str, float]] = defaultdict(dict)
    not_repro: set[str] = set()
    for r in runs:
        if r.get("class", "exact") != klass or r.get("status") != "ran":
            continue
        v = cell_value(r, sets)
        if v:
            vals[cell_key(r)][r["engine"]] = v
            if r.get("reproducible") is False:
                not_repro.add(r["engine"])
    vals = dict(vals)
    ref = pick_reference(reference, vals)
    idx = speed_index(vals, ref) if ref else {}
    comparative = [c for c, d in vals.items() if len(d) >= 2]
    n_cells = {e: sum(1 for c in comparative if e in vals[c]) for e in idx}
    need = max(1, min(min_cells, len(comparative)))         # a one-cell category can still have a ranking
    ranking = rank(idx, n_cells, need, tie_within)
    for row in ranking:
        row["reproducible"] = row["engine"] not in not_repro
    return {"reference": ref, "ranking": ranking, "head_to_head": head_to_head(vals), "cells": sorted(vals),
            "min_cells": need}


def withhold(ed: dict) -> dict:
    """Drop what ed["withheld"] lists from everything published. "models": a model's cells, runs, records and peaks, its
    entry in the model list, and any category left with no model. "cells" ("gpu/model"): one GPU's results for one
    model, the same way. The rows stay in the edition file, so they come back by removing them from the list."""
    w = ed.get("withheld") or {}
    held = set(w.get("models") or [])
    held_cells = set(w.get("cells") or [])           # "gpu/model": one GPU's results for one model
    if not held and not held_cells:
        return ed

    def gone(x: dict) -> bool:
        return x.get("model") in held or f"{x.get('gpu')}/{x.get('model')}" in held_cells

    out = dict(ed)
    out["cells"] = [c for c in ed["cells"] if not gone(c)]
    out["runs"] = [r for r in ed["runs"] if not gone(r)]
    left = {r["engine"] for r in out["runs"]}
    scope = ed.get("engine_scope") or {}
    if scope.get("engines"):                     # an engine whose only rows were withheld isn't published either
        out["engine_scope"] = dict(scope, engines={e: v for e, v in scope["engines"].items() if e in left})
    out["models"] = [m for m in ed["models"] if m["id"] not in held]
    if ed.get("categories"):
        cats = []
        for c in ed["categories"]:
            if c.get("models"):
                kept = [m for m in c["models"] if m not in held]
                if not kept:
                    continue
                c = dict(c, models=kept)
            cats.append(c)
        out["categories"] = cats
    rec = dict(ed.get("records") or {})
    for k in ("rows", "peaks", "peak_notes"):
        if rec.get(k):
            rec[k] = [x for x in rec[k] if not gone(x)]
    out["records"] = rec
    return out


def score_edition(ed: dict) -> dict:
    """Everything the Lab page shows, computed from an edition file: one entry per category (a filter over cells,
    models and engine scope, with its own scored prompt sets), each with a plain board and a fast board."""
    sets, ref = ed["sets"], ed["reference_engine"]
    ed = withhold(ed)
    regimes = {f"{c['gpu']}/{c['model']}": c.get("regime", "full_vram") for c in ed["cells"]}
    scope = (ed.get("engine_scope") or {}).get("engines", {})
    runs = []
    for r in ed["runs"]:
        row = dict(r)
        row["cell"] = cell_key(r)
        row["regime"] = regimes.get(row["cell"], "full_vram")
        runs.append(row)

    categories = ed.get("categories") or [
        {"id": reg, "title": reg, "regime": reg} for reg in sorted(set(regimes.values()))]
    out_cats = []
    for cat in categories:
        cat_sets = {k: dict(v, scored=k in cat["scored_sets"]) for k, v in sets.items()}             if cat.get("scored_sets") else sets
        keep = _category_filter(cat, scope, regimes)
        cruns = [dict(r, value=(round(cell_value(r, cat_sets), 2) if r.get("status") == "ran"
                                and cell_value(r, cat_sets) else None)) for r in runs if keep(r)]
        cells = [c for c in ed["cells"] if any(r["cell"] == f"{c['gpu']}/{c['model']}" for r in cruns)]
        mc, tw = ed.get("min_cells_to_rank", 2), ed.get("tie_within", 0.03)
        out_cats.append({
            **{k: cat.get(k) for k in ("id", "title", "about", "regime", "models", "engine_scope", "summary")},
            "scored_sets": [k for k, v in cat_sets.items() if v.get("scored")],
            "cells": cells, "runs": cruns,
            "plain": board(cruns, "exact", cat_sets, ref, mc, tw),
            "fast": board(cruns, "fast", cat_sets, ref, mc, tw),
            "coverage": coverage(cruns),
        })
    return {
        "edition": ed["edition"], "published": ed.get("published"), "rules": ed.get("rules"),
        "results_as_of": ed.get("results_as_of"), "cutoff": ed.get("cutoff"),
        "reference_engine": ref, "sets": sets, "protocol": ed.get("protocol"),
        "models": ed["models"], "gpus": ed["gpus"], "caveats": ed.get("caveats", []),
        "engine_scope": ed.get("engine_scope"), "categories": out_cats,
        "records": records(ed.get("records") or {}),
    }


def records(rec: dict) -> dict:
    """Speed records: each engine's best repeatable single-prompt speed (computed upstream from per-prompt files).
    Returned as the best per engine (any GPU, model or mode) plus every engine × mode × GPU × model best."""
    rows = sorted(rec.get("rows") or [], key=lambda r: -r["tok_s"])
    top: dict[str, dict] = {}
    for r in rows:
        top.setdefault(r["engine"], r)
    # Peak speed by task type: each engine's best on each kind of work (computed upstream, passed through).
    return {"rule": rec.get("rule"), "best_per_engine": list(top.values()), "rows": rows,
            "task_types": rec.get("task_types") or [], "peaks": rec.get("peaks") or [],
            "peak_notes": rec.get("peak_notes") or []}
