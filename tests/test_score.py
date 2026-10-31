"""InferArena rules v2 scoring: the fit, ties, the provisional rule, coverage, categories, and the 2026-10 lab edition.

    python tests/test_score.py
"""
import json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inferarena import score as sc  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  -> " + str(detail)) if detail and not cond else ""))
    if not cond:
        fails.append(name)


# ── cell value ──
sets = {"A": {"prompts": 4, "scored": True}, "V": {"prompts": 14, "scored": True}, "R": {"prompts": 7, "scored": False}}
v = sc.cell_value({"A": 100.0, "V": 200.0, "R": 9999.0}, sets)
check("cell value = prompt-weighted geomean, unscored set ignored",
      abs(v - math.exp((4 * math.log(100) + 14 * math.log(200)) / 18)) < 1e-9, v)
check("cell value needs every scored set", sc.cell_value({"A": 100.0}, sets) is None)

# ── the fit recovers known strengths (synthetic: value = strength × difficulty), with missing cells ──
true = {"ref": 1.0, "x": 3.0, "y": 1.5, "z": 1.2}
diff = {"c1": 100, "c2": 40, "c3": 250}
present = {"c1": ["ref", "x", "y", "z"], "c2": ["ref", "x", "y"], "c3": ["ref", "z"]}
vals = {c: {e: true[e] * diff[c] for e in es} for c, es in present.items()}
idx = sc.speed_index(vals, "ref")
check("fit recovers strengths exactly on consistent data, missing cells and all",
      all(abs(idx[e] - true[e]) < 1e-6 for e in true), idx)
check("single-engine cells carry no comparison", sc.speed_index({"solo": {"x": 5.0}}, "x") == {})
check("engines not linked to the reference get no index",
      "p" not in sc.speed_index(dict(vals, island={"p": 10.0, "q": 20.0}), "ref"))

# ── ranking: provisional rule and ties ──
r = sc.rank({"a": 3.0, "b": 2.0, "c": 1.97, "d": 1.5, "e": 9.0}, {"a": 3, "b": 2, "c": 2, "d": 2, "e": 1}, 2, 0.03)
by = {x["engine"]: x for x in r}
check("fewer than min cells → provisional, no rank", by["e"]["provisional"] and by["e"]["rank"] is None)
check("ranks follow the index", by["a"]["rank"] == 1 and by["b"]["rank"] == 2 and by["d"]["rank"] == 4, r)
check("within 3% shares the rank", by["c"]["rank"] == 2, r)

# ── coverage: per cell (GPU × model), any mode counts ──
runs = [{"engine": "a", "gpu": "g", "model": "m1", "status": "ran"},
        {"engine": "a", "gpu": "g", "model": "m2", "status": "failed_start"},
        {"engine": "a", "gpu": "g", "model": "m3", "status": "not_supported"},
        {"engine": "a", "gpu": "g", "model": "m4", "status": "env_broken"},
        {"engine": "b", "gpu": "g", "model": "m1", "status": "env_broken"},
        {"engine": "c", "gpu": "g", "model": "m1", "status": "not_supported", "class": "exact"},
        {"engine": "c", "gpu": "g", "model": "m1", "status": "ran", "class": "fast"}]
cov = sc.coverage(runs)
check("coverage: env_broken excluded, failures and unsupported count as tried", cov["a"] == {"ran": 1, "tried": 3}, cov)
check("coverage: an engine with only env_broken shows 0/0", cov["b"] == {"ran": 0, "tried": 0}, cov)
check("coverage: a fast-only engine still runs on that cell", cov["c"] == {"ran": 1, "tried": 1}, cov)

# ── reference fallback ──
check("reference = the preferred engine when it ran", sc.pick_reference("ref", vals) == "ref")
check("no preferred engine: the best-covered one, ties to the slower",
      sc.pick_reference("absent", {"c1": {"x": 300.0, "y": 100.0}}) == "y")
check("no comparison: no reference", sc.pick_reference("x", {"c1": {"x": 1.0}}) is None)

# ── the 2026-10 lab edition, by category ──
ed = json.loads((ROOT / "data" / "editions" / "2026-10.json").read_text(encoding="utf-8"))
out = sc.score_edition(ed)
cats = {c["id"]: c for c in out["categories"]}
check("Flash-Next withheld (2026-10-30): three published categories, in the edition's order",
      list(cats) == ["overall", "general", "offload_35b_27b"], list(cats))
pub = json.dumps(out)
check("withheld: no Flash-Next model, cell, run, record or peak is published, and no Strata row (its only rows were Flash-Next)",
      "qwen3.8-flash-next" not in pub and not any(r["engine"] == "strata" for c in out["categories"] for r in c["runs"])
      and not any(x["model"] == "qwen3.8-flash-next" for x in out["records"]["rows"] + out["records"]["peaks"]))
check("withheld: an engine whose only rows were withheld (Strata) isn't published at all",
      "strata" not in pub.lower(), [k for k in ("engine_scope",) if "strata" in json.dumps(out[k]).lower()])
check("withheld: the Flash-Next caveats and cutoff exception are held back with it",
      not any("Flash-Next" in c for c in out["caveats"]) and "Strata" not in out["cutoff"]["about"])
# The withheld rows stay in the file: score the full edition to keep checking them for when they come back.
full = sc.score_edition(dict(ed, withheld=None))
full_cats = {c["id"]: c for c in full["categories"]}
check("full edition (withheld list cleared): Flash-Next comes back as the fourth category",
      list(full_cats) == ["overall", "general", "offload_35b_27b", "offload_flash_next"], list(full_cats))


def ranking(cid, klass="plain"):
    return {x["engine"]: x for x in cats[cid][klass]["ranking"]}


ov = ranking("overall")
# After the RTX 3090 re-run (2026-10-27): Ollama 0.40.11 is ~1.6x faster there, and AlphaLlama 27B left the exact board.
# 2026-10-28: AlphaLlama 35B/27B on the 3090 re-run on its newer backend (27B back on the exact board).
for e, (ix, rk) in {"alphallama": (3.00, 1), "ninfer": (1.44, 2), "freetoken": (1.05, 3), "vllm": (1.05, 3),
                    "ollama": (1.00, 5)}.items():
    check(f"overall: {e} {ix}x rank {rk}", e in ov and abs(ov[e]["index"] - ix) < 0.01 and ov[e]["rank"] == rk, ov.get(e))
check("overall: sglang provisional (1 scored cell)", ov["sglang"]["provisional"])

gen = ranking("general")
check("general-purpose: only engines whose scope is 'general'", set(gen) == {"alphallama", "vllm", "sglang", "ollama"}, set(gen))
check("general-purpose: AlphaLlama 1st, vLLM and Ollama share 2nd (within 3%)",
      gen["alphallama"]["rank"] == 1 and gen["vllm"]["rank"] == 2 and gen["ollama"]["rank"] == 2, gen)

off = ranking("offload_35b_27b")
# 2026-10-29: the 4070 laptop 35B re-run (newer backend, residency reset) repeats 4/4, so the offload board has no badge.
check("35B/27B offload: AlphaLlama ranked 1st, every exact run reproducible across starts",
      off["alphallama"]["rank"] == 1 and off["alphallama"]["reproducible"] is True
      and abs(off["alphallama"]["index"] - 2.25) < 0.01, off.get("alphallama"))
# The 4080 protocol rerun reproduces plain decoding 4/4; the badge now comes from the 4070 laptop row alone.
r4080 = [r for r in ed["runs"] if r["gpu"] == "rtx4080" and r["engine"] == "alphallama" and r["class"] == "exact"
         and r["status"] == "ran" and r["model"] != "qwen3.8-flash-next"]
check("4080: our 35B and 27B exact-class runs (both plain decoding) reproduce across starts",
      sorted(r["model"] for r in r4080) == ["qwen3.6-35b-a3b", "qwen3.8-27b"] and all(r["reproducible"] for r in r4080), r4080)
check("35B/27B offload: Ollama ranked on all three cells (4080 35B + 27B, 4070 laptop)",
      cats["offload_35b_27b"]["plain"]["min_cells"] == 2 and off["ollama"]["rank"] == 2 and off["ollama"]["cells"] == 3, off)
check("35B/27B offload: freetoken provisional (finished one GPU only)", off["freetoken"]["provisional"], off["freetoken"])
cv_off = cats["offload_35b_27b"]["coverage"]
# freetoken: 4080 35B ran, 4070 failed mid-test (tried), 4080 27B env_broken (no checkpoint: not counted)
# freetoken: 4080 35B ran, 4070 failed mid-test, 4080 27B failed to start (both count as tried)
check("a run that failed mid-test counts as tried, not ran", cv_off["freetoken"] == {"ran": 1, "tried": 3}, cv_off)
check("an edition-wide check: no derived numbers left (all from per-prompt files)",
      not any(r.get("derived") for r in ed["runs"]))
fl = sc.coverage([{"engine": "x", "gpu": "g", "model": "m", "status": "failed_run"}])
check("failed_run is a recognised status", fl["x"] == {"ran": 0, "tried": 1}, fl)

fn = full_cats["offload_flash_next"]
check("R is scored as a general rule: every category scores A + V + R",
      all(c["scored_sets"] == ["A", "V", "R"] for c in out["categories"]), [c["scored_sets"] for c in out["categories"]])
fnp = {x["engine"]: x for x in fn["plain"]["ranking"]}
fnf = {x["engine"]: x for x in fn["fast"]["ranking"]}
# Flash-Next: the RTX 5090 run is outdated and was removed on 2026-10-28; the 3090 is the only cell until the 4080 lands.
check("Flash-Next: RTX 3090 and 4080 only, no 5090 rows or cell", not any(r["gpu"] == "rtx5090" for r in fn["runs"])
      and sorted(c["gpu"] for c in fn["cells"]) == ["rtx3090", "rtx4080"], [c["gpu"] for c in fn["cells"]])
check("Flash-Next plain: no comparison yet (AlphaLlama is the only engine with exact decoding there)",
      fn["plain"]["reference"] is None and not fn["plain"]["ranking"], fn["plain"])
check("Flash-Next fast: AlphaLlama and Strata share 1st (within 3%); 4080 has Strata 0.2.7.1 vs AlphaLlama's release build",
      fnf["strata"]["rank"] == 1 and fnf["alphallama"]["rank"] == 1 and abs(fnf["alphallama"]["index"] - 1.00) < 0.01
      and abs(fnf["strata"]["index"] - 1.014) < 0.01, fnf)
check("Flash-Next coverage: Strata runs on both cells, 3090 and 4080 (fast mode only)",
      fn["coverage"]["strata"] == {"ran": 2, "tried": 2}, fn["coverage"].get("strata"))
check("Flash-Next 3090: FreeToken not supported there (126 GB file > 62 GB RAM + 24 GB VRAM), AlphaLlama badged not reproducible",
      any(r["gpu"] == "rtx3090" and r["engine"] == "freetoken" and r["status"] == "not_supported" for r in fn["runs"])
      and any(r["gpu"] == "rtx3090" and r["engine"] == "alphallama" and r["class"] == "exact" and r["reproducible"] is False
              for r in fn["runs"]))
check("categories never mix regimes",
      all(all(r["regime"] == c["regime"] for r in c["runs"]) for c in out["categories"] if c.get("regime")))
check("every ran row carries all three scored sets", all(
    all(k in r for k in ("A", "V", "R")) for r in ed["runs"] if r.get("status") == "ran"))
rec = out["records"]
best = {r["engine"]: r for r in rec["best_per_engine"]}
check("records: one best per engine, sorted fastest first",
      [r["tok_s"] for r in rec["best_per_engine"]] == sorted((r["tok_s"] for r in rec["best_per_engine"]), reverse=True)
      and len(best) == len(rec["best_per_engine"]), rec["best_per_engine"])
# Records need the shared streaming client: the 5090 AlphaLlama/NInfer arms were CLI-timed (audit, 2026-10-27).
check("records: no CLI-timed 5090 AlphaLlama/NInfer rows (shared-client rule applies to us too)",
      not any(r["gpu"] == "rtx5090" and r["engine"] in ("alphallama", "ninfer") and r["model"] != "qwen3.8-flash-next"
              for r in rec["rows"]))
check("records: AlphaLlama's best shared-client record is the 3090 35B fast run on its newer backend (1,175.2)",
      best["alphallama"]["tok_s"] == 1175.2 and best["alphallama"]["gpu"] == "rtx3090", best.get("alphallama"))
check("records: the 4080 27B thinking-token artifact (84.1) is gone",
      not any(r["gpu"] == "rtx4080" and r["model"] == "qwen3.8-27b" and r["engine"] == "alphallama" and r["tok_s"] > 60
              and r["class"] == "exact" for r in rec["rows"]))
# 4080 27B (2026-10-29, newer backend, one configuration): plain is exact class; verified MTP matched plain only 1/4 there.
r27 = {(r["class"], r.get("variant")): r for r in ed["runs"] if r["gpu"] == "rtx4080" and r["model"] == "qwen3.8-27b"
       and r["engine"] == "alphallama"}
check("4080 27B: plain decoding is the exact-class row; verified MTP (1/4 vs plain) and the speed mode are on the fast board",
      r27[("exact", None)]["A"] == 22.9 and not r27[("exact", None)].get("drafting")
      and r27[("fast", "verified MTP")]["identity"] == "1/4" and ("fast", None) in r27, list(r27))
check("AlphaLlama 27B on the 3090: exact run identical to its plain decode (4/4), with verified drafting",
      any(r["model"] == "qwen3.8-27b" and r["engine"] == "alphallama" and r["class"] == "exact" and r["status"] == "ran"
          and r["gpu"] == "rtx3090" and r["identity"] == "4/4" and r.get("drafting") for r in ed["runs"]))
check("records: every row repeated (>= 2 passes within 10%)",
      all(r["passes"] >= 2 and r["spread_pct"] <= 10 for r in rec["rows"]), rec["rows"][:3])

# ── Peak speed by task type ──
pk = rec["peaks"]
pks = [p for p in pk if p.get("timing") != "engine"]   # shared-client peaks only
check("peak speed: every engine's top shared-client peak equals its overall record",
      all(max(p["tok_s"] for p in pks if p["engine"] == e) == r["tok_s"] for e, r in best.items()),
      {e: (max(p["tok_s"] for p in pks if p["engine"] == e), r["tok_s"]) for e, r in best.items()})
check("peak speed: one peak per GPU, model, engine and task type (never pooled across hardware or models), all repeatable",
      len({(p["gpu"], p["model"], p["engine"], p["task_type"]) for p in pk}) == len(pk)
      and all(p["passes"] >= 2 and p["spread_pct"] <= 10 for p in pk))
et = {(p["gpu"], p["engine"]) for p in pk if p.get("timing") == "engine"}
check("peak speed: 5090 AlphaLlama/NInfer are shown on their charts but flagged engine-timed, and never records",
      et == {("rtx5090", "alphallama"), ("rtx5090", "ninfer")}
      and not any(r["gpu"] == "rtx5090" and r["engine"] in ("alphallama", "ninfer") and r["model"] != "qwen3.8-flash-next"
                  for r in rec["rows"]), et)
check("peak speed: every task type is a declared one", {p["task_type"] for p in pk} <= {t["id"] for t in rec["task_types"]})

print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
