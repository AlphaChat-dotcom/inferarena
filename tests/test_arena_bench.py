"""InferArena harness: the maths is right, requests are engine-neutral, and the output is accepted
by the server's own validator.

    python tests/test_arena_bench.py

Uses an in-process fake OpenAI-compatible server (httpx.MockTransport) and a fake clock that
advances 10 ms per tick, so every expected number is exact.
"""
import json, os, pathlib, sys, types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT))

import httpx                                   # noqa: E402
import arena_bench as ab                       # noqa: E402

N_CHUNKS = 200
seen_bodies = []


def sse(chunks, usage):
    lines = []
    for c in chunks:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": c}}]}))
    if usage is not None:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def handler_factory(report_usage=True):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_bodies.append(body)
        prompt = body["messages"][0]["content"]
        pt = len(prompt) // 4
        if not body.get("stream"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}],
                                             "usage": {"prompt_tokens": pt, "completion_tokens": 1}})
        n = min(body["max_tokens"], N_CHUNKS)
        usage = {"prompt_tokens": pt, "completion_tokens": n} if report_usage else None
        return httpx.Response(200, content=sse([f"tok{i} " for i in range(n)], usage),
                              headers={"content-type": "text/event-stream"})
    return handler


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.01
        return self.t


def bench(report_usage=True, clock=None):
    client = httpx.Client(transport=httpx.MockTransport(handler_factory(report_usage)))
    return ab.Bench("http://fake/v1", "llama3.1:8b", client=client, clock=clock or Clock())


fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  -> " + str(detail)) if detail and not cond else ""))
    if not cond:
        fails.append(name)


# ── one streamed request ─────────────────────────────────────────────────────────────────────────
b = bench()
r = b.stream_once("hello " * 100, 256)
check("text reassembled from the stream", r["text"].startswith("tok0 tok1 ") and r["text"].count("tok") == N_CHUNKS)
check("completion tokens taken from usage", r["completion_tokens"] == N_CHUNKS and r["usage_reported"])
check("TTFT is one clock tick", abs(r["ttft_s"] - 0.01) < 1e-9, r["ttft_s"])
check("decode spans first..last token", abs(r["decode_s"] - 0.01 * (N_CHUNKS - 1)) < 1e-9, r["decode_s"])
m = ab._metrics(r, 512)
check("decode tok/s = (n-1)/decode time = 100", abs(m["decode_tps"] - 100.0) < 1e-6, m["decode_tps"])
check("prefill tok/s = prompt tokens / TTFT", abs(m["prefill_tps"] - r["prompt_tokens"] / 0.01) < 1e-6, m)

r = bench(report_usage=False).stream_once("hi", 256)
check("no usage in stream -> count chunks instead", r["completion_tokens"] == N_CHUNKS and not r["usage_reported"])

# ── requests are engine-neutral and greedy ──────────────────────────────────────────────────────
streamed = [x for x in seen_bodies if x.get("stream")]
allowed = {"model", "messages", "max_tokens", "temperature", "seed", "stream", "stream_options"}
check("only standard OpenAI fields are sent (no engine-specific params)",
      all(set(x) <= allowed for x in seen_bodies), [sorted(set(x) - allowed) for x in seen_bodies])
check("greedy: temperature 0 and fixed seed", all(x["temperature"] == 0 and x["seed"] == ab.SEED for x in seen_bodies))

# ── prompt construction ─────────────────────────────────────────────────────────────────────────
p1, p2 = ab.build_prompt(512, 0.25, "aaa"), ab.build_prompt(512, 0.25, "bbb")
check("salt is the very first thing (defeats prefix caching)", p1.startswith("[run aaa]") and p1[:9] != p2[:9])
check("prompt length tracks the token target", abs(len(p1) * 0.25 - 512) < 5, len(p1) * 0.25)
check("unsalted prompt is deterministic", ab.build_prompt(512, 0.25) == ab.build_prompt(512, 0.25))
check("calibration reads tokens/char from usage", abs(bench().calibrate() - 0.25) < 0.01)

# ── whole workloads ─────────────────────────────────────────────────────────────────────────────
b = bench()
short = ab.run_workload(b, "short", 0.25, repeats=3)
check("short workload: median of repeats", short["repeats"] == 3 and abs(short["decode_tps"] - 100.0) < 0.01, short)
conc = ab.run_workload(bench(clock=None), "concurrent", 0.25, repeats=1)
check("concurrent workload has aggregate tok/s and concurrency 8",
      conc.get("aggregate_tps", 0) > 0 and conc["concurrency"] == 8, conc)
fp1, fp2 = ab.output_fingerprint(bench(), 0.25), ab.output_fingerprint(bench(), 0.25)
check("output fingerprint is a stable sha256", fp1 == fp2 and len(fp1) == 64)

# ── contract: the server accepts what the harness writes ───────────────────────────────────────
args = ab.parse_args(["--base-url", "http://fake/v1", "--model", "llama3.1:8b", "--engine", "ollama",
                      "--engine-version", "0.6.2", "--quant", "Q4_K_M", "--mode", "full_vram"])
hw = {"gpu_name": "NVIDIA GeForce RTX 4090", "gpu_vram_gb": 23.99, "gpu_count": 1, "driver": "560.0"}
out = ab.build_result(args, {"short": short, "concurrent": conc}, fp1, hw, 9.5)
json.loads(json.dumps(out))   # serialisable
try:
    from inferarena.validate import ArenaResult
    ArenaResult.model_validate(out)
    ok, err = True, ""
except Exception as e:      # noqa: BLE001
    ok, err = False, str(e)[:300]
check("server-side ArenaResult validates the harness output", ok, err)
from inferarena.validate import SUPPORTED_HARNESS_MAJORS, SUITE_WORKLOADS, plausibility_flags  # noqa: E402
check("harness and server agree on schema/suite versions",
      out["schema_version"] == 1 and out["suite"] == "v1"
      and out["harness_version"].split(".")[0] in SUPPORTED_HARNESS_MAJORS)

# ── suite v2 (InferArena rules v2) ─────────────────────────────────────────────────────────────
import prompts_v2  # noqa: E402
check("v2 prompt sets: A = 4 short, V = 14 held-out, R = 7 headline, sets match the server",
      len(prompts_v2.SETS["A"]["prompts"]) == 4 and len(prompts_v2.SETS["V"]["prompts"]) == 14
      and len(prompts_v2.SETS["R"]["prompts"]) == 7
      and tuple(sorted(prompts_v2.SETS)) == tuple(sorted(SUITE_WORKLOADS["v2"])))
check("v2 prompt ids are unique", len({p for s in prompts_v2.SETS.values() for p, _ in s["prompts"]}) == 25)
check("v2 prompts fit their context (≤ 8 KB input)", all(len(t) <= 8192 for s in prompts_v2.SETS.values() for _, t in s["prompts"]))

seen_bodies.clear()
b2 = bench()
b2.extra_body = dict(ab.V2_EXTRA_BODY)
b2.count_reasoning = True
spec = {"max_tokens": 64, "prompts": [("p1", "one " * 40), ("p2", "two " * 80), ("p3", "three " * 20)]}
m, hashes, repro = ab.run_set_v2(b2, "X", spec, passes=2)
order = [bb["messages"][0]["content"].split()[0] for bb in seen_bodies]
check("v2: pass 1 in order, pass 2 reversed", order == ["one", "two", "three", "three", "two", "one"], order)
check("v2: greedy + thinking-off hints sent, identical for every engine",
      all(bb.get("top_k") == 1 and bb.get("chat_template_kwargs") == {"enable_thinking": False} for bb in seen_bodies))
check("v2: decode is the geomean over prompts (all 100 tok/s here)", abs(m["decode_tps"] - 100.0) < 0.01, m)
check("v2: per-prompt decode reported, prompts and passes counted",
      set(m["per_prompt_decode_tps"]) == {"p1", "p2", "p3"} and m["prompts"] == 3 and m["repeats"] == 2, m)
check("v2: identical outputs across passes → reproducible", repro and len(hashes) == 3 and all(len(h) == 64 for h in hashes))
check("v2: geomean helper", abs(ab._geomean([1.0, 100.0]) - 10.0) < 1e-9)

ref_path = ROOT / "tests" / "_tmp_plain_ref.json"
ref_path.write_text(json.dumps({"output_hashes": {"A": hashes[:2] + ["0" * 64]}}), encoding="utf-8")
try:
    ident = ab.identity_vs(str(ref_path), hashes)
finally:
    ref_path.unlink()
check("v2: identity vs the plain run counts matching outputs", ident == {"identical": 2, "of": 3}, ident)

args2 = ab.parse_args(["--base-url", "http://fake/v1", "--model", "qwen3.6-35b-a3b", "--engine", "ollama",
                       "--engine-version", "0.40.8", "--quant", "Q4_K_M", "--mode", "full_vram"])
check("v2 is the default suite, exact the default class", args2.suite == "v2" and args2.decode_class == "exact")
setA, hA, _ = ab.run_set_v2(bench(), "A", {"max_tokens": 64, "prompts": prompts_v2.SETS["A"]["prompts"]}, passes=2)
setV, hV, _ = ab.run_set_v2(bench(), "V", {"max_tokens": 64, "prompts": prompts_v2.SETS["V"]["prompts"][:3]}, passes=2)
setR, hR, _ = ab.run_set_v2(bench(), "R", {"max_tokens": 64, "prompts": prompts_v2.SETS["R"]["prompts"][:2]}, passes=2)
out2 = ab.build_result_v2(args2, {"A": setA, "V": setV, "R": setR}, {"A": hA, "V": hV, "R": hR}, True, None, hw, 20.1)
json.loads(json.dumps(out2))
try:
    r2 = ArenaResult.model_validate(out2)
    ok2, err2 = True, ""
except Exception as e:      # noqa: BLE001
    r2, ok2, err2 = None, False, str(e)[:300]
check("server-side ArenaResult validates a v2 result", ok2 and out2["suite"] == "v2" and out2["schema_version"] == 2, err2)
check("v2 result carries hashes only, never output text",
      "output_hashes" in out2 and "text" not in json.dumps(out2["workloads"]))
bad_v2 = dict(out2, workloads={"A": setA, "V": setV})
try:
    ArenaResult.model_validate(bad_v2)
    check("server refuses a v2 result without set R (R is scored)", False)
except Exception:           # noqa: BLE001
    check("server refuses a v2 result without set R (R is scored)", True)
if r2 is not None:
    nr = ArenaResult.model_validate(dict(out2, reproducible=False))
    mm = ArenaResult.model_validate(dict(out2, identity={"identical": 3, "of": 4}))
    fast = ArenaResult.model_validate(dict(out2, decode_class="fast", identity={"identical": 1, "of": 4}))
    check("exact run not reproducible → flagged not_reproducible", "not_reproducible" in plausibility_flags(nr))
    check("exact run differing from its plain run → flagged exact_output_mismatch",
          "exact_output_mismatch" in plausibility_flags(mm))
    check("fast run: no exact flags, no bandwidth-bound flag",
          not any(f.startswith(("exact_", "not_repro", "exceeds_bandwidth")) for f in plausibility_flags(fast)))

fail_args = ab.parse_args(["--base-url", "http://fake/v1", "--model", "qwen3.8-27b", "--engine", "freetoken",
                           "--engine-version", "0.1.2", "--quant", "NVFP4", "--mode", "full_vram",
                           "--record-status", "failed_start", "--reason", "out of memory at start"])
rec = ab.failure_record(fail_args, hw)
check("v2 coverage record for an engine that could not run",
      rec["status"] == "failed_start" and rec["reason"] and "workloads" not in rec, rec)

print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
