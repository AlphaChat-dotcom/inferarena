#!/usr/bin/env python3
"""InferArena benchmark harness — measure any local inference engine, the same way.

    pip install httpx
    python arena_bench.py --base-url http://127.0.0.1:11434/v1 --model llama3.1:8b \\
        --engine ollama --engine-version 0.6.2 --quant Q4_K_M --mode full_vram

It drives the engine's OpenAI-compatible streaming endpoint (/v1/chat/completions), which vLLM,
SGLang, Ollama, llama.cpp-server, AlphaLlama and most others expose. There is NO engine-specific
code and no engine-specific request parameter: every engine gets byte-identical requests, so the
numbers are comparable. Timings are taken on the client, so they include local HTTP overhead —
the same for every engine.

Suite v2 (the default; InferArena rules v2) runs the public prompt set in prompts_v2.py (download
it next to this file): set A (4 short chat prompts) and set V (14 held-out tasks), greedy, thinking
off, 2 passes in reversed order. Decode speed is the geometric mean over each set's prompts. It
also checks that the output is reproducible across the passes, and with --plain-ref how many
outputs match the engine's own exact (plain) run — fast modes are reported as such. Suite v1
(--suite v1) is the original synthetic short / long / concurrent test.

Writes arena_result.json, which you upload on the InferArena page. See README.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

try:
    import httpx
except ImportError:                                   # pragma: no cover
    sys.exit("arena_bench needs httpx:  pip install httpx")

HARNESS_VERSION = "2.0.0"
SCHEMA_VERSION = 1        # suite v1 result files keep schema 1
SUITE = "v1"
SCHEMA_VERSION_V2 = 2
SUITE_V2 = "v2"
PASSES_V2 = 2             # pass 1 in order, pass 2 reversed; the median per prompt

# Suite v1. Changing any of these is a new suite version, never an edit to v1.
WORKLOADS = {
    "short": {"prompt_tokens": 512, "max_tokens": 256, "concurrency": 1},
    "long": {"prompt_tokens": 4096, "max_tokens": 256, "concurrency": 1},
    "concurrent": {"prompt_tokens": 512, "max_tokens": 256, "concurrency": 8},
}
SEED = 42
HASH_CHARS = 256          # output fingerprint = sha256 of the first 256 characters of greedy output

_FILLER = ("The history of computing is a history of trade-offs between speed, memory and cost. "
           "Engineers repeatedly discovered that moving data costs more than computing on it. ")
_TASK = ("\n\nUsing the background above, write a long, detailed explanation of how a modern CPU "
         "fetches, decodes and executes instructions, with examples. Write at least 600 words.")


def build_prompt(target_tokens: int, tokens_per_char: float, salt: str = "") -> str:
    """A prompt of roughly target_tokens. The salt goes FIRST so that no two timed requests share
    a prefix — otherwise engines with prefix caching would skip prefill on repeats."""
    head = f"[run {salt}]\n" if salt else ""
    budget_chars = max(0, int(target_tokens / max(tokens_per_char, 1e-6)) - len(head) - len(_TASK))
    body = (_FILLER * (budget_chars // len(_FILLER) + 1))[:budget_chars]
    return head + body + _TASK


class Bench:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 timeout: float = 600.0, client: Optional[httpx.Client] = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.clock = clock
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout, headers=headers)

    # Suite v2 adds greedy + thinking-off hints, sent identically to every engine (engines ignore what they
    # don't know). Set by run_v2(); empty for suite v1, whose requests stay standard OpenAI fields only.
    extra_body: dict = {}
    count_reasoning = False   # v2: reasoning deltas are generated tokens too

    def _body(self, prompt: str, max_tokens: int, stream: bool) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0, "seed": SEED, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        body.update(self.extra_body)
        return body

    def calibrate(self) -> float:
        """Tokens per character for this model's tokenizer, from the engine's own usage count."""
        probe = _FILLER * 20 + _TASK
        r = self.client.post(self.url, json=self._body(probe, 1, stream=False))
        r.raise_for_status()
        pt = (r.json().get("usage") or {}).get("prompt_tokens")
        if not pt:
            return 0.25           # ~4 characters per token; prompt_tokens is still measured per run
        return pt / len(probe)

    def stream_once(self, prompt: str, max_tokens: int) -> dict:
        """One streamed request. Returns timings, token counts and the generated text."""
        t0 = self.clock()
        t_first = t_last = None
        text_parts: list[str] = []
        chunks = 0
        usage: dict = {}
        with self.client.stream("POST", self.url, json=self._body(prompt, max_tokens, stream=True)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    if self.count_reasoning and not piece:
                        piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece:
                        now = self.clock()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        chunks += 1
                        text_parts.append(piece)
        if t_first is None:
            raise RuntimeError("the engine streamed no content")
        completion = usage.get("completion_tokens") or chunks
        return {
            "ttft_s": t_first - t0,
            "decode_s": max(t_last - t_first, 1e-9),
            "end_s": t_last - t0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion,
            "usage_reported": bool(usage),
            "text": "".join(text_parts),
        }


def _metrics(run: dict, fallback_prompt_tokens: int) -> dict:
    pt = run["prompt_tokens"] or fallback_prompt_tokens
    ct = run["completion_tokens"]
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_ms": run["ttft_s"] * 1000,
        "prefill_tps": pt / run["ttft_s"],
        # The first token arrives at TTFT, so decode rate is over the remaining ct-1 tokens.
        "decode_tps": max(ct - 1, 1) / run["decode_s"],
    }


def _median_of(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if not vals:
            continue
        m = statistics.median(vals)
        out[k] = int(round(m)) if k.endswith("tokens") or k in ("concurrency", "repeats") else round(m, 2)
    return out


def run_workload(bench: Bench, name: str, tpc: float, repeats: int, concurrency: Optional[int] = None) -> dict:
    spec = WORKLOADS[name]
    conc = concurrency or spec["concurrency"]
    per_repeat = []
    for _ in range(repeats):
        prompts = [build_prompt(spec["prompt_tokens"], tpc, uuid.uuid4().hex[:12]) for _ in range(conc)]
        if conc == 1:
            r = bench.stream_once(prompts[0], spec["max_tokens"])
            m = _metrics(r, spec["prompt_tokens"])
        else:
            t0 = bench.clock()
            with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                runs = list(ex.map(lambda p: bench.stream_once(p, spec["max_tokens"]), prompts))
            wall = max(bench.clock() - t0, 1e-9)
            each = [_metrics(r, spec["prompt_tokens"]) for r in runs]
            m = _median_of(each)
            m["aggregate_tps"] = round(sum(r["completion_tokens"] for r in runs) / wall, 2)
        m["concurrency"] = conc
        per_repeat.append(m)
    out = _median_of(per_repeat)
    out["repeats"] = repeats
    out["concurrency"] = conc
    return out


def output_fingerprint(bench: Bench, tpc: float) -> str:
    """Greedy output of a FIXED, unsalted prompt. The same engine version, model and quant should
    reproduce it exactly; a mismatch is flagged for review (it can mean work is being skipped)."""
    text = bench.stream_once(build_prompt(WORKLOADS["short"]["prompt_tokens"], tpc), 128)["text"]
    return hashlib.sha256(text[:HASH_CHARS].encode("utf-8")).hexdigest()


# ── Suite v2 (InferArena rules v2) ─────────────────────────────────────────────────────────────────

V2_EXTRA_BODY = {"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}}


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_set_v2(bench: "Bench", name: str, spec: dict, passes: int = PASSES_V2) -> tuple[dict, list[str], bool]:
    """Runs one prompt set. Returns (metrics, per-prompt output hashes of pass 1, reproducible across passes).
    Pass order alternates (forward, reversed, …) so a warming engine can't favour one end of the list. TTFT and
    prefill come from pass 1 only (later passes may hit a prefix cache); decode is the median over passes."""
    prompts = spec["prompts"]
    per: dict[str, list[dict]] = {pid: [] for pid, _ in prompts}
    hashes: dict[str, list[str]] = {pid: [] for pid, _ in prompts}
    for k in range(passes):
        order = prompts if k % 2 == 0 else list(reversed(prompts))
        for pid, text in order:
            r = bench.stream_once(text, spec["max_tokens"])
            m = _metrics(r, len(text) // 4)
            m["pass"] = k
            per[pid].append(m)
            hashes[pid].append(_sha(r["text"]))
    first = [next(m for m in per[pid] if m["pass"] == 0) for pid, _ in prompts]
    decode = [statistics.median(m["decode_tps"] for m in per[pid]) for pid, _ in prompts]
    out = {
        "decode_tps": round(_geomean(decode), 2),
        "ttft_ms": round(statistics.median(m["ttft_ms"] for m in first), 2),
        "prefill_tps": round(_geomean([m["prefill_tps"] for m in first]), 2),
        "prompt_tokens": int(statistics.median(m["prompt_tokens"] for m in first)),
        "completion_tokens": int(statistics.median(m["completion_tokens"] for m in first)),
        "concurrency": 1, "repeats": passes, "prompts": len(prompts),
        "per_prompt_decode_tps": {pid: round(d, 2) for (pid, _), d in zip(prompts, decode)},
    }
    reproducible = all(len(set(h)) == 1 for h in hashes.values())
    return out, [hashes[pid][0] for pid, _ in prompts], reproducible


def identity_vs(plain_ref: Optional[str], hashes_a: list[str]) -> Optional[dict]:
    """How many set-A outputs match the engine's own exact (plain) run, from that run's result file."""
    if not plain_ref:
        return None
    with open(plain_ref, encoding="utf-8") as f:
        ref = json.load(f)
    ref_hashes = (ref.get("output_hashes") or {}).get("A") or []
    if len(ref_hashes) != len(hashes_a):
        sys.exit(f"--plain-ref {plain_ref} has no matching set-A output hashes (is it a suite v2 exact run?)")
    return {"identical": sum(a == b for a, b in zip(hashes_a, ref_hashes)), "of": len(hashes_a)}


def load_prompts_v2() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    try:
        import prompts_v2  # noqa: WPS433 — shipped next to this file
    except ImportError:
        sys.exit("suite v2 needs prompts_v2.py next to arena_bench.py (download it from the InferArena Submit tab)")
    return prompts_v2.SETS


# ── Hardware fingerprint ─────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def gpu_info() -> dict:
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
        rows = [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]
        if rows:
            return {"gpu_name": rows[0][0], "gpu_vram_gb": round(float(rows[0][1]) / 1024, 2),
                    "gpu_count": len(rows), "driver": rows[0][2] if len(rows[0]) > 2 else None}
    return {"gpu_name": "unknown", "gpu_vram_gb": 0.0, "gpu_count": 0, "driver": None}


def cpu_name() -> str:
    if sys.platform.startswith("linux"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if out:
            return out.splitlines()[0].strip()
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or platform.machine()


def ram_gb() -> Optional[float]:
    try:
        import psutil                                 # optional
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal"):
                return round(int(re.findall(r"\d+", line)[0]) / 1024 ** 2, 1)
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / 1024 ** 3, 1)
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out.isdigit():
            return round(int(out) / 1024 ** 3, 1)
    return None


class VramSampler:
    """Peak GPU memory in use during the run (all processes, summed over GPUs) via nvidia-smi."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            try:
                self.peak_mib = max(self.peak_mib, sum(float(x) for x in out.split()))
            except ValueError:
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)

    @property
    def peak_gb(self) -> Optional[float]:
        return round(self.peak_mib / 1024, 2) if self.peak_mib else None


# ── Main ─────────────────────────────────────────────────────────────────────────────────────────

def build_result(args, workloads: dict, output_hash: str, hw: dict, peak_vram_gb: Optional[float]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "suite": SUITE,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "model": args.model,
        "quant": args.quant,
        "mode": args.mode,
        "hardware": {"gpu_name": hw["gpu_name"], "gpu_vram_gb": hw["gpu_vram_gb"],
                     "gpu_count": hw["gpu_count"], "driver": hw.get("driver"),
                     "cpu": cpu_name(), "ram_gb": ram_gb(),
                     "os": f"{platform.system()} {platform.release()}"},
        "workloads": workloads,
        "output_hash": output_hash,
        "peak_vram_gb": peak_vram_gb,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": args.notes or "",
    }


def build_result_v2(args, sets: dict, hashes: dict, reproducible: bool, identity: Optional[dict], hw: dict,
                    peak_vram_gb: Optional[float]) -> dict:
    out = build_result(args, sets, _sha("".join(hashes["A"])), hw, peak_vram_gb)
    out.update({"schema_version": SCHEMA_VERSION_V2, "suite": SUITE_V2, "decode_class": args.decode_class,
                "reproducible": reproducible, "identity": identity, "output_hashes": hashes})
    return out


def failure_record(args, hw: dict) -> dict:
    """A run that could not happen still counts for coverage ("runs on k/N"); it is never dropped."""
    return {"schema_version": SCHEMA_VERSION_V2, "harness_version": HARNESS_VERSION, "suite": SUITE_V2,
            "engine": args.engine, "engine_version": args.engine_version, "model": args.model, "quant": args.quant,
            "mode": args.mode, "status": args.record_status, "reason": args.reason or "",
            "hardware": {"gpu_name": hw["gpu_name"], "gpu_vram_gb": hw["gpu_vram_gb"], "gpu_count": hw["gpu_count"]},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="InferArena benchmark harness (suite v2 by default; v1 available)")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True, help="model id as the engine names it")
    p.add_argument("--engine", required=True, help="engine slug: alphallama, vllm, sglang, ollama, llamacpp, ...")
    p.add_argument("--engine-version", required=True)
    p.add_argument("--quant", required=True, help="e.g. Q4_K_M, AWQ, fp16")
    p.add_argument("--mode", required=True, choices=["full_vram", "offload"],
                   help="full_vram: the whole model is in GPU memory; offload: it is not")
    p.add_argument("--suite", choices=["v2", "v1"], default="v2")
    p.add_argument("--decode-class", choices=["exact", "fast"], default="exact",
                   help="v2: exact = plain greedy decoding; fast = a speculative / multi-token mode")
    p.add_argument("--plain-ref", help="v2, fast runs: the engine's own exact v2 result file, to count identical outputs")
    p.add_argument("--record-status", choices=["failed_start", "not_supported"],
                   help="v2: write a coverage record instead of benchmarking (the engine could not run)")
    p.add_argument("--reason", default="", help="with --record-status: what happened (max 300 chars)")
    p.add_argument("--workloads", default="short,long,concurrent", help="v1 only")
    p.add_argument("--repeats", type=int, default=3, help="v1 only")
    p.add_argument("--api-key", default=os.environ.get("ARENA_API_KEY"))
    p.add_argument("--notes", default="", help="launch flags / anything a reviewer should know (max 500 chars)")
    p.add_argument("--out", default="arena_result.json")
    p.add_argument("--timeout", type=float, default=600.0)
    return p.parse_args(argv)


def main_v2(args) -> int:
    hw = gpu_info()
    if args.record_status:
        rec = failure_record(args, hw)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"Wrote coverage record {args.out} ({args.record_status}).")
        return 0
    sets_spec = load_prompts_v2()
    bench = Bench(args.base_url, args.model, args.api_key, args.timeout)
    bench.extra_body = dict(V2_EXTRA_BODY)
    bench.count_reasoning = True
    print(f"InferArena harness {HARNESS_VERSION}  suite {SUITE_V2}  class {args.decode_class}")
    print(f"  engine {args.engine} {args.engine_version}  model {args.model} ({args.quant}, {args.mode})")
    print(f"  GPU {hw['gpu_name']} x{hw['gpu_count']}  {hw['gpu_vram_gb']} GB")
    print("  warm-up ...")
    bench.stream_once("Say hello in five words.", 16)
    sets, hashes, repro = {}, {}, True
    with VramSampler() as vram:
        for name, spec in sets_spec.items():
            print(f"  set {name}: {len(spec['prompts'])} prompts x{PASSES_V2} passes ...", flush=True)
            sets[name], hashes[name], ok = run_set_v2(bench, name, spec)
            repro = repro and ok
            print(f"    decode {sets[name]['decode_tps']:.1f} tok/s (geomean)  TTFT {sets[name]['ttft_ms']:.0f} ms")
    identity = identity_vs(args.plain_ref, hashes["A"])
    if not repro:
        print("  note: outputs differed between passes, so this run is not exact class (it will be flagged)")
    if identity:
        print(f"  identical to the plain run: {identity['identical']}/{identity['of']}")
    out = build_result_v2(args, sets, hashes, repro, identity, hw, vram.peak_gb)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}. Upload it on the InferArena page (Submit tab).")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.suite == "v2":
        return main_v2(args)
    names = [w.strip() for w in args.workloads.split(",") if w.strip()]
    bad = [w for w in names if w not in WORKLOADS]
    if bad or "short" not in names:
        sys.exit(f"workloads must include 'short' and be from {list(WORKLOADS)}; got {names}")

    bench = Bench(args.base_url, args.model, args.api_key, args.timeout)
    hw = gpu_info()
    print(f"InferArena harness {HARNESS_VERSION}  suite {SUITE}")
    print(f"  engine {args.engine} {args.engine_version}  model {args.model} ({args.quant}, {args.mode})")
    print(f"  GPU {hw['gpu_name']} x{hw['gpu_count']}  {hw['gpu_vram_gb']} GB")

    tpc = bench.calibrate()
    print("  warm-up ...")
    bench.stream_once(build_prompt(64, tpc, "warmup"), 16)

    results = {}
    with VramSampler() as vram:
        for name in names:
            print(f"  {name} x{args.repeats} ...", flush=True)
            results[name] = run_workload(bench, name, tpc, args.repeats)
            m = results[name]
            agg = f"  aggregate {m['aggregate_tps']:.1f} tok/s" if "aggregate_tps" in m else ""
            print(f"    prefill {m['prefill_tps']:.0f} tok/s  decode {m['decode_tps']:.1f} tok/s  "
                  f"TTFT {m['ttft_ms']:.0f} ms{agg}")
            if m["completion_tokens"] < 64:
                print("    warning: the model stopped early (<64 tokens); decode speed is less reliable")
        fp = output_fingerprint(bench, tpc)

    out = build_result(args, results, fp, hw, vram.peak_gb)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}. Upload it on the InferArena page (Submit tab).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
