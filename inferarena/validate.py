"""
InferArena / TrainArena — validation, classification and confidence rules for benchmark submissions.

Everything here is generic: no engine, model or GPU is special-cased. Engines are rows in
arena_engines, GPUs are rows in static/arena_gpus.json, and model size is read from the model id.
The two arenas differ only by their entry in KINDS (workloads, metrics, headline); every function
below reads the spec from there instead of knowing which arena it is serving.

Plausibility checks only ever add FLAGS for a human reviewer. A result that looks too fast may be
legitimate (speculative decoding emits several tokens per weight read), so nothing is rejected on
speed alone — the reviewer decides, and the flag is shown to them.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
SUPPORTED_HARNESS_MAJORS = {"1", "2"}
MODES = ("full_vram", "offload")

# Decode can legitimately exceed the one-token-per-weight-read bound with speculative decoding /
# multi-token prediction, so the flag threshold is deliberately generous.
ROOFLINE_MARGIN = 4.0
# Training at a real-world MFU above ~60% is rare; x2 over the spec-sheet peak leaves room for
# FLOP-count approximations without ever flagging an honest run.
COMPUTE_MARGIN = 2.0
# Training FLOPs per token for adapter fine-tuning: forward (2N) + input-gradient backward (2N);
# frozen base weights need no weight gradients.
ADAPTER_FLOPS_PER_PARAM = 4.0
EVAL_LOSS_PEER_TOLERANCE = 0.05   # same model/workload/quant class: eval loss should land within 5%
AGREE_WITHIN = 0.10      # two runs within ±10% of each other agree
DISPUTE_BEYOND = 0.25    # a run more than 25% away from every other run is disputed

_GPU_FILE = Path(__file__).resolve().parents[1] / "data" / "gpus.json"


# ── Schema ───────────────────────────────────────────────────────────────────────────────────────

class WorkloadMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(ge=1, le=1_000_000)
    completion_tokens: int = Field(ge=1, le=1_000_000)
    ttft_ms: float = Field(gt=0, lt=3_600_000)
    prefill_tps: float = Field(gt=0, lt=10_000_000)
    decode_tps: float = Field(gt=0, lt=100_000)
    aggregate_tps: Optional[float] = Field(default=None, gt=0, lt=10_000_000)
    concurrency: int = Field(default=1, ge=1, le=1024)
    repeats: int = Field(default=1, ge=1, le=100)


class Hardware(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu_name: str = Field(min_length=1, max_length=200)
    gpu_vram_gb: float = Field(ge=0, le=4096)
    gpu_count: int = Field(ge=0, le=64)
    cpu: Optional[str] = Field(default=None, max_length=200)
    ram_gb: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    os: Optional[str] = Field(default=None, max_length=200)
    driver: Optional[str] = Field(default=None, max_length=100)


class ArenaResult(BaseModel):
    """The JSON the inference harness writes (scripts/arena_bench/arena_bench.py)."""
    model_config = ConfigDict(extra="ignore")

    kind: Literal["infer"] = "infer"
    schema_version: Literal[1, 2]
    harness_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    suite: Literal["v1", "v2"]
    engine: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,40}$")
    engine_version: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=200)
    quant: str = Field(min_length=1, max_length=60)
    mode: Literal["full_vram", "offload"]
    hardware: Hardware
    workloads: dict[str, WorkloadMetrics]
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    peak_vram_gb: Optional[float] = Field(default=None, ge=0, le=4096)
    created_at: Optional[str] = Field(default=None, max_length=40)
    notes: str = Field(default="", max_length=500)
    # Rules v2 (suite v2) only.
    decode_class: Literal["exact", "fast"] = "exact"
    reproducible: Optional[bool] = None
    identity: Optional[dict[str, int]] = None

    @field_validator("harness_version")
    @classmethod
    def _supported_harness(cls, v: str) -> str:
        if v.split(".")[0] not in SUPPORTED_HARNESS_MAJORS:
            raise ValueError(f"unsupported harness version {v}; please update the harness")
        return v

    @field_validator("workloads")
    @classmethod
    def _known_workloads(cls, v: dict) -> dict:
        _check_workloads("infer", v)
        conc = v.get("concurrent")
        if conc is not None and conc.aggregate_tps is None:
            raise ValueError("the 'concurrent' workload needs aggregate_tps")
        return v

    @model_validator(mode="after")
    def _suite_workloads(self):
        need = SUITE_REQUIRED[self.suite]
        missing = [w for w in need if w not in self.workloads]
        if missing:
            raise ValueError(f"suite {self.suite} needs the workloads {', '.join(need)}")
        if self.suite == "v1" and set(self.workloads) - set(SUITE_WORKLOADS["v1"]):
            raise ValueError("suite v1 results hold only short / long / concurrent")
        if self.suite == "v2" and set(self.workloads) - set(SUITE_WORKLOADS["v2"]):
            raise ValueError("suite v2 results hold only the prompt sets A, V and R")
        return self


class TrainWorkloadMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    train_tps: float = Field(gt=0, lt=100_000_000)          # steady-state dataset tokens / s
    time_to_train_s: float = Field(gt=0, lt=30 * 86_400)    # end-to-end, incl. load + warm-up
    tokens: int = Field(ge=1, le=10_000_000_000)
    samples: int = Field(ge=1, le=100_000_000)
    peak_vram_gb: Optional[float] = Field(default=None, ge=0, le=4096)
    eval_loss: Optional[float] = Field(default=None, gt=0, lt=100)       # None = eval skipped
    base_eval_loss: Optional[float] = Field(default=None, gt=0, lt=100)


class TrainResult(BaseModel):
    """The JSON the training harness writes (scripts/arena_bench/train_bench.py)."""
    model_config = ConfigDict(extra="ignore")

    kind: Literal["train"]
    schema_version: Literal[1]
    harness_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    suite: Literal["train-v1"]
    engine: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,40}$")
    engine_version: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=200)
    quant: str = Field(min_length=1, max_length=60)          # precision of the frozen base
    mode: Literal["full_vram", "offload"]
    hardware: Hardware
    workloads: dict[str, TrainWorkloadMetrics]
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    peak_vram_gb: Optional[float] = Field(default=None, ge=0, le=4096)
    created_at: Optional[str] = Field(default=None, max_length=40)
    notes: str = Field(default="", max_length=500)

    @field_validator("harness_version")
    @classmethod
    def _supported_harness(cls, v: str) -> str:
        if v.split(".")[0] not in SUPPORTED_HARNESS_MAJORS:
            raise ValueError(f"unsupported harness version {v}; please update the harness")
        return v

    @field_validator("workloads")
    @classmethod
    def _known_workloads(cls, v: dict) -> dict:
        _check_workloads("train", v)
        return v


AnyResult = Annotated[Union[ArenaResult, TrainResult], Field(discriminator="kind")]


class ArenaSubmitRequest(BaseModel):
    result: AnyResult
    captcha_token: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _default_kind(cls, data):
        # Result files written before TrainArena existed carry no `kind`; they are inference runs.
        if isinstance(data, dict) and isinstance(data.get("result"), dict) and "kind" not in data["result"]:
            data = {**data, "result": {**data["result"], "kind": "infer"}}
        return data


# ── Arena kinds ──────────────────────────────────────────────────────────────────────────────────
# The only place the two arenas differ. `primary` is the ranking metric per workload, `headline`
# the (workload, metric) that confidence compares (workload None = the file's only workload),
# `row_metrics` what a leaderboard row reports, `default` the workload a board opens on.
# A training file holds exactly ONE workload because each workload trains on a different base
# precision (bf16 for LoRA, 4-bit for QLoRA) and `quant` records that precision — and so that a
# model which only fits as QLoRA can still enter.

KINDS: dict[str, dict] = {
    "infer": {
        "suite": "v2",
        # v1: synthetic short / long / concurrent; v2 (rules v2): prompt set A (short chat) and V (held-out tasks),
        # decode = geometric mean over the set's prompts. Which ones a file must hold depends on its suite.
        "workloads": ("V", "A", "R", "short", "long", "concurrent"),
        "required": None,
        "default": "V",
        "primary": {"V": "decode_tps", "A": "decode_tps", "R": "decode_tps", "short": "decode_tps", "long": "decode_tps",
                    "concurrent": "aggregate_tps"},
        "headline": ("V", "decode_tps"),
        "headline_fallback": ("short", "decode_tps"),
        "row_metrics": ("decode_tps", "prefill_tps", "ttft_ms", "aggregate_tps"),
    },
    "train": {
        "suite": "train-v1",
        "workloads": ("lora", "qlora"),
        "required": None,
        "exactly_one": True,
        "default": "lora",
        "primary": {"lora": "train_tps", "qlora": "train_tps"},
        "headline": (None, "train_tps"),
        "row_metrics": ("train_tps", "time_to_train_s", "peak_vram_gb", "eval_loss", "base_eval_loss"),
    },
}
KIND_NAMES = tuple(KINDS)
SUITE_WORKLOADS = {"v1": ("short", "long", "concurrent"), "v2": ("A", "V", "R")}
SUITE_REQUIRED = {"v1": ("short",), "v2": ("A", "V", "R")}
WORKLOADS = KINDS["infer"]["workloads"]          # kept for existing importers
# A run carrying any of these flags cannot become `verified`, however well it agrees.
# not_reproducible is shown, never blocking (plain decoding counts even if not bit-identical across
# restarts); exact_output_mismatch means a run claims plain decoding but isn't, so it can't be verified.
BLOCKS_VERIFIED = {"no_quality_check", "exact_output_mismatch"}


def _check_workloads(kind: str, v: dict) -> None:
    spec = KINDS[kind]
    unknown = set(v) - set(spec["workloads"])
    if unknown:
        raise ValueError(f"unknown workloads: {sorted(unknown)}")
    if spec["required"] and spec["required"] not in v:
        raise ValueError(f"the '{spec['required']}' workload is required")
    if spec.get("exactly_one") and len(v) != 1:
        raise ValueError(f"a result file holds exactly one workload (one of: {', '.join(spec['workloads'])})")


def workload_order(kind: str, w: str) -> int:
    wl = KINDS[kind]["workloads"]
    return wl.index(w) if w in wl else 99


# ── Classification ───────────────────────────────────────────────────────────────────────────────

def canonical_sha256(raw: dict) -> str:
    """Hash of the submission with keys sorted, so re-uploading the same file is detected."""
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_QUANT_BITS = {"16-bit": 16.0, "8-bit": 8.0, "6-bit": 6.6, "5-bit": 5.5, "4-bit": 4.5, "3-bit": 3.5, "2-bit": 2.6}


def quant_class(quant: str) -> str:
    """Group engine-specific quant names (Q4_K_M, AWQ, GPTQ-int4, fp16...) by bit width.

    Formats are not identical across engines, so this is a comparison CLASS, not an equivalence;
    the exact quant is always shown next to the number.
    """
    q = quant.lower().replace(" ", "")
    if re.search(r"(^|[^a-z0-9])(b?f16|fp16|bf16|half)([^a-z0-9]|$)", q) or q in ("f16", "fp16", "bf16"):
        return "16-bit"
    m = re.search(r"(?:^|[^a-z0-9])(?:q|iq|int|w|fp|nf)(\d)", q) or re.search(r"(\d)\s*-?bit", q)
    if m:
        bits = int(m.group(1))
        if bits in (2, 3, 4, 5, 6, 8):
            return f"{bits}-bit"
    if "awq" in q or "gptq" in q or "nf4" in q:
        return "4-bit"
    return "other"


def quant_bits(quant: str) -> Optional[float]:
    return _QUANT_BITS.get(quant_class(quant))


def vram_tier(gpu_vram_gb: float) -> str:
    """Per-GPU VRAM bucket. nvidia-smi reports e.g. 23.99 GB for a 24 GB card, so round first."""
    gb = round(gpu_vram_gb)
    for t in (8, 12, 16, 24, 32):
        if gb <= t:
            return str(t)
    return "48+"


def model_params_b(model_id: str) -> tuple[Optional[float], Optional[float]]:
    """(total, active) parameters in billions, read from the model id ("qwen3-30b-a3b" -> 30, 3).

    Returns (None, None) when the id carries no size; callers then skip the size-based checks.
    """
    s = model_id.lower()
    totals = re.findall(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])", s)
    if not totals:
        return None, None
    total = float(totals[0])
    a = re.search(r"(?<![a-z0-9])a(\d+(?:\.\d+)?)b(?![a-z0-9])", s)
    active = float(a.group(1)) if a else total
    return total, active


@lru_cache(maxsize=1)
def _gpu_table() -> list[tuple[str, dict]]:
    data = json.loads(_GPU_FILE.read_text(encoding="utf-8"))
    rows = [(g["match"].lower(), g) for g in data.get("gpus", [])]
    return sorted(rows, key=lambda r: len(r[0]), reverse=True)     # longest pattern first


def _gpu_spec(gpu_name: str, field: str) -> Optional[float]:
    name = gpu_name.lower()
    for pattern, row in _gpu_table():
        if pattern in name:
            v = row.get(field)
            return float(v) if v is not None else None
    return None


def gpu_bandwidth_gbps(gpu_name: str) -> Optional[float]:
    return _gpu_spec(gpu_name, "bandwidth_gbps")


def gpu_bf16_tflops(gpu_name: str) -> Optional[float]:
    return _gpu_spec(gpu_name, "bf16_tflops")


def plausibility_flags(r) -> list[str]:
    """Reasons a reviewer should look twice. Never a reason to reject on its own."""
    flags: list[str] = []
    hw = r.hardware
    total_b, active_b = model_params_b(r.model)
    bits = quant_bits(r.quant)
    # Inference decode is bounded by memory bandwidth, training by compute.
    bw = gpu_bandwidth_gbps(hw.gpu_name) if r.kind == "infer" else None
    tflops = gpu_bf16_tflops(hw.gpu_name) if r.kind == "train" else None

    if bw is None and tflops is None:
        flags.append("unknown_hardware")
    if total_b is None:
        flags.append("unknown_model_size")
    if bits is None:
        flags.append("unknown_quant")

    if r.kind == "train":
        if tflops is not None and active_b:
            bound = (tflops * 1e12 * max(1, hw.gpu_count)
                     / (ADAPTER_FLOPS_PER_PARAM * active_b * 1e9) * COMPUTE_MARGIN)
            for name, w in r.workloads.items():
                if w.train_tps > bound:
                    flags.append(f"exceeds_compute_bound:{name}")
        for name, w in r.workloads.items():
            if w.eval_loss is None:
                if "no_quality_check" not in flags:
                    flags.append("no_quality_check")
            elif w.base_eval_loss is not None and w.eval_loss >= w.base_eval_loss:
                flags.append(f"no_learning:{name}")

    # Rules v2: an exact-class run must reproduce its own output across passes and, when it says how many outputs
    # match a plain run, match all of them. Fast-class runs are expected to exceed the bandwidth bound.
    decode_class = getattr(r, "decode_class", "exact")
    if r.kind == "infer" and getattr(r, "suite", "v1") == "v2" and decode_class == "exact":
        if r.reproducible is False:
            flags.append("not_reproducible")
        ident = r.identity or {}
        if ident and ident.get("identical", 0) < ident.get("of", 0):
            flags.append("exact_output_mismatch")

    if bw is not None and active_b and bits and decode_class == "exact":
        active_gb = active_b * bits / 8
        bound = bw * max(1, hw.gpu_count) / active_gb * ROOFLINE_MARGIN
        for name in ("short", "long", "A", "V", "R"):
            w = r.workloads.get(name)
            if w is not None and w.decode_tps > bound:
                flags.append(f"exceeds_bandwidth_bound:{name}")

    if total_b and bits and hw.gpu_vram_gb > 0:
        weights_gb = total_b * bits / 8
        vram_total = hw.gpu_vram_gb * max(1, hw.gpu_count)
        if r.mode == "full_vram" and weights_gb > vram_total * 1.05:
            flags.append("mode_mismatch:weights_exceed_vram")
        if r.mode == "offload" and weights_gb < vram_total * 0.5:
            flags.append("mode_mismatch:fits_in_vram")
    return flags


# ── Confidence ───────────────────────────────────────────────────────────────────────────────────

def engine_major(version: str) -> str:
    m = re.match(r"\D*(\d+)", version)
    return m.group(1) if m else version


def group_key(sub) -> tuple:
    """Runs that measure the same thing and so should agree with each other."""
    return (sub.kind, sub.engine_slug, engine_major(sub.engine_version), sub.mode, sub.model_id,
            sub.quant, sub.gpu_name, sub.gpu_count, sub.suite, decode_class_of(sub))


def decode_class_of(sub) -> str:
    """exact | fast (rules v2); v1 runs and training runs are 'exact'."""
    return ((getattr(sub, "raw", None) or {}).get("decode_class")) or "exact"


def headline(sub) -> Optional[float]:
    """The one number confidence compares: decode tok/s for inference, training tok/s for training."""
    spec = KINDS[sub.kind or "infer"]
    workload, metric = spec["headline"]
    metrics = sub.metrics or {}
    if workload is None:
        workload = next((w for w in spec["workloads"] if w in metrics), None)
    elif workload not in metrics and spec.get("headline_fallback"):
        workload, metric = spec["headline_fallback"]
    w = metrics.get(workload) or {}
    return w.get(metric)


def _rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(a, b)


def assign_confidence(group: list) -> None:
    """Set .confidence on every APPROVED run of one group, in place.

    verified       another approved run, by a different user, agrees within ±10% — and the two are
                   not both lab runs (our own runs agreeing is not independent confirmation), and
                   the run carries no BLOCKS_VERIFIED flag (a training run whose quality was never
                   checked)
    disputed       other users' runs exist and every one is more than 25% away
    lab            measured by us and not yet independently confirmed
    single_source  everything else
    """
    for r in group:
        v = headline(r)
        base = "lab" if r.is_lab else "single_source"
        others = [o for o in group if o is not r and o.user_id != r.user_id and headline(o)]
        if not v or not others:
            r.confidence = base
            continue
        agree = [o for o in others if _rel_diff(v, headline(o)) <= AGREE_WITHIN
                 and not (r.is_lab and o.is_lab)]
        if agree:
            r.confidence = base if BLOCKS_VERIFIED.intersection(r.flags or []) else "verified"
        elif all(_rel_diff(v, headline(o)) > DISPUTE_BEYOND for o in others):
            r.confidence = "disputed"
        else:
            r.confidence = base


CONFIDENCE_RANK = {"disputed": 0, "single_source": 1, "lab": 2, "verified": 3}


def summarise_confidence(values: list[str]) -> str:
    """One badge for a leaderboard row. Any dispute is surfaced rather than hidden by a better run."""
    if "disputed" in values:
        return "disputed"
    return max(values, key=lambda c: CONFIDENCE_RANK.get(c, 0)) if values else "single_source"


def med(values: list[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(median(vals), 2) if vals else None
