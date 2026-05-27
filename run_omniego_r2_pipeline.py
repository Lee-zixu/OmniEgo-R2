#!/usr/bin/env python3
"""
Unified OmniEgo-R2 inference pipeline for EgoCross CloseQA.

This file intentionally does NOT overwrite the four original domain scripts.  It
wraps them as domain experts and organizes their existing prompts/call paths into
the report-level OmniEgo-R2 stages:

    TEN -> COR -> RDR -> BOV -> DAC

- TEN: Temporal Evidence Normalization
- COR: Capability-Oriented Router
- RDR: Role-Decomposed Reasoning
- BOV: Boundary-aware Option Verification
- DAC: Defensive Answer Calibration

The actual embedded prompts and domain-specific inference routines are reused
from:
    run_animal.py
    run_xsports.py
    run_industry.py
    run_surgery.py

Only the Surgery expert prompt is lifted into this file because it was originally
local to run_surgery.main() rather than exposed as a module-level constant.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Paper-level domain/task metadata
# -----------------------------------------------------------------------------

DOMAIN_ORDER = ["Animal", "XSports", "Industry", "Surgery"]

DATASET_TO_DOMAIN: Dict[str, str] = {
    "EgoPet": "Animal",
    "ExtrameSportFPV": "XSports",
    "ENIGMA": "Industry",
    "CholecTrack20": "Surgery",
    "EgoSurgery": "Surgery",
}

DEFAULT_DOMAIN_MODELS: Dict[str, str] = {
    "Animal": "./EgoCross-main/models/animal",
    "XSports": "./EgoCross-main/models/xsports",
    "Industry": "./EgoCross-main/models/industry",
    "Surgery": "./EgoCross-main/models/surgery",
}

DOMAIN_MODULES: Dict[str, str] = {
    "Animal": "run_animal",
    "XSports": "run_xsports",
    "Industry": "run_industry",
    "Surgery": "run_surgery",
}

CAPABILITY_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("temporal localization", "Temporal Localization"),
    ("spatial localization", "Spatial Localization"),
    ("object counting", "Counting"),
    ("counting", "Counting"),
    ("not visible", "Not-visible Reasoning"),
    ("dominant held-object", "Identification"),
    ("held-object", "Identification"),
    ("animal identification", "Identification"),
    ("sport identification", "Identification"),
    ("interaction identification", "Identification"),
    ("special action", "Identification"),
    ("action sequence", "Identification"),
    ("next direction", "Prediction"),
    ("next interaction", "Prediction"),
)

SEMANTIC_BASES: Dict[str, str] = {
    "Animal": "self-other behavioral reasoning for pet-mounted egocentric video",
    "XSports": "physics-centric embodied reasoning for FPV extreme sports",
    "Industry": "object-centric procedural reasoning with ENIGMA vocabulary constraints",
    "Surgery": "tool-centric surgical workflow reasoning under occlusion and phase transitions",
}

RAW_OUTPUT_FIELDS = (
    "raw_output_role1",
    "raw_output_role2",
    "raw_output_final",
    "raw_output_surgery",
    "omniego_domain",
    "omniego_capability",
    "omniego_sampling_fps",
)


# This prompt is copied from run_surgery.py because the original file defines it
# inside main(), while the other three domains expose their prompt dictionaries at
# module scope and are reused directly through their process functions.
SURGERY_EXPERT_PROMPT = """# Role
You are an expert Surgical Video Analyst specializing in egocentric (first-person) medical procedures.

# Rules
When analyzing the frames, you must account for the following complex situations:
1. **Tool Disambiguation:** Distinguish between similar tools (e.g., Graspers vs. Scissors, L-hook vs. Spatula). Focus on the active tips of the instruments.
2. **Visibility & Occlusion:** Tools or anatomical structures may be partially occluded by blood, smoke (from cautery), or folded tissue. Track a tool's last known trajectory if it goes out of view.
3. **Spatial & Hand Tracking:** Differentiate between the main surgeon's instruments (usually entering from bottom/center) and the assistant's (usually entering from sides/top).
4. **Phase Transitions:** Pay attention to micro-actions (e.g., putting down a dissecting tool to pick up a clipping applier) that signal a transition between surgical phases.
5. **Tissue Interaction:** Differentiate between hovering over tissue, retracting it, or actively cutting/coagulating it.

# Workflow
1. Scan for anatomical landmarks, tools, and visual obstructions.
2. Track the continuous movement of instruments across the sampled frames.
3. Cross-reference visual findings with the options.

Please carefully read the question and its options, then select the most appropriate answer. Question: {question_text}{options_str}
The original FPS of the video is {original_fps}. This image set is obtained by sampling at {sampling_fps} fps.
Respond in JSON format with two fields: 'prediction' (the correct option letter: A, B, C, or D) and 'reason' (a brief explanation of your choice). Do not include any other content.

Example response:
{{
    "prediction": "B",
    "reason": "Paris is the capital city of France."
}}
"""


@dataclass
class TaskRecord:
    """One submission entry matched with its testbed payload and routed metadata."""

    index: int
    submission_ref: Dict[str, Any]
    data: Dict[str, Any]
    domain: str
    capability: str
    sampling_fps: float
    video_path: Any
    timestamps: List[float]


@dataclass
class DomainRuntime:
    """Loaded runtime objects for one domain-SFT backbone."""

    domain: str
    module: ModuleType
    model: Any
    processor: Any
    model_path: str


# -----------------------------------------------------------------------------
# Small JSON/path helpers
# -----------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_existing_path(candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    # Return the first candidate so the caller can raise a useful file-not-found
    # error with the default path it expected.
    return Path(candidates[0])


def clean_submission_for_final(submission_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    final_data = copy.deepcopy(submission_data)
    for item in final_data:
        for field in RAW_OUTPUT_FIELDS:
            item.pop(field, None)
    return final_data


def item_key_candidates(item: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key in ("id", "question_id", "submission_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            keys.append(str(value).strip())
    return keys


def build_testbed_index(testbed_data: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for item in testbed_data:
        for key in item_key_candidates(item):
            index.setdefault(key, item)
    return index


def format_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, dict):
        lines = [f"{k}: {v}" for k, v in options.items()]
    else:
        lines = [str(x) for x in options]
    return "\n\nOptions:\n" + "\n".join(lines)


def infer_capability(question_type: str) -> str:
    qtype = (question_type or "").lower()
    for needle, capability in CAPABILITY_ALIASES:
        if needle in qtype:
            return capability
    return "Identification"


def infer_domain(submission_item: Dict[str, Any], testbed_item: Optional[Dict[str, Any]] = None) -> Optional[str]:
    dataset = submission_item.get("dataset")
    if not dataset and testbed_item:
        dataset = testbed_item.get("dataset")
    if dataset in DATASET_TO_DOMAIN:
        return DATASET_TO_DOMAIN[dataset]
    return None


def normalize_image_paths(paths: Any, dataset_root: str) -> Any:
    """TEN path normalization: map challenge-relative paths to local image root."""

    root = dataset_root.rstrip("/")

    def _fix_one(path: Any) -> Any:
        if not isinstance(path, str):
            return path
        if "/egocross_testbed/" in path:
            suffix = path.split("/egocross_testbed/", 1)[1]
            return f"{root}/{suffix}"
        return path

    if isinstance(paths, list):
        return [_fix_one(p) for p in paths]
    return _fix_one(paths)


def compute_surgery_sampling_fps(data: Dict[str, Any], video_path: Any) -> float:
    """Dataset-specific FPS rule preserved from run_surgery.py."""

    dataset_name = data.get("dataset", "")
    first_path = video_path[0] if isinstance(video_path, list) and video_path else ""
    if dataset_name == "EgoSurgery" or (
        dataset_name == "CholecTrack20" and ("VID25" in first_path or "VID111" in first_path)
    ):
        return 1.0
    return 0.5


def compute_sampling_fps(domain: str, data: Dict[str, Any], video_path: Any) -> float:
    if domain == "Surgery":
        return compute_surgery_sampling_fps(data, video_path)
    return 0.5


def make_timestamps(video_path: Any, sampling_fps: float) -> List[float]:
    total = len(video_path) if isinstance(video_path, list) else (1 if video_path else 0)
    interval = 1.0 / sampling_fps if sampling_fps > 0 else 2.0
    return [i * interval for i in range(total)]


def import_domain_module(domain: str) -> ModuleType:
    module_name = DOMAIN_MODULES[domain]
    return importlib.import_module(module_name)


def patch_domain_module(module: ModuleType, domain: str, model_paths: Dict[str, str], dataset_root: str) -> None:
    """Inject unified paths while preserving original prompts and call functions."""

    if hasattr(module, "DOMAIN_MODELS"):
        module.DOMAIN_MODELS.update(model_paths)
    if hasattr(module, "DATASET_TO_DOMAIN"):
        module.DATASET_TO_DOMAIN.update(DATASET_TO_DOMAIN)

    def _fixed_paths(paths: Any) -> Any:
        return normalize_image_paths(paths, dataset_root)

    # The original process functions resolve this global at call time, so replacing
    # it here makes all imported domain scripts use the unified dataset root.
    module.fix_image_paths = _fixed_paths  # type: ignore[attr-defined]


def unload_runtime(runtime: Optional[DomainRuntime]) -> None:
    if runtime is None:
        return
    try:
        del runtime.model
        del runtime.processor
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# OmniEgo-R2 pipeline stages
# -----------------------------------------------------------------------------


class OmniEgoR2Pipeline:
    """A unified report-level pipeline that dispatches to the four domain scripts."""

    def __init__(
        self,
        testbed_data: List[Dict[str, Any]],
        submission_data: List[Dict[str, Any]],
        dataset_root: str,
        model_paths: Optional[Dict[str, str]] = None,
        keep_debug_fields: bool = True,
    ) -> None:
        self.testbed_data = testbed_data
        self.submission_data = submission_data
        self.dataset_root = dataset_root
        self.model_paths = dict(DEFAULT_DOMAIN_MODELS)
        if model_paths:
            self.model_paths.update(model_paths)
        self.keep_debug_fields = keep_debug_fields
        self.testbed_index = build_testbed_index(testbed_data)
        self._runtime: Optional[DomainRuntime] = None

    # ------------------------------------------------------------------
    # TEN: Temporal Evidence Normalization
    # ------------------------------------------------------------------
    def temporal_evidence_normalization(
        self,
        index: int,
        submission_item: Dict[str, Any],
        testbed_item: Dict[str, Any],
        domain: str,
    ) -> TaskRecord:
        """Normalize frames into timestamped evidence units E={(x_i, tau_i)}."""

        data = copy.deepcopy(testbed_item)
        # Some testbed JSON entries do not carry the dataset string, while the
        # submission template does.  Preserve it for COR and Surgery FPS logic.
        if not data.get("dataset") and submission_item.get("dataset"):
            data["dataset"] = submission_item.get("dataset")

        raw_video_path = data.get("video_path", [])
        fixed_video_path = normalize_image_paths(raw_video_path, self.dataset_root)
        data["video_path"] = fixed_video_path

        sampling_fps = compute_sampling_fps(domain, data, fixed_video_path)
        timestamps = make_timestamps(fixed_video_path, sampling_fps)
        capability = infer_capability(data.get("question_type", ""))

        return TaskRecord(
            index=index,
            submission_ref=submission_item,
            data=data,
            domain=domain,
            capability=capability,
            sampling_fps=sampling_fps,
            video_path=fixed_video_path,
            timestamps=timestamps,
        )

    # ------------------------------------------------------------------
    # COR: Capability-Oriented Router
    # ------------------------------------------------------------------
    def capability_oriented_router(
        self,
        domains: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[TaskRecord]]:
        """Route each sample by dataset/domain and question capability."""

        allowed = set(domains or DOMAIN_ORDER)
        routed: Dict[str, List[TaskRecord]] = {domain: [] for domain in DOMAIN_ORDER if domain in allowed}

        for i, sub_item in enumerate(self.submission_data):
            testbed_item: Optional[Dict[str, Any]] = None
            for key in item_key_candidates(sub_item):
                testbed_item = self.testbed_index.get(key)
                if testbed_item is not None:
                    break
            if testbed_item is None and i < len(self.testbed_data):
                testbed_item = self.testbed_data[i]
            if testbed_item is None:
                print(f"[WARN] Cannot match submission item at index {i}; skip.")
                continue

            domain = infer_domain(sub_item, testbed_item)
            if domain is None:
                dataset = sub_item.get("dataset") or testbed_item.get("dataset")
                print(f"[WARN] Unknown dataset/domain for item {item_key_candidates(sub_item)} ({dataset}); skip.")
                continue
            if domain not in allowed:
                continue

            task = self.temporal_evidence_normalization(i, sub_item, testbed_item, domain)
            routed.setdefault(domain, []).append(task)

        return routed

    def load_runtime(self, domain: str) -> DomainRuntime:
        if self._runtime and self._runtime.domain == domain:
            return self._runtime
        unload_runtime(self._runtime)

        module = import_domain_module(domain)
        patch_domain_module(module, domain, self.model_paths, self.dataset_root)
        model_path = self.model_paths[domain]

        print(f"\n[OmniEgo-R2/COR] Load {domain} semantic basis")
        print(f"  - model: {model_path}")
        print(f"  - basis: {SEMANTIC_BASES[domain]}")

        if domain == "Surgery":
            model, processor = module.load_domain_model(model_path)
        else:
            model, processor = module.load_domain_model(domain)

        self._runtime = DomainRuntime(domain=domain, module=module, model=model, processor=processor, model_path=model_path)
        return self._runtime

    # ------------------------------------------------------------------
    # RDR: Role-Decomposed Reasoning
    # ------------------------------------------------------------------
    def role_decomposed_reasoning(self, runtime: DomainRuntime, task: TaskRecord) -> Dict[str, Any]:
        """Run the domain-specific reasoning routine selected by COR."""

        domain = task.domain
        module = runtime.module
        task_payload = {"submission_ref": task.submission_ref, "data": task.data}

        print("\n" + "=" * 72)
        print(f"[OmniEgo-R2] Task index={task.index} id={task.submission_ref.get('id', task.submission_ref.get('question_id', 'Unknown'))}")
        print(f"  TEN: {len(task.timestamps)} timestamped frames, fps={task.sampling_fps}")
        print(f"  COR: domain={task.domain}, capability={task.capability}")
        print(f"  Semantic basis: {SEMANTIC_BASES[task.domain]}")
        print("=" * 72)

        if domain in {"Animal", "XSports"}:
            # Reuses the original multi-agent prompts and call chain:
            # run_animal.process_single_task_with_agents / run_xsports.process_single_task_with_agents
            final_response, role1_output, role2_output = module.process_single_task_with_agents(
                runtime.model,
                runtime.processor,
                task_payload,
            )
            return {
                "final_response": final_response,
                "role1_output": role1_output,
                "role2_output": role2_output,
            }

        if domain == "Industry":
            # Reuses the original ENIGMA taxonomy and single-agent expert prompts:
            # run_industry.process_single_task
            final_response = module.process_single_task(runtime.model, runtime.processor, task_payload)
            return {"final_response": final_response}

        if domain == "Surgery":
            # Reconstructs the local prompt from run_surgery.main() and reuses
            # run_surgery.inference_agent plus run_surgery.parse_answer.
            final_response = self._run_surgery_expert(runtime, task)
            return {"final_response": final_response}

        raise ValueError(f"Unsupported domain: {domain}")

    def _run_surgery_expert(self, runtime: DomainRuntime, task: TaskRecord) -> str:
        data = task.data
        options_str = format_options(data.get("options", []))
        prompt = SURGERY_EXPERT_PROMPT.format(
            question_text=data.get("question_text", ""),
            options_str=options_str,
            original_fps=data.get("original_video_fps", 1.0),
            sampling_fps=task.sampling_fps,
        )
        return runtime.module.inference_agent(
            model=runtime.model,
            processor=runtime.processor,
            prompt=prompt,
            video_path=task.video_path,
            start_frame_index=0,
            sampling_fps=task.sampling_fps,
        )

    # ------------------------------------------------------------------
    # BOV: Boundary-aware Option Verification
    # ------------------------------------------------------------------
    def boundary_aware_option_verification(self, task: TaskRecord, reasoning: Dict[str, Any]) -> str:
        """Return the option-oriented verifier output.

        The verification rules are embedded in the original domain prompts:
        - Animal/XSports Role3 prompts compare options after perception/dynamics.
        - Industry prompts treat options as constrained hypotheses.
        - Surgery prompt cross-references observations with options.
        Therefore this stage records the verified final response rather than
        issuing an additional model call that would change the original behavior.
        """

        final_response = reasoning.get("final_response", "")
        if not isinstance(final_response, str):
            final_response = str(final_response)
        return final_response

    # ------------------------------------------------------------------
    # DAC: Defensive Answer Calibration
    # ------------------------------------------------------------------
    def defensive_answer_calibration(self, runtime: DomainRuntime, verified_response: str) -> str:
        """Recover a valid A/B/C/D label using each domain script's parser."""

        answer = runtime.module.parse_answer(verified_response)
        answer = str(answer).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            print(f"[WARN] Parser returned invalid label {answer!r}; fallback to A.")
            answer = "A"
        return answer

    def process_task(self, runtime: DomainRuntime, task: TaskRecord) -> None:
        reasoning = self.role_decomposed_reasoning(runtime, task)
        verified_response = self.boundary_aware_option_verification(task, reasoning)
        answer = self.defensive_answer_calibration(runtime, verified_response)

        task.submission_ref["answer"] = answer
        if self.keep_debug_fields:
            task.submission_ref["omniego_domain"] = task.domain
            task.submission_ref["omniego_capability"] = task.capability
            task.submission_ref["omniego_sampling_fps"] = task.sampling_fps
            if "role1_output" in reasoning:
                task.submission_ref["raw_output_role1"] = reasoning["role1_output"]
            if "role2_output" in reasoning:
                task.submission_ref["raw_output_role2"] = reasoning["role2_output"]
            if task.domain == "Surgery":
                task.submission_ref["raw_output_surgery"] = verified_response
            else:
                task.submission_ref["raw_output_final"] = verified_response

        print(f"[OmniEgo-R2/DAC] calibrated answer = {answer}")

    def run(self, domains: Optional[Iterable[str]] = None) -> Dict[str, int]:
        routed = self.capability_oriented_router(domains)
        counts: Dict[str, int] = {}
        try:
            for domain in DOMAIN_ORDER:
                tasks = routed.get(domain, [])
                if not tasks:
                    continue
                runtime = self.load_runtime(domain)
                print(f"\n[OmniEgo-R2] Start domain={domain}, tasks={len(tasks)}")
                for task in tasks:
                    try:
                        self.process_task(runtime, task)
                    except Exception as exc:
                        print(f"[ERROR] domain={domain} task_index={task.index}: {exc}")
                        task.submission_ref["answer"] = "A"
                        if self.keep_debug_fields:
                            task.submission_ref["omniego_domain"] = task.domain
                            task.submission_ref["omniego_capability"] = task.capability
                            task.submission_ref["raw_output_final"] = f"ERROR: {exc}"
                counts[domain] = len(tasks)
        finally:
            unload_runtime(self._runtime)
            self._runtime = None
        return counts


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_domains(value: str) -> List[str]:
    if not value or value.lower() == "all":
        return list(DOMAIN_ORDER)
    aliases = {d.lower(): d for d in DOMAIN_ORDER}
    domains: List[str] = []
    for token in value.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"unknown domain {token!r}; choose from {', '.join(DOMAIN_ORDER)}")
        domains.append(aliases[key])
    return domains


def parse_model_overrides(values: Optional[Sequence[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--model must be DOMAIN=/path/to/model")
        domain, path = value.split("=", 1)
        domain = domain.strip()
        if domain not in DOMAIN_ORDER:
            raise argparse.ArgumentTypeError(f"unknown domain {domain!r}")
        overrides[domain] = path.strip()
    return overrides


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the unified OmniEgo-R2 routed reasoning pipeline over EgoCross submission data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--testbed",
        default=None,
        help="Path to egocross_testbed_imgs.json. If omitted, tries datasets/egocross_testbed_imgs.json then egocross_testbed_imgs.json.",
    )
    parser.add_argument(
        "--submission",
        default=None,
        help="Path to submission template/input JSON. If omitted, tries submission_template.json then merged_all_answers_ours.json.",
    )
    parser.add_argument("--output", default="submission_omniego_r2.json", help="Final clean submission JSON path.")
    parser.add_argument("--debug-output", default="submission_omniego_r2_debug.json", help="Debug JSON path with raw model outputs.")
    parser.add_argument("--dataset-root", default="./EgoCross-main/datasets/egocross_testbed", help="Local root corresponding to /egocross_testbed/.")
    parser.add_argument("--domains", type=parse_domains, default=list(DOMAIN_ORDER), help="Comma-separated domains or 'all'.")
    parser.add_argument("--model", action="append", default=None, help="Override one model path, e.g. --model Animal=/path/to/animal.")
    parser.add_argument("--no-debug-fields", action="store_true", help="Do not keep raw outputs in the in-memory debug data.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    testbed_path = Path(args.testbed) if args.testbed else resolve_existing_path(
        ["datasets/egocross_testbed_imgs.json", "egocross_testbed_imgs.json"]
    )
    submission_path = Path(args.submission) if args.submission else resolve_existing_path(
        ["submission_template.json", "merged_all_answers_ours.json"]
    )

    if not testbed_path.exists():
        parser.error(f"testbed JSON not found: {testbed_path}")
    if not submission_path.exists():
        parser.error(f"submission JSON not found: {submission_path}")

    testbed_data = read_json(testbed_path)
    submission_data = read_json(submission_path)
    if not isinstance(testbed_data, list):
        parser.error(f"testbed JSON must be a list: {testbed_path}")
    if not isinstance(submission_data, list):
        parser.error(f"submission JSON must be a list: {submission_path}")

    model_overrides = parse_model_overrides(args.model)
    pipeline = OmniEgoR2Pipeline(
        testbed_data=testbed_data,
        submission_data=submission_data,
        dataset_root=args.dataset_root,
        model_paths=model_overrides,
        keep_debug_fields=not args.no_debug_fields,
    )

    print("[OmniEgo-R2] Unified routed reasoning pipeline")
    print(f"  testbed:    {testbed_path}")
    print(f"  submission: {submission_path}")
    print(f"  domains:    {', '.join(args.domains)}")
    print(f"  output:     {args.output}")
    print(f"  debug:      {args.debug_output}")

    counts = pipeline.run(args.domains)

    if not args.no_debug_fields:
        write_json(Path(args.debug_output), submission_data)
    write_json(Path(args.output), clean_submission_for_final(submission_data))

    print("\n[OmniEgo-R2] Done")
    for domain in DOMAIN_ORDER:
        if domain in counts:
            print(f"  {domain}: {counts[domain]} tasks")
    print(f"  final submission: {Path(args.output).resolve()}")
    if not args.no_debug_fields:
        print(f"  debug output:     {Path(args.debug_output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
