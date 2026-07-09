#!/usr/bin/env python
"""Optional GEPA optimization stage for the DSPy OCR signature.

This stage keeps the held-out Testing split intact, inserts missing Training and
Validation rows into the local LanceDB table, optimizes the OCR signature with
GEPA, and saves the resulting DSPy program state for later use by
``src/02_ocr.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
from PIL import Image

from config import REFLECTION_MODEL, SOURCE_DATASET, VISION_MODEL, ensure_output_dir, load_local_env
from data import load_training_labels, load_validation_labels
from db import open_table
from dspy_programs import api_key_env_for_model, litellm_model_id, make_handwriting_reader
from metrics import edit_distance, normalize_text


def limit_value(value: int | None) -> int | None:
    if value is None or value == 0:
        return None
    return value


def existing_split_ids(split: str) -> set[str]:
    table = open_table()
    rows = (
        table.search()
        .select(["id"])
        .where(f"split = '{split}'")
        .limit(max(table.count_rows(), 1))
        .to_list()
    )
    return {row["id"] for row in rows}


def ensure_split_rows(split: str, id_prefix: str, labels_loader) -> int:
    table = open_table()
    existing = existing_split_ids(split)
    labels = labels_loader()
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(labels.iter_rows(named=True)):
        row_id = f"{id_prefix}_{i:05d}"
        if row_id in existing:
            continue
        image_path = row["image_path"]
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        rows.append(
            {
                "id": row_id,
                "split": split,
                "image_filename": row["IMAGE"],
                "image": image_bytes,
                "medicine_name": row["MEDICINE_NAME"],
                "generic_name": row["GENERIC_NAME"],
                "source_dataset": SOURCE_DATASET,
            }
        )
    if rows:
        table.add(rows)
    return len(rows)


def rows_for_split(split: str, limit: int | None) -> list[dict]:
    table = open_table()
    row_limit = limit or table.count_rows()
    return (
        table.search()
        .select(["id", "split", "image", "medicine_name", "generic_name"])
        .where(f"split = '{split}'")
        .limit(row_limit)
        .to_list()
    )


def shuffle_rows(rows: list[dict], seed: int) -> list[dict]:
    shuffled = sorted(rows, key=lambda row: row["id"])
    random.Random(seed).shuffle(shuffled)
    return shuffled


def sample_rows(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    return rows[:limit]


def make_examples(rows: list[dict], *, dspy) -> list:
    examples = []
    for row in rows:
        image = Image.open(io.BytesIO(row["image"]))
        image.load()
        examples.append(
            dspy.Example(
                image=dspy.Image(image),
                medicine_name=row["medicine_name"],
                id=row["id"],
            ).with_inputs("image")
        )
    return examples


def score_prediction(
    predicted: str | None,
    gold: str,
    metric_style: str,
) -> tuple[float, int, bool, bool, float]:
    distance = edit_distance(predicted, gold)
    label = normalize_text(gold)
    predicted_norm = normalize_text(predicted)
    normalized_match = predicted_norm == label
    exact_match = (predicted or "").strip() == gold.strip()
    edit_similarity = max(0.0, 1.0 - (distance / max(len(label), 1)))
    if metric_style == "normalized_weighted_edit":
        score = 0.90 * float(normalized_match) + 0.10 * edit_similarity
    elif metric_style == "normalized_exact":
        score = 1.0 if normalized_match else 0.0
    elif normalized_match:
        score = 1.0 if exact_match else 0.95
    else:
        score = 0.25 * edit_similarity
    return score, distance, normalized_match, exact_match, edit_similarity


def output_shape_penalty(predicted: str, target: str) -> tuple[float, list[str]]:
    penalty = 1.0
    issues: list[str] = []
    predicted_lower = predicted.lower()
    predicted_tokens = predicted.split()
    target_tokens = target.split()

    banned_terms = {"tab", "tablet", "cap", "capsule", "syr", "syrup", "inj", "injection", "mg", "ml"}
    if any(re.search(rf"\b{re.escape(term)}\.?\b", predicted_lower) for term in banned_terms):
        penalty *= 0.90
        issues.append("output contains dosage/form/prefix text instead of only the handwritten medicine name")

    if len(predicted_tokens) > max(len(target_tokens), 1):
        penalty *= 0.95
        issues.append("output has more whitespace-separated tokens than the target transcription")

    unsupported_punctuation = set(predicted) & set(",/;:()[]{}")
    if unsupported_punctuation:
        penalty *= 0.95
        issues.append(f"output contains unsupported punctuation: {''.join(sorted(unsupported_punctuation))!r}")

    predicted_norm = normalize_text(predicted)
    target_norm = normalize_text(target)
    if predicted_norm and target_norm:
        length_delta = abs(len(predicted_norm) - len(target_norm))
        if length_delta >= 3:
            penalty *= 0.97
            issues.append("output length differs substantially from the target, suggesting inserted or dropped letters")

    if not predicted_norm:
        penalty *= 0.50
        issues.append("output is empty after normalization")

    return penalty, issues


def ocr_gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, *, metric_style: str):
    predicted = str(getattr(pred, "ocr_text", "") or "").strip()
    target = str(gold.medicine_name).strip()
    base_score, distance, normalized_match, exact_match, edit_similarity = score_prediction(
        predicted,
        target,
        metric_style,
    )
    penalty, shape_issues = output_shape_penalty(predicted, target)
    score = base_score if metric_style == "normalized_exact" else base_score * penalty
    issue_feedback = (
        " Output-shape issues: " + "; ".join(shape_issues) + "."
        if shape_issues
        else " Output shape is acceptable: one concise transcription with no obvious extra metadata."
    )
    objective_feedback = (
        "Metric style: normalized_weighted_edit. The score is a simple weighted objective: "
        "90% normalized exact match and 10% edit-distance similarity. "
        "This strongly prioritizes normalized accuracy while still giving GEPA directional signal "
        "for near misses. "
        if metric_style == "normalized_weighted_edit"
        else
        "Metric style: normalized_exact. The score is exactly the normalized OCR accuracy objective: "
        "1.0 for a normalized exact match and 0.0 for any normalized miss. "
        if metric_style == "normalized_exact"
        else "Metric style: normalized_edit_tiebreak. Normalized exact match dominates, and edit-distance similarity "
        "only provides a capped tie-breaker for normalized misses. "
    )
    feedback = (
        f"Target transcription: {target!r}. Predicted OCR text: {predicted!r}. "
        f"Exact string match: {exact_match}. Normalized exact match: {normalized_match}. "
        f"Edit distance: {distance}. Edit similarity: {edit_similarity:.3f}. "
        f"Base score: {base_score:.3f}. Final score after output-shape penalty: {score:.3f}. "
        f"{issue_feedback} "
        "This is primarily a visual OCR task. "
        f"{objective_feedback}"
        "Exact casing/punctuation and output shape are secondary tie-breakers after normalized match. "
        "The instruction should improve general handwriting reading behavior without listing specific "
        "training labels, medicine names, or prior answers from your memorized knowledge. "
        "Medicine-style spelling may be used only as a weak tie-breaker when the letters are visually ambiguous; "
        "it should not override the visible strokes. "
        "Prefer instructions that help the model check the specific letter shapes, avoid inserted "
        "or dropped letters, and return one concise transcription. "
        "Avoid extra words, dosage markers, explanations, and multiple guesses, but do not make the instruction "
        "so restrictive that it ignores plausible visually supported medicine-name spellings."
    )
    return dspy.Prediction(score=score, feedback=feedback)


def make_ocr_gepa_metric(metric_style: str):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return ocr_gepa_metric(
            gold,
            pred,
            trace=trace,
            pred_name=pred_name,
            pred_trace=pred_trace,
            metric_style=metric_style,
        )

    return metric


def write_summary(
    output_path: Path,
    *,
    args: argparse.Namespace,
    program_output: Path,
    train_count: int,
    val_count: int,
) -> None:
    payload = {
        "student_model": VISION_MODEL,
        "reflection_model": args.reflection_model,
        "metric_style": args.metric_style,
        "train_examples": train_count,
        "val_examples": val_count,
        "program_output": str(program_output),
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as output_file:
        json.dump({"summary": payload}, output_file, indent=2)

    body = [
        "# OCR GEPA Optimization",
        "",
        f"- Student model: `{VISION_MODEL}`",
        f"- Reflection model: `{args.reflection_model}`",
        f"- Metric style: `{args.metric_style}`",
        f"- Training examples: {train_count}",
        f"- Validation examples: {val_count}",
        f"- Optimized program: `{program_output}`",
        "",
        "## Next Step",
        "",
        "Apply this program with `src/02_ocr.py --program-path`, then run `src/07_evaluate.py` on Testing.",
        "",
    ]
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(body))


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Optimize the DSPy OCR signature with GEPA on local LanceDB data.")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Training rows to sample after shuffling. Defaults to 0, which means all Training rows.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=192,
        help="Validation rows to sample after shuffling. Defaults to 192. Use 0 for all Validation rows.",
    )
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--max-full-evals", type=int, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument(
        "--metric-style",
        choices=["normalized_weighted_edit", "normalized_exact", "normalized_edit_tiebreak"],
        default="normalized_weighted_edit",
        help="GEPA score objective. Defaults to 90%% normalized match and 10%% edit similarity.",
    )
    parser.add_argument("--reflection-model", default=REFLECTION_MODEL)
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=None,
        help="Defaults to 3 for light runs and 8 for medium/heavy or explicit-budget runs.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Defaults to 1 for light runs and 4 for medium/heavy or explicit-budget runs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--program-output", default=str(ensure_output_dir() / "optimized_ocr_reader.json"))
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    explicit_budget = args.max_full_evals is not None or args.max_metric_calls is not None
    if explicit_budget:
        args.auto = None
    if args.max_full_evals is not None and args.max_metric_calls is not None:
        parser.error("Set only one explicit GEPA budget: --max-full-evals or --max-metric-calls.")
    larger_run = args.auto in {"medium", "heavy"} or explicit_budget
    if args.reflection_minibatch_size is None:
        args.reflection_minibatch_size = 8 if larger_run else 3
    if args.num_threads is None:
        args.num_threads = 4 if larger_run else 1
    if args.reflection_minibatch_size < 1:
        parser.error("--reflection-minibatch-size must be at least 1.")
    if args.num_threads < 1:
        parser.error("--num-threads must be at least 1.")

    train_cap = limit_value(args.train_limit)
    val_cap = limit_value(args.val_limit)
    added_train = ensure_split_rows("Training", "train", load_training_labels)
    added_val = ensure_split_rows("Validation", "val", load_validation_labels)
    print(f"Inserted {added_train} missing Training rows into the local LanceDB table.")
    print(f"Inserted {added_val} missing Validation rows into the local LanceDB table.")

    train_rows = sample_rows(shuffle_rows(rows_for_split("Training", None), args.seed), train_cap)
    val_rows = sample_rows(shuffle_rows(rows_for_split("Validation", None), args.seed + 1), val_cap)
    if not train_rows or not val_rows:
        raise RuntimeError("GEPA optimization requires at least one Training row and one Validation row.")

    dspy, student = make_handwriting_reader()
    trainset = make_examples(train_rows, dspy=dspy)
    valset = make_examples(val_rows, dspy=dspy)

    reflection_model_id = litellm_model_id(args.reflection_model)
    reflection_key_env = api_key_env_for_model(reflection_model_id)
    if reflection_key_env and not os.environ.get(reflection_key_env):
        raise RuntimeError(f"{reflection_key_env} is required for GEPA reflection with {reflection_model_id}.")
    reflection_lm = dspy.LM(reflection_model_id, temperature=1.0, max_tokens=32000, cache=False)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or str(ensure_output_dir("optimization", f"ocr_gepa_{run_id}", "logs"))

    optimizer = dspy.GEPA(
        metric=make_ocr_gepa_metric(args.metric_style),
        auto=args.auto,
        max_full_evals=args.max_full_evals,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.reflection_minibatch_size,
        instruction_proposer=__import__(
            "dspy.teleprompt.gepa.instruction_proposal",
            fromlist=["MultiModalInstructionProposer"],
        ).MultiModalInstructionProposer(),
        num_threads=args.num_threads,
        log_dir=log_dir,
        track_stats=True,
        seed=args.seed,
    )
    optimized = optimizer.compile(student, trainset=trainset, valset=valset)

    program_output = Path(args.program_output)
    program_output.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(program_output))
    print(f"Saved optimized OCR DSPy program to {program_output}.")

    summary_dir = program_output.parent / "optimization"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"ocr_gepa_{run_id}.md"
    write_summary(
        summary_path,
        args=args,
        program_output=program_output,
        train_count=len(trainset),
        val_count=len(valset),
    )
    print(f"Wrote optimization report to {summary_path}.")


if __name__ == "__main__":
    main()
