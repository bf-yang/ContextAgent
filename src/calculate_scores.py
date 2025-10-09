# -*- coding: utf-8 -*-
import os
import ast
import csv
import json
import sys
import argparse
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()
from transformers.utils.versions import require_version
require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

import ollama  # noqa: F401

from utils import (
    azure_inference, ollama_inference,  # noqa: F401
    calculate_set_metrics,
    calculate_accuracy,
    calculate_regression_metrics
)

# ============================================================
# Path builders (same naming logic as your original)
# ============================================================

def build_pred_path(dataset: str,
                    methods: str,
                    model_base_icl: str,
                    model_base_sft: str,
                    zs: str,
                    personas: str,
                    think: str) -> str:
    if methods == 'icl':
        if zs == 'true':
            return f'results/{dataset}/predictions/{methods}/pred_{model_base_icl}_zs.json'
        return f'results/{dataset}/predictions/{methods}/pred_{model_base_icl}_fs_10_{personas}_{think}.json'
    # sft
    return f'results/{dataset}/predictions/{methods}/pred_{model_base_sft}_{personas}_{think}.json'


def build_csv_path(dataset: str,
                   methods: str,
                   model_base_icl: str,
                   model_base_sft: str,
                   zs: str,
                   personas: str,
                   think: str) -> str:
    if methods == 'icl':
        if zs == 'true':
            return f'results/{dataset}/scores/metrics_{methods}_{model_base_icl}_zs.csv'
        return f'results/{dataset}/scores/metrics_{methods}_{model_base_icl}_fs_10_{personas}_{think}.csv'
    # sft
    return f'results/{dataset}/scores/metrics_{methods}_{model_base_sft}_{personas}_{think}.csv'


# ============================================================
# Helpers (robust parsing & sanitization)
# ============================================================

def safe_eval_tools(tools_str: str) -> List[str]:
    """
    Parse tools from a json-like string (list of dicts with 'name').

    Returns:
        ['None']  if tools_str == 'None'
        ['Error'] if parsing fails or unexpected format
        [name...] otherwise
    """
    if tools_str == 'None':
        return ['None']
    try:
        obj = ast.literal_eval(tools_str) if isinstance(tools_str, str) else tools_str
    except (SyntaxError, ValueError):
        return ['Error']

    if isinstance(obj, list):
        names = []
        for item in obj:
            if isinstance(item, dict) and "name" in item:
                names.append(item["name"])
            else:
                return ['Error']
        return names
    return ['Error']


def parse_tools_object(tools_str_or_list):
    """Return a Python list of tool dicts; [] on failure."""
    if tools_str_or_list == 'None':
        return []
    if isinstance(tools_str_or_list, list):
        return tools_str_or_list
    try:
        obj = ast.literal_eval(tools_str_or_list)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def assign_level_by_gt_tool_count(gt_tool_names_set: set) -> str:
    """
      0/1 -> level1
      2   -> level2
      3+  -> level3
    """
    n = len(gt_tool_names_set)
    if n in (0, 1):
        return "level1"
    if n == 2:
        return "level2"
    return "level3"


def normalize_proactive_idx(val: str) -> str:
    """
    Normalize proactive index to 'true'/'false':
      - lowercase first letter
      - 'medium' treated as 'true'
    """
    if not val:
        return 'false'
    s = str(val)
    s = s[0].lower() + s[1:] if s else s
    if s == "medium":
        s = "true"
    return s


def threshold_idx_from_score(score_str: str) -> str:
    """
    >= 3 => 'true', else 'false'. 'None' -> 0
    """
    s = '0' if score_str == 'None' else score_str
    try:
        v = int(s)
    except Exception:
        v = 0
    return "true" if v >= 3 else "false"


def sanitize_scores(seq: List[str]) -> Tuple[List[float], int]:
    """
    Convert list[str] -> list[float] robustly.
    - Accepts 'None'/None/''/non-numeric → coerce to 0.0 and count as an error.
    Returns: (floats, num_errors)
    """
    out: List[float] = []
    errors = 0
    for x in seq:
        if x is None:
            out.append(0.0); errors += 1; continue
        s = str(x).strip()
        if s.lower() == 'none' or s == '':
            out.append(0.0); errors += 1; continue
        try:
            out.append(float(s))
        except Exception:
            out.append(0.0); errors += 1
    return out, errors


def is_none_like(x) -> bool:
    """True if x is conceptually None (None / 'None' / '')"""
    if x is None:
        return True
    s = str(x).strip().lower()
    return s == 'none' or s == ''


# ============================================================
# Core collectors & evaluators
# ============================================================

def load_predictions(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def initialize_level_metrics() -> Dict[str, Dict]:
    def blank():
        return {
            "gt_tools_list": [],
            "pred_tools_list": [],
            "gt_proactive_idx_list": [],
            "pred_proactive_idx_list": [],
            "gt_proactive_score_list": [],
            "pred_proactive_score_list": [],
            "P_tool_names_list": [],
            "R_tool_names_list": [],
            "F1_tool_names_list": [],
            "arg_false": 0,   # per-level numerator
            "arg_counts": 0   # per-level denominator
        }
    return {"level1": blank(), "level2": blank(), "level3": blank()}


def collect_metrics(dataset: Dict):
    """
    Iterate predictions, compute level-wise containers + overall lists/counters.

    Returns:
        level_metrics: dict(level -> metrics lists & per-level arg counters)
        level_indices: dict(level -> sample keys)
        global_lists : overall lists for macro evaluation
        counters     : overall counters including arg_false/arg_counts
    """
    level_metrics = initialize_level_metrics()
    level_indices: Dict[str, List[str]] = {"level1": [], "level2": [], "level3": []}

    # Global containers
    gt_proactive_idx_list, pred_proactive_idx_list = [], []
    gt_proactive_score_list, pred_proactive_score_list = [], []
    P_tool_names_list, R_tool_names_list, F1_tool_names_list = [], [], []

    # Overall argument counters
    arg_false_overall, arg_counts_overall = 0, 0

    # Stats
    cnt_proactive, cnt_non_proactive = 0, 0
    tool_level_hist: List[int] = []

    for key in tqdm(dataset.keys()):
        data = dataset[key]

        # Ground truth
        gt_proactive_idx = normalize_proactive_idx(str(data.get('Proactive index', 'false')))
        gt_proactive_score = str(data.get('Proactive score', '0'))
        gt_tools = data.get('Tools', 'None')

        # Predictions (robust access)
        preds = data.get("predictions", {})
        pred_proactive_idx = normalize_proactive_idx(preds.get('proactive_idx', 'false'))
        pred_proactive_score = preds.get('proactive_score', 'None')
        pred_tools = preds.get('tools', 'None')

        # Normalize idx via score threshold to match reference logic
        # (use score thresholds unconditionally for GT; for Pred, threshold if available else 'false')
        gt_proactive_idx = threshold_idx_from_score(gt_proactive_score)
        if pred_proactive_score != 'None':
            pred_proactive_idx = threshold_idx_from_score(pred_proactive_score)
        else:
            pred_proactive_idx = "false"

        # Tool names & sets
        pred_tool_names = safe_eval_tools(pred_tools)
        gt_tool_names = safe_eval_tools(gt_tools)
        pred_tool_names_set = set(pred_tool_names)
        gt_tool_names_set = set(gt_tool_names)

        # Level bucketing by GT tool count
        tool_level_hist.append(len(gt_tool_names_set))
        level = assign_level_by_gt_tool_count(gt_tool_names_set)
        level_indices[level].append(key)

        # Set metrics (precision/recall/F1 over tool-name sets)
        P, R, F1 = calculate_set_metrics(pred_tool_names_set, gt_tool_names_set, "Tool Names")
        P_tool_names_list.append(P)
        R_tool_names_list.append(R)
        F1_tool_names_list.append(F1)

        # ---------- Argument Accuracy counting (match reference) ----------
        # 1) Proactive mismatch contributes: GT true & Pred false => error
        if gt_proactive_idx == 'true' and pred_proactive_idx == 'false':
            arg_false_overall += 1
            arg_counts_overall += 1
            level_metrics[level]["arg_false"] += 1
            level_metrics[level]["arg_counts"] += 1

        # 2) Tool results errors: count error if tool not in GT, or results contain 'error'
        tools_results = preds.get('tools_results', 'None')
        if tools_results != 'None':
            for tool_result in tools_results:
                tool_name = tool_result.get("tool_name")
                results_str = str(tool_result.get("results", ""))

                # Counters (overall + per-level)
                arg_counts_overall += 1
                level_metrics[level]["arg_counts"] += 1

                is_error = False
                if tool_name not in gt_tool_names_set:
                    is_error = True
                else:
                    if "error" in results_str.lower():
                        is_error = True

                if is_error:
                    arg_false_overall += 1
                    level_metrics[level]["arg_false"] += 1
        # -----------------------------------------------------------------

        # Global lists
        gt_proactive_idx_list.append(gt_proactive_idx)
        pred_proactive_idx_list.append(pred_proactive_idx)
        gt_proactive_score_list.append(gt_proactive_score)
        pred_proactive_score_list.append(pred_proactive_score)

        # Level lists
        lm = level_metrics[level]
        lm["gt_tools_list"].append(gt_tools)
        lm["pred_tools_list"].append(pred_tools)
        lm["gt_proactive_idx_list"].append(gt_proactive_idx)
        lm["pred_proactive_idx_list"].append(pred_proactive_idx)
        lm["gt_proactive_score_list"].append(gt_proactive_score)
        lm["pred_proactive_score_list"].append(pred_proactive_score)
        lm["P_tool_names_list"].append(P)
        lm["R_tool_names_list"].append(R)
        lm["F1_tool_names_list"].append(F1)

        # Stats
        if gt_proactive_idx == 'false':
            cnt_non_proactive += 1
        else:
            cnt_proactive += 1

    global_lists = {
        "gt_idx": gt_proactive_idx_list,
        "pred_idx": pred_proactive_idx_list,
        "gt_score": gt_proactive_score_list,
        "pred_score": pred_proactive_score_list,
        "P_tool": P_tool_names_list,
        "R_tool": R_tool_names_list,
        "F1_tool": F1_tool_names_list,
    }
    counters = {
        "cnt_proactive": cnt_proactive,
        "cnt_non_proactive": cnt_non_proactive,
        "tool_level_hist": tool_level_hist,
        "arg_false": arg_false_overall,
        "arg_counts": arg_counts_overall,
    }
    return level_metrics, level_indices, global_lists, counters


def evaluate_level(level_name: str, metrics: Dict[str, Dict]) -> None:
    """Pretty-print metrics for a single level, robust to malformed scores."""
    print(f"Evaluating {level_name}")
    print("=" * 50)

    # Tool names F1/PR
    print(f"Average Precision for Tool Names: {np.mean(metrics[level_name]['P_tool_names_list']):.4f}")
    print(f"Average Recall for Tool Names:    {np.mean(metrics[level_name]['R_tool_names_list']):.4f}")
    print(f"Average F1 for Tool Names:        {np.mean(metrics[level_name]['F1_tool_names_list']):.4f}")

    # Argument accuracy (per level)
    arg_false = metrics[level_name]["arg_false"]
    arg_counts = metrics[level_name]["arg_counts"]
    arg_acc = 1 - (arg_false / arg_counts) if arg_counts > 0 else 0.0
    print(f"Argument Accuracy: {arg_acc*100:.2f}%")

    # Proactive Index Acc
    print(50 * "=")
    acc_idx, incorrect_idx, miss_needed, false_detection = calculate_accuracy(
        metrics[level_name]['pred_proactive_idx_list'],
        metrics[level_name]['gt_proactive_idx_list'],
        "Proactive Index"
    )
    print(f"Accuracy for Proactive Index:  {acc_idx:.2%}")
    print(f"Incorrect indices (Index):     {incorrect_idx}")
    print(f"Miss-Needed Rate:              {miss_needed:.2%}")
    print(f"False-Detection Rate:          {false_detection:.2%}")

    # Proactive Score Acc
    print(50 * "=")
    acc_score, incorrect_score, _, _ = calculate_accuracy(
        metrics[level_name]['pred_proactive_score_list'],
        metrics[level_name]['gt_proactive_score_list'],
        "Proactive Score"
    )
    print(f"Accuracy for Proactive Score:  {acc_score:.2%}")
    print(f"Incorrect indices (Score):     {incorrect_score}")

    # Proactive Score RMSE —— sanitize to avoid ValueError
    print(50 * "=")
    pred_scores_f, err_p = sanitize_scores(metrics[level_name]['pred_proactive_score_list'])
    gt_scores_f, err_g = sanitize_scores(metrics[level_name]['gt_proactive_score_list'])
    _, rmse = calculate_regression_metrics(pred_scores_f, gt_scores_f, "Proactive Score")
    print(f"RMSE for Proactive Score:      {rmse:.4f}")
    if (err_p + err_g) > 0:
        print(f"(Note) Sanitized {err_p + err_g} malformed score(s) in {level_name}.")
    print(50 * "=")


def evaluate_by_level(level_metrics: Dict[str, Dict]) -> None:
    print("\033[94m" + 50 * "*" + "\033[0m")
    for lv in ["level1", "level2", "level3"]:
        evaluate_level(lv, level_metrics)
        print("\033[94m" + 50 * "*" + "\033[0m")


def evaluate_overall(global_lists: Dict[str, List[str]],
                     counters: Dict[str, int]) -> Dict[str, float]:
    print("Evaluating Overall")
    print(50 * "=")

    # Tool names macro metrics
    P_mean = np.mean(global_lists["P_tool"])
    R_mean = np.mean(global_lists["R_tool"])
    F1_mean = np.mean(global_lists["F1_tool"])
    print(f"Average Precision for Tool Names: {P_mean:.4f}")
    print(f"Average Recall for Tool Names:    {R_mean:.4f}")
    print(f"Average F1 for Tool Names:        {F1_mean:.4f}")

    # Argument accuracy (overall)
    arg_false = counters["arg_false"]
    arg_counts = counters["arg_counts"]
    arg_acc_overall = 1 - (arg_false / arg_counts) if arg_counts > 0 else 0.0
    print(f"Argument Accuracy: {arg_acc_overall*100:.2f}%")

    # Proactive index accuracy
    print(50 * "=")
    acc_idx, incorrect_idx, miss_needed, false_detection = calculate_accuracy(
        global_lists["pred_idx"], global_lists["gt_idx"], "Proactive Index"
    )
    print(f"Accuracy for Proactive Index:   {acc_idx:.2%}")
    print(f"Incorrect indices (Index):      {incorrect_idx}")
    print(f"Miss-Needed Rate:               {miss_needed:.2%}")
    print(f"False-Detection Rate:           {false_detection:.2%}")

    # Proactive score accuracy
    print(50 * "=")
    acc_score, incorrect_score, _, _ = calculate_accuracy(
        global_lists["pred_score"], global_lists["gt_score"], "Proactive Score"
    )
    print(f"Accuracy for Proactive Score:   {acc_score:.2%}")
    print(f"Incorrect indices (Score):      {incorrect_score}")

    # RMSE —— sanitize to avoid ValueError
    pred_scores_f, err_p = sanitize_scores(global_lists["pred_score"])
    gt_scores_f, err_g = sanitize_scores(global_lists["gt_score"])
    _, rmse_overall = calculate_regression_metrics(pred_scores_f, gt_scores_f, "Proactive Score")
    print(50 * "=")
    print(f"RMSE for Proactive Score:       {rmse_overall:.4f}")
    if (err_p + err_g) > 0:
        print(f"(Note) Sanitized {err_p + err_g} malformed score(s) overall.")
    print(50 * "=")

    return {
        "F1_tool_overall": float(F1_mean),
        "ArgAcc_overall": float(arg_acc_overall),
        "IdxAcc_overall": float(acc_idx),
        "ScoreAcc_overall": float(acc_score),
        "RMSE_overall": float(rmse_overall),
    }


def build_metrics_table(level_metrics: Dict[str, Dict],
                        global_lists: Dict[str, List[str]],
                        counters: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """Build the CSV table: {metric_name: {level1, level2, level3, overall}}."""
    # Overall values
    acc_idx_overall = calculate_accuracy(global_lists["pred_idx"], global_lists["gt_idx"], "Proactive Index")[0]
    miss_needed_overall = calculate_accuracy(global_lists["pred_idx"], global_lists["gt_idx"], "Proactive Index")[2]
    false_det_overall = calculate_accuracy(global_lists["pred_idx"], global_lists["gt_idx"], "Proactive Index")[3]
    acc_score_overall = calculate_accuracy(global_lists["pred_score"], global_lists["gt_score"], "Proactive Index")[0]

    # RMSE overall (robust)
    pred_scores_overall, _ = sanitize_scores(global_lists["pred_score"])
    gt_scores_overall, _ = sanitize_scores(global_lists["gt_score"])
    rmse_overall = calculate_regression_metrics(pred_scores_overall, gt_scores_overall, "Proactive Score")[1]

    arg_false = counters["arg_false"]
    arg_counts = counters["arg_counts"]
    arg_acc_overall = 1 - (arg_false / arg_counts) if arg_counts > 0 else 0.0

    def lv(name: str):
        return level_metrics[name]

    # Per-level RMSE —— sanitize to avoid ValueError
    l1_pred, _ = sanitize_scores(lv("level1")["pred_proactive_score_list"])
    l1_gt, _   = sanitize_scores(lv("level1")["gt_proactive_score_list"])
    l2_pred, _ = sanitize_scores(lv("level2")["pred_proactive_score_list"])
    l2_gt, _   = sanitize_scores(lv("level2")["gt_proactive_score_list"])
    l3_pred, _ = sanitize_scores(lv("level3")["pred_proactive_score_list"])
    l3_gt, _   = sanitize_scores(lv("level3")["gt_proactive_score_list"])

    table = {
        "Proactive_Index_Accuracy": {
            "level1": calculate_accuracy(lv("level1")["pred_proactive_idx_list"], lv("level1")["gt_proactive_idx_list"], "Proactive Index")[0],
            "level2": calculate_accuracy(lv("level2")["pred_proactive_idx_list"], lv("level2")["gt_proactive_idx_list"], "Proactive Index")[0],
            "level3": calculate_accuracy(lv("level3")["pred_proactive_idx_list"], lv("level3")["gt_proactive_idx_list"], "Proactive Index")[0],
            "overall": acc_idx_overall,
        },
        "Miss_Needed": {
            "level1": calculate_accuracy(lv("level1")["pred_proactive_idx_list"], lv("level1")["gt_proactive_idx_list"], "Proactive Index")[2],
            "level2": calculate_accuracy(lv("level2")["pred_proactive_idx_list"], lv("level2")["gt_proactive_idx_list"], "Proactive Index")[2],
            "level3": calculate_accuracy(lv("level3")["pred_proactive_idx_list"], lv("level3")["gt_proactive_idx_list"], "Proactive Index")[2],
            "overall": miss_needed_overall,
        },
        "False_Detection": {
            "level1": calculate_accuracy(lv("level1")["pred_proactive_idx_list"], lv("level1")["gt_proactive_idx_list"], "Proactive Index")[3],
            "level2": calculate_accuracy(lv("level2")["pred_proactive_idx_list"], lv("level2")["gt_proactive_idx_list"], "Proactive Index")[3],
            "level3": calculate_accuracy(lv("level3")["pred_proactive_idx_list"], lv("level3")["gt_proactive_idx_list"], "Proactive Index")[3],
            "overall": false_det_overall,
        },
        "Proactive_Score_Accuracy": {
            "level1": calculate_accuracy(lv("level1")["pred_proactive_score_list"], lv("level1")["gt_proactive_score_list"], "Proactive Index")[0],
            "level2": calculate_accuracy(lv("level2")["pred_proactive_score_list"], lv("level2")["gt_proactive_score_list"], "Proactive Index")[0],
            "level3": calculate_accuracy(lv("level3")["pred_proactive_score_list"], lv("level3")["gt_proactive_score_list"], "Proactive Index")[0],
            "overall": acc_score_overall,
        },
        "RMSE": {
            "level1": calculate_regression_metrics(l1_pred, l1_gt, "Proactive Score")[1],
            "level2": calculate_regression_metrics(l2_pred, l2_gt, "Proactive Score")[1],
            "level3": calculate_regression_metrics(l3_pred, l3_gt, "Proactive Score")[1],
            "overall": rmse_overall,
        },
        "Precision_Tool_Names": {
            "level1": float(np.mean(lv("level1")["P_tool_names_list"])),
            "level2": float(np.mean(lv("level2")["P_tool_names_list"])),
            "level3": float(np.mean(lv("level3")["P_tool_names_list"])),
            "overall": float(np.mean(global_lists["P_tool"])),
        },
        "Recall_Tool_Names": {
            "level1": float(np.mean(lv("level1")["R_tool_names_list"])),
            "level2": float(np.mean(lv("level2")["R_tool_names_list"])),
            "level3": float(np.mean(lv("level3")["R_tool_names_list"])),
            "overall": float(np.mean(global_lists["R_tool"])),
        },
        "F1_Tool_Names": {
            "level1": float(np.mean(lv("level1")["F1_tool_names_list"])),
            "level2": float(np.mean(lv("level2")["F1_tool_names_list"])),
            "level3": float(np.mean(lv("level3")["F1_tool_names_list"])),
            "overall": float(np.mean(global_lists["F1_tool"])),
        },
        "Argument_Accuracy": {
            "level1": 1 - (lv("level1")["arg_false"] / lv("level1")["arg_counts"]) if lv("level1")["arg_counts"] > 0 else 0.0,
            "level2": 1 - (lv("level2")["arg_false"] / lv("level2")["arg_counts"]) if lv("level2")["arg_counts"] > 0 else 0.0,
            "level3": 1 - (lv("level3")["arg_false"] / lv("level3")["arg_counts"]) if lv("level3")["arg_counts"] > 0 else 0.0,
            "overall": 1 - (counters["arg_false"] / counters["arg_counts"]) if counters["arg_counts"] > 0 else 0.0,
        }
    }
    return table


def write_csv(csv_path: str, table: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Level1", "Level2", "Level3", "Overall"])
        for metric_name, values in table.items():
            writer.writerow([
                metric_name,
                round(values["level1"], 4),
                round(values["level2"], 4),
                round(values["level3"], 4),
                round(values["overall"], 4),
            ])
    print(f"[Save] Metrics summary saved to: {os.path.abspath(csv_path)}")


def print_dataset_stats(level_metrics: Dict[str, Dict],
                        counters: Dict[str, int]) -> None:
    print("\033[94m" + 50 * "*" + "\033[0m")
    total = counters["cnt_proactive"] + counters["cnt_non_proactive"]
    print(f"Total examples:             {total}")
    print(f"Proactive examples:         {counters['cnt_proactive']}")
    print(f"Non-proactive examples:     {counters['cnt_non_proactive']}")

    gt_tools_counter = Counter(counters["tool_level_hist"])
    print("GT Tools Count Distribution:")
    for tool_count, count in gt_tools_counter.items():
        print(f"  {tool_count}: {count}")

    for level_name, metrics in level_metrics.items():
        sample_count = len(metrics["gt_tools_list"])
        print(f"Number of Samples for {level_name}: {sample_count}")


def print_digest(summary: Dict[str, float], csv_path: str) -> None:
    print("\n\033[92m" + "=" * 60 + "\033[0m")
    print("Overall Summary (key metrics)")
    print("-" * 60)
    print(f"Proactive Index Accuracy: {summary['IdxAcc_overall']:.2%}")
    print(f"Proactive Score Accuracy: {summary['ScoreAcc_overall']:.2%}")
    print(f"Proactive Score RMSE    : {summary['RMSE_overall']:.4f}")
    print(f"Tool Names F1 (overall): {summary['F1_tool_overall']:.4f}")
    print(f"Argument Accuracy       : {summary['ArgAcc_overall']*100:.2f}%")
    print(f"CSV saved to            : {os.path.abspath(csv_path)}")
    print("\033[92m" + "=" * 60 + "\033[0m")


# ============================================================
# Orchestration
# ============================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cab', help='cab, cab_lite, cab_ood')
    parser.add_argument("--methods", type=str, default='icl', help='icl, sft')
    parser.add_argument("--model_base_sft", type=str, default='llama8b', help='base SFT model: qwen7b, llama8b')
    parser.add_argument("--model_base_icl", type=str, default='qwen2.5:latest', help='qwen2.5:latest, qwen2.5:72b, gpt-4o')
    parser.add_argument("--zs", type=str, default='false', help='false, true')
    parser.add_argument("--think", type=str, default='w_t', help='w_t, wo_t')
    parser.add_argument("--personas", type=str, default='w_p', help='w_p, wo_p')
    return parser.parse_args()


def run(args) -> None:
    # Paths
    pred_path = build_pred_path(
        dataset=args.dataset,
        methods=args.methods,
        model_base_icl=args.model_base_icl,
        model_base_sft=args.model_base_sft,
        zs=args.zs,
        personas=args.personas,
        think=args.think,
    )
    csv_path = build_csv_path(
        dataset=args.dataset,
        methods=args.methods,
        model_base_icl=args.model_base_icl,
        model_base_sft=args.model_base_sft,
        zs=args.zs,
        personas=args.personas,
        think=args.think,
    )

    print(f"[Load] Prediction file: {os.path.abspath(pred_path)}")
    dataset = load_predictions(pred_path)

    # Collect
    level_metrics, level_indices, global_lists, counters = collect_metrics(dataset)

    # Eval (levels + overall)
    evaluate_by_level(level_metrics)
    summary = evaluate_overall(global_lists, counters)

    # Save table
    table = build_metrics_table(level_metrics, global_lists, counters)
    write_csv(csv_path, table)

    # Extra prints (dataset stats + digest)
    print_dataset_stats(level_metrics, counters)
    print("Indices for each level:")
    for lvl, indices in level_indices.items():
        print(f"{lvl}: {indices}")

    print_digest(summary, csv_path)


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
