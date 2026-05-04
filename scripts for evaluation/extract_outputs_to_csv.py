#!/usr/bin/env python3
"""Extract result summaries from outputs/ into CSV files."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ODINW_DATASETS = [
    "AerialMaritimeDrone_large",
    "Aquarium",
    "CottontailRabbits",
    "EgoHands_generic",
    "NorthAmericaMushrooms",
    "Packages",
    "PascalVOC",
    "Raccoon",
    "ShellfishOpenImages",
    "VehiclesOpenImages",
    "pistols",
    "pothole",
    "thermalDogsAndPeople",
]

TABLE3_RUNS = [
    "table3_odinw_text_and_visual",
    "table3_odinw_text_only",
    "table3_odinw_visual_only",
]

SACO_TABLE1_SUBSETS = [
    "gold_attributes",
    "gold_crowded",
    "gold_fg_food",
    "gold_fg_sports_equipment",
    "gold_metaclip_nps",
    "gold_sa1b_nps",
    "gold_wiki_common",
]

PRESENCE_HEAD_CONDITIONS = {
    "with_presence": [
        "gold_metaclip_nps",
    ],
    "without_presence": [
        "gold_metaclip_nps",
    ],
}

DETECTION_METRICS = {
    "bbox_ap": "coco_eval_bbox_AP",
    "bbox_ap50": "coco_eval_bbox_AP_50",
    "bbox_ap75": "coco_eval_bbox_AP_75",
    "bbox_ap_small": "coco_eval_bbox_AP_small",
    "bbox_ap_medium": "coco_eval_bbox_AP_medium",
    "bbox_ap_large": "coco_eval_bbox_AP_large",
    "bbox_ar_1": "coco_eval_bbox_AR_maxDets@1",
    "bbox_ar_10": "coco_eval_bbox_AR_maxDets@10",
    "bbox_ar_100": "coco_eval_bbox_AR_maxDets@100",
    "bbox_ar_small": "coco_eval_bbox_AR_small",
    "bbox_ar_medium": "coco_eval_bbox_AR_medium",
    "bbox_ar_large": "coco_eval_bbox_AR_large",
}

LVIS_METRICS = {
    "bbox_ap": "lvis_eval_bbox_AP",
    "bbox_ap50": "lvis_eval_bbox_AP50",
    "bbox_ap75": "lvis_eval_bbox_AP75",
    "bbox_ap_rare": "lvis_eval_bbox_APr",
    "bbox_ap_common": "lvis_eval_bbox_APc",
    "bbox_ap_frequent": "lvis_eval_bbox_APf",
    "bbox_ap_small": "lvis_eval_bbox_APs",
    "bbox_ap_medium": "lvis_eval_bbox_APm",
    "bbox_ap_large": "lvis_eval_bbox_APl",
    "bbox_ar_300": "lvis_eval_bbox_AR@300",
    "bbox_ar_small_300": "lvis_eval_bbox_ARs@300",
    "bbox_ar_medium_300": "lvis_eval_bbox_ARm@300",
    "bbox_ar_large_300": "lvis_eval_bbox_ARl@300",
}

CGF1_METRICS = {
    "bbox_cgf1": "cgF1_eval_bbox_cgF1",
    "bbox_precision": "cgF1_eval_bbox_precision",
    "bbox_recall": "cgF1_eval_bbox_recall",
    "bbox_f1": "cgF1_eval_bbox_F1",
    "bbox_positive_macro_f1": "cgF1_eval_bbox_positive_macro_F1",
    "bbox_positive_micro_f1": "cgF1_eval_bbox_positive_micro_F1",
    "bbox_positive_micro_precision": "cgF1_eval_bbox_positive_micro_precision",
    "bbox_il_precision": "cgF1_eval_bbox_IL_precision",
    "bbox_il_recall": "cgF1_eval_bbox_IL_recall",
    "bbox_il_f1": "cgF1_eval_bbox_IL_F1",
    "bbox_il_fpr": "cgF1_eval_bbox_IL_FPR",
    "bbox_il_mcc": "cgF1_eval_bbox_IL_MCC",
    "bbox_cgf1_50": "cgF1_eval_bbox_cgF1@0.5",
    "bbox_precision_50": "cgF1_eval_bbox_precision@0.5",
    "bbox_recall_50": "cgF1_eval_bbox_recall@0.5",
    "bbox_f1_50": "cgF1_eval_bbox_F1@0.5",
    "bbox_positive_macro_f1_50": "cgF1_eval_bbox_positive_macro_F1@0.5",
    "bbox_positive_micro_f1_50": "cgF1_eval_bbox_positive_micro_F1@0.5",
    "bbox_positive_micro_precision_50": "cgF1_eval_bbox_positive_micro_precision@0.5",
    "bbox_cgf1_75": "cgF1_eval_bbox_cgF1@0.75",
    "bbox_precision_75": "cgF1_eval_bbox_precision@0.75",
    "bbox_recall_75": "cgF1_eval_bbox_recall@0.75",
    "bbox_f1_75": "cgF1_eval_bbox_F1@0.75",
    "bbox_positive_macro_f1_75": "cgF1_eval_bbox_positive_macro_F1@0.75",
    "bbox_positive_micro_f1_75": "cgF1_eval_bbox_positive_micro_F1@0.75",
    "bbox_positive_micro_precision_75": "cgF1_eval_bbox_positive_micro_precision@0.75",
    "segm_cgf1": "cgF1_eval_segm_cgF1",
    "segm_precision": "cgF1_eval_segm_precision",
    "segm_recall": "cgF1_eval_segm_recall",
    "segm_f1": "cgF1_eval_segm_F1",
    "segm_positive_macro_f1": "cgF1_eval_segm_positive_macro_F1",
    "segm_positive_micro_f1": "cgF1_eval_segm_positive_micro_F1",
    "segm_positive_micro_precision": "cgF1_eval_segm_positive_micro_precision",
    "segm_il_precision": "cgF1_eval_segm_IL_precision",
    "segm_il_recall": "cgF1_eval_segm_IL_recall",
    "segm_il_f1": "cgF1_eval_segm_IL_F1",
    "segm_il_fpr": "cgF1_eval_segm_IL_FPR",
    "segm_il_mcc": "cgF1_eval_segm_IL_MCC",
    "segm_cgf1_50": "cgF1_eval_segm_cgF1@0.5",
    "segm_precision_50": "cgF1_eval_segm_precision@0.5",
    "segm_recall_50": "cgF1_eval_segm_recall@0.5",
    "segm_f1_50": "cgF1_eval_segm_F1@0.5",
    "segm_positive_macro_f1_50": "cgF1_eval_segm_positive_macro_F1@0.5",
    "segm_positive_micro_f1_50": "cgF1_eval_segm_positive_micro_F1@0.5",
    "segm_positive_micro_precision_50": "cgF1_eval_segm_positive_micro_precision@0.5",
    "segm_cgf1_75": "cgF1_eval_segm_cgF1@0.75",
    "segm_precision_75": "cgF1_eval_segm_precision@0.75",
    "segm_recall_75": "cgF1_eval_segm_recall@0.75",
    "segm_f1_75": "cgF1_eval_segm_F1@0.75",
    "segm_positive_macro_f1_75": "cgF1_eval_segm_positive_macro_F1@0.75",
    "segm_positive_micro_f1_75": "cgF1_eval_segm_positive_micro_F1@0.75",
    "segm_positive_micro_precision_75": "cgF1_eval_segm_positive_micro_precision@0.75",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to the outputs directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/summary"),
        help="Directory where CSV files will be written",
    )
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_by_suffix(
    payload: Optional[Dict[str, float]], metric_suffixes: Dict[str, str]
) -> Dict[str, Optional[float]]:
    row: Dict[str, Optional[float]] = {key: None for key in metric_suffixes}
    if not payload:
        return row
    for out_key, suffix in metric_suffixes.items():
        for metric_key, value in payload.items():
            if metric_key.endswith(suffix):
                row[out_key] = value
                break
    return row


def average_rows(
    rows: Iterable[Dict[str, Optional[float]]], metric_names: Iterable[str]
) -> Dict[str, Optional[float]]:
    averaged: Dict[str, Optional[float]] = {}
    row_list = list(rows)
    for metric in metric_names:
        values = [row[metric] for row in row_list if row.get(metric) is not None]
        averaged[metric] = (sum(values) / len(values)) if values else None
    return averaged


def summarize_status(statuses: Iterable[str]) -> str:
    unique_statuses = set(statuses)
    if not unique_statuses or unique_statuses == {"missing"}:
        return "missing"
    if unique_statuses == {"ok"}:
        return "ok"
    return "partial_missing"


def write_csv(path: Path, rows: List[Dict[str, Optional[float]]]) -> None:
    all_fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_table1_rows(outputs_dir: Path) -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []

    lvis_payload = load_json(outputs_dir / "table1_lvis" / "logs" / "val_stats.json")
    lvis_row: Dict[str, Optional[float]] = {
        "evaluation": "table1_lvis",
        "subset": "table1_lvis",
        "dataset_family": "lvis",
        "num_subsets": 1,
        "status": "ok" if lvis_payload else "missing",
    }
    lvis_row.update(extract_by_suffix(lvis_payload, LVIS_METRICS))
    rows.append(lvis_row)

    saco_rows: List[Dict[str, Optional[float]]] = []
    for subset in SACO_TABLE1_SUBSETS:
        payload = load_json(
            outputs_dir / "table1_saco_gold" / subset / "logs" / "val_stats.json"
        )
        row: Dict[str, Optional[float]] = {
            "evaluation": "table1_saco_gold",
            "subset": subset,
            "dataset_family": "saco_gold",
            "num_subsets": 1,
            "status": "ok" if payload else "missing",
        }
        row.update(extract_by_suffix(payload, CGF1_METRICS))
        saco_rows.append(row)

    rows.extend(saco_rows)

    average_row: Dict[str, Optional[float]] = {
        "evaluation": "table1_saco_gold",
        "subset": "saco_gold_average",
        "dataset_family": "saco_gold",
        "num_subsets": len(saco_rows),
        "status": summarize_status(row["status"] for row in saco_rows),
    }
    average_row.update(average_rows(saco_rows, CGF1_METRICS))
    rows.append(average_row)
    return rows


def summarize_detection_run(
    outputs_dir: Path, run_name: str, dataset_family: str, datasets: List[str]
) -> Dict[str, Optional[float]]:
    subset_rows: List[Dict[str, Optional[float]]] = []
    subset_statuses: List[str] = []
    for dataset in datasets:
        payload = load_json(outputs_dir / run_name / "logs" / dataset / "val_stats.json")
        subset_rows.append(extract_by_suffix(payload, DETECTION_METRICS))
        subset_statuses.append("ok" if payload else "missing")

    row: Dict[str, Optional[float]] = {
        "evaluation": run_name,
        "dataset_family": dataset_family,
        "num_subsets": len(datasets),
        "status": summarize_status(subset_statuses),
    }
    row.update(average_rows(subset_rows, DETECTION_METRICS))
    return row


def build_table2_rows(outputs_dir: Path) -> List[Dict[str, Optional[float]]]:
    rf100_datasets = sorted(
        path.name for path in (outputs_dir / "table2_rf100_zero_shot" / "logs").iterdir()
        if path.is_dir()
    )
    return [
        summarize_detection_run(
            outputs_dir, "table2_odinw_zero_shot", "odinw35", ODINW_DATASETS
        ),
        summarize_detection_run(
            outputs_dir, "table2_rf100_zero_shot", "roboflow100", rf100_datasets
        ),
    ]


def build_table3_rows(outputs_dir: Path) -> List[Dict[str, Optional[float]]]:
    return [
        summarize_detection_run(outputs_dir, run_name, "odinw35", ODINW_DATASETS)
        for run_name in TABLE3_RUNS
    ]


def build_presence_head_rows(
    outputs_dir: Path,
) -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []
    # The current repo still stores presence-head runs under this folder name.
    base_dir = outputs_dir / "saco_gold_presence_ablation"

    for condition, subsets in PRESENCE_HEAD_CONDITIONS.items():
        subset_rows: List[Dict[str, Optional[float]]] = []
        for subset in subsets:
            payload = load_json(base_dir / condition / subset / "logs" / "val_stats.json")
            row: Dict[str, Optional[float]] = {
                "evaluation": "presence_head",
                "condition": condition,
                "subset": subset,
                "dataset_family": "saco_gold",
                "num_subsets": 1,
                "status": "ok" if payload else "missing",
            }
            row.update(extract_by_suffix(payload, CGF1_METRICS))
            rows.append(row)
            subset_rows.append(row)

        average_row: Dict[str, Optional[float]] = {
            "evaluation": "presence_head",
            "condition": condition,
            "subset": f"{condition}_average",
            "dataset_family": "saco_gold",
            "num_subsets": len(subsets),
            "status": summarize_status(row["status"] for row in subset_rows),
        }
        average_row.update(average_rows(subset_rows, CGF1_METRICS))
        rows.append(average_row)

    return rows


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    out_dir = args.out_dir.resolve()

    write_csv(out_dir / "table1.csv", build_table1_rows(outputs_dir))
    write_csv(out_dir / "table2.csv", build_table2_rows(outputs_dir))
    write_csv(out_dir / "table3.csv", build_table3_rows(outputs_dir))
    write_csv(
        out_dir / "presence_head.csv",
        build_presence_head_rows(outputs_dir),
    )


if __name__ == "__main__":
    main()
