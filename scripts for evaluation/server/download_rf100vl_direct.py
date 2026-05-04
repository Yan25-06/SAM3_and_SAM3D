#!/usr/bin/env python3
"""Download RF100-VL COCO exports directly from Roboflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import roboflow
from roboflow import Project

try:
    from rf100vl.util import get_basename
except Exception:
    def get_basename(dataset_name: str) -> str:
        return dataset_name.strip().lower().replace(" ", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RF100-VL via Roboflow REST export links")
    parser.add_argument("--root", required=True, help="Target root directory")
    parser.add_argument("--variant", default="full", choices=("full", "fsod"))
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_api_key(api_key: Optional[str]) -> str:
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY")
    return api_key


def dataset_ready(dataset_dir: Path) -> bool:
    return (
        (dataset_dir / "train" / "_annotations.coco.json").is_file()
        and (dataset_dir / "test" / "_annotations.coco.json").is_file()
    )


def clean_coco_annotations(dataset_dir: Path) -> None:
    for split in ("train", "test", "valid"):
        ann_path = dataset_dir / split / "_annotations.coco.json"
        if not ann_path.is_file():
            continue

        with ann_path.open("r", encoding="utf-8") as f:
            data_ann = json.load(f)

        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(get_clean_ann_data(data_ann), f)


def get_clean_ann_data(data_ann: dict) -> dict:
    categories = data_ann.get("categories") or []
    if not categories:
        return data_ann
    if categories[0].get("supercategory") != "none":
        return data_ann

    new_data_ann = {}
    if data_ann.get("info"):
        new_data_ann["info"] = data_ann["info"]
    if data_ann.get("licenses"):
        new_data_ann["licenses"] = data_ann["licenses"]
    new_data_ann["categories"] = [
        {
            "id": cat["id"] - 1,
            "name": cat["name"],
            "supercategory": cat["supercategory"],
        }
        for cat in categories
        if cat["id"] != 0
    ]
    new_data_ann["images"] = deepcopy(data_ann.get("images", []))
    new_data_ann["annotations"] = deepcopy(data_ann.get("annotations", []))

    annotations = new_data_ann["annotations"]
    if annotations:
        annotation_ids_shift = 1 - min(z["id"] for z in annotations)
        for ann in annotations:
            ann["category_id"] = ann["category_id"] - 1
            ann["id"] = ann["id"] + annotation_ids_shift

    return new_data_ann


def normalize_extracted_layout(dataset_dir: Path) -> None:
    if dataset_ready(dataset_dir):
        return

    nested_candidates = []
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        if dataset_ready(child):
            nested_candidates.append(child)
            continue
        if (child / "train").is_dir() or (child / "test").is_dir() or (child / "valid").is_dir():
            nested_candidates.append(child)

    if len(nested_candidates) != 1:
        return

    nested_root = nested_candidates[0]
    for nested_child in nested_root.iterdir():
        shutil.move(str(nested_child), str(dataset_dir / nested_child.name))
    shutil.rmtree(nested_root)


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def get_workspace_id(variant: str) -> str:
    return "rf100-vl-fsod" if variant == "fsod" else "rf100-vl"


def get_projects(api_key: str, workspace_id: str) -> List[Tuple[str, Project]]:
    rf = roboflow.Roboflow(api_key=api_key)
    workspace = rf.workspace(workspace_id)
    projects: List[Tuple[str, Project]] = []
    for project_data in workspace.project_list:
        project = Project(api_key=rf.api_key, a_project=project_data, model_format="coco")
        projects.append((get_basename(project.name), project))
    return sorted(projects, key=lambda item: item[0])


def get_project_slug(project: Project) -> str:
    project_id = getattr(project, "id", "")
    if isinstance(project_id, str) and "/" in project_id:
        return project_id.split("/", 1)[1]
    if isinstance(project_id, str) and project_id:
        return project_id
    raise RuntimeError(f"Unable to determine project slug for {project.name!r}")


def get_latest_version_number(project: Project) -> str:
    versions = project.versions()
    if not versions:
        raise RuntimeError(f"No versions found for {project.name!r}")

    def version_key(version: object) -> int:
        for attr in ("id", "version"):
            value = getattr(version, attr, None)
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        raise RuntimeError(f"Unable to determine version number for {project.name!r}")

    latest = max(versions, key=version_key)
    return str(version_key(latest))


def get_export_link(api_key: str, workspace_id: str, project_slug: str, version_number: str) -> str:
    encoded_key = urllib.parse.quote(api_key, safe="")
    url = (
        f"https://api.roboflow.com/{workspace_id}/{project_slug}/{version_number}/coco"
        f"?api_key={encoded_key}"
    )
    payload = fetch_json(url)
    link = (payload.get("export") or {}).get("link")
    if not link:
        raise RuntimeError(f"Missing export link for {workspace_id}/{project_slug}/{version_number}")
    return link


def download_project(
    api_key: str,
    workspace_id: str,
    basename: str,
    project: Project,
    root_dir: Path,
    overwrite: bool,
) -> None:
    dataset_dir = root_dir / basename
    if dataset_ready(dataset_dir) and not overwrite:
        print(f"[SKIP] {basename}")
        return

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    project_slug = get_project_slug(project)
    version_number = get_latest_version_number(project)
    print(f"[DOWN] {basename}: {project_slug}/{version_number}")

    export_link = get_export_link(api_key, workspace_id, project_slug, version_number)

    with tempfile.TemporaryDirectory(prefix=f"rf100_{basename}_") as tmpdir:
        zip_path = Path(tmpdir) / "dataset.zip"
        download_file(export_link, zip_path)
        extract_zip(zip_path, dataset_dir)

    normalize_extracted_layout(dataset_dir)
    clean_coco_annotations(dataset_dir)

    if not dataset_ready(dataset_dir):
        raise RuntimeError(f"{basename}: expected train/test COCO annotations were not found")

    print(f"[ OK ] {basename}")


def main() -> int:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)
    root_dir = Path(args.root).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    workspace_id = get_workspace_id(args.variant)
    failures = []
    for basename, project in get_projects(api_key, workspace_id):
        try:
            download_project(api_key, workspace_id, basename, project, root_dir, args.overwrite)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failures.append((basename, str(exc)))
            print(f"[ERR ] {basename}: {exc}", file=sys.stderr)

    if failures:
        print("\nDownload completed with failures:", file=sys.stderr)
        for basename, message in failures:
            print(f"  - {basename}: {message}", file=sys.stderr)
        return 1

    print(f"\nDownloaded RF100-VL successfully to {root_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
