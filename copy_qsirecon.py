#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import TrkFile


FIB_RE = re.compile(r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_space-ACPC_model-gqi_dwimap\.fib\.gz$")
BUNDLE_RE = re.compile(r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_space-ACPC_model-gqi_bundle-(?P<bundle>.+)_streamlines\.tck\.gz$")
CONVERSION_MODE = "tck_acpc_rasmm_to_qa_voxmm_trk"

DEFAULT_BUNDLE_MANIFEST = [
    {"official_base": "AssociationCingulum", "target_tract": "CG", "hemis": ["L", "R"]},
    {"official_base": "ProjectionBrainstemCorticospinalTract", "target_tract": "CST", "hemis": ["L", "R"]},
    {"official_base": "ProjectionBasalGangliaFornix", "target_tract": "FX", "hemis": ["L", "R"]},
    {"official_base": "AssociationInferiorFrontoOccipitalFasciculus", "target_tract": "IFOF", "hemis": ["L", "R"]},
    {"official_base": "AssociationInferiorLongitudinalFasciculus", "target_tract": "ILF", "hemis": ["L", "R"]},
    {"official_base": "ProjectionBasalGangliaOpticRadiation", "target_tract": "OR", "hemis": ["L", "R"]},
    {"official_base": "AssociationUncinateFasciculus", "target_tract": "UF", "hemis": ["L", "R"]},
]


@dataclass
class CopyTask:
    subject: str
    session: str
    file_type: str
    official_bundle: str
    target_tract: str
    hemi: str
    source: str
    reference_nii: str
    destination: str
    conversion_mode: str


@dataclass
class CopyResult:
    subject: str
    session: str
    file_type: str
    official_bundle: str
    target_tract: str
    hemi: str
    source: str
    reference_nii: str
    destination: str
    conversion_mode: str
    status: str
    message: str


def resolve_qsirecon_root(source_root: Path) -> Path:
    source_root = source_root.expanduser().resolve()
    if source_root.name == "qsirecon-DSIStudio":
        return source_root
    candidate = source_root / "qsirecon-DSIStudio"
    if candidate.exists():
        return candidate.resolve()
    return source_root


def qsi_dwi_dir(qsi_root: Path, subject: str, session: str) -> Path:
    return qsi_root / subject / session / "dwi"


def fib_source_path(qsi_root: Path, subject: str, session: str) -> Path:
    return qsi_dwi_dir(qsi_root, subject, session) / f"{subject}_{session}_space-ACPC_model-gqi_dwimap.fib.gz"


def qa_json_source_path(qsi_root: Path, subject: str, session: str) -> Path:
    return qsi_dwi_dir(qsi_root, subject, session) / f"{subject}_{session}_space-ACPC_model-gqi_param-qa_dwimap.json"


def qa_nii_source_path(qsi_root: Path, subject: str, session: str) -> Path:
    return qsi_dwi_dir(qsi_root, subject, session) / f"{subject}_{session}_space-ACPC_model-gqi_param-qa_dwimap.nii.gz"


def bundle_source_path(qsi_root: Path, subject: str, session: str, official_bundle: str) -> Path:
    return qsi_dwi_dir(qsi_root, subject, session) / f"{subject}_{session}_space-ACPC_model-gqi_bundle-{official_bundle}_streamlines.tck.gz"


def pipeline_trk_name(subject: str, session: str, target_tract: str, hemi: str) -> str:
    return f"{target_tract}_{hemi}_{subject}_{session}_T2w.trk"


def discover_existing_recon_session_dirs(recon_root: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for session_dir in sorted(recon_root.glob("sub-*/ses-*")):
        if not session_dir.is_dir():
            continue
        subject = session_dir.parent.name
        session = session_dir.name
        rows.append((subject, session, session_dir))
    return rows


def discover_existing_bundle_dirs(bundle_root: Path) -> list[tuple[str, str, str, Path]]:
    rows: list[tuple[str, str, str, Path]] = []
    for tract_dir in sorted(bundle_root.glob("sub-*/ses-*/*_tracts")):
        if not tract_dir.is_dir():
            continue
        subject = tract_dir.parents[1].name
        session = tract_dir.parent.name
        target_tract = tract_dir.name[:-7]
        rows.append((subject, session, target_tract, tract_dir))
    return rows


def manifest_by_target_tract() -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for item in DEFAULT_BUNDLE_MANIFEST:
        target = item["target_tract"]
        official_base = item["official_base"]
        hemis = item["hemis"]
        out.setdefault(target, [])
        for hemi in hemis:
            out[target].append((f"{official_base}{hemi}", hemi))
    return out


def normalize_filters(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(x).strip() for x in values if str(x).strip()}


def copy_file(src: Path, dst: Path, overwrite: bool) -> tuple[str, str]:
    if not src.exists():
        return "missing_source", "source file does not exist"
    if not src.is_file():
        return "invalid_source", "source path is not a file"
    if dst.exists() and not overwrite:
        if dst.stat().st_size == src.stat().st_size:
            return "existing", "destination exists with matching size"
        return "exists_size_mismatch", "destination exists with different size; use --overwrite to replace"
    tmp = dst.with_name(dst.name + ".tmpcopy")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return "copied", ""


def convert_tck_to_dsi_voxmm_trk(tck_path: Path, ref_nii_path: Path, out_trk_path: Path, overwrite: bool) -> tuple[str, str]:
    if not tck_path.exists():
        return "missing_source", "source .tck.gz file does not exist"
    if not ref_nii_path.exists():
        return "missing_reference_nii", "ACPC QA dwimap NIfTI reference does not exist"
    if out_trk_path.exists() and not overwrite:
        return "existing", "destination .trk exists; use --overwrite to replace"

    ref = nib.load(str(ref_nii_path))
    inv_aff = np.linalg.inv(ref.affine)
    zooms = np.asarray(ref.header.get_zooms()[:3], dtype=np.float32)
    shape = np.asarray(ref.shape[:3], dtype=np.int16)

    tck = nib.streamlines.load(str(tck_path))
    out_streamlines: list[np.ndarray] = []
    for sl in tck.streamlines:
        sl = np.asarray(sl, dtype=np.float32)
        if sl.shape[0] < 2:
            continue
        vox = apply_affine(inv_aff, sl).astype(np.float32)
        vox[:, 0] = (shape[0] - 1) - vox[:, 0]
        voxmm = vox * zooms
        out_streamlines.append(voxmm.astype(np.float32))

    if not out_streamlines:
        return "empty_streamlines", "source tract contains no streamlines with at least two points"

    header = TrkFile.create_empty_header()
    header["dimensions"] = shape
    header["voxel_sizes"] = zooms
    header["voxel_to_rasmm"] = np.diag([zooms[0], zooms[1], zooms[2], 1.0]).astype(np.float32)
    header["voxel_order"] = b"RAS"

    tractogram = Tractogram(out_streamlines, affine_to_rasmm=np.eye(4))
    out_trk_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_trk_path.with_name(out_trk_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    nib.streamlines.save(TrkFile(tractogram, header=header), str(tmp))
    tmp.replace(out_trk_path)
    return "converted", f"{len(out_streamlines)} streamlines converted"


def plan_copy_tasks(
    qsi_root: Path,
    recon_root: Path,
    bundle_root: Path,
    subjects_filter: set[str] | None,
    sessions_filter: set[str] | None,
) -> list[CopyTask]:
    tasks: list[CopyTask] = []

    for subject, session, recon_dir in discover_existing_recon_session_dirs(recon_root):
        if subjects_filter is not None and subject not in subjects_filter:
            continue
        if sessions_filter is not None and session not in sessions_filter:
            continue
        for file_type, src in [
            ("fib", fib_source_path(qsi_root, subject, session)),
            ("qa_json", qa_json_source_path(qsi_root, subject, session)),
            ("qa_nii", qa_nii_source_path(qsi_root, subject, session)),
        ]:
            dst = recon_dir / src.name
            tasks.append(CopyTask(subject, session, file_type, "", "", "", str(src), "", str(dst), "copy"))

    target_manifest = manifest_by_target_tract()
    for subject, session, target_tract, tract_dir in discover_existing_bundle_dirs(bundle_root):
        if subjects_filter is not None and subject not in subjects_filter:
            continue
        if sessions_filter is not None and session not in sessions_filter:
            continue
        ref_nii = qa_nii_source_path(qsi_root, subject, session)
        for official_bundle, hemi in target_manifest.get(target_tract, []):
            src = bundle_source_path(qsi_root, subject, session, official_bundle)
            dst = tract_dir / pipeline_trk_name(subject, session, target_tract, hemi)
            tasks.append(CopyTask(subject, session, "bundle_trk", official_bundle, target_tract, hemi, str(src), str(ref_nii), str(dst), CONVERSION_MODE))

    return tasks


def execute_task(task: CopyTask, overwrite: bool, dry_run: bool) -> CopyResult:
    src = Path(task.source)
    dst = Path(task.destination)
    ref = Path(task.reference_nii) if task.reference_nii else None
    if not dst.parent.exists():
        return CopyResult(**asdict(task), status="missing_target_dir", message="destination parent directory does not exist")
    if dry_run:
        if not src.exists() or not src.is_file():
            return CopyResult(**asdict(task), status="missing_source", message="source file does not exist")
        if task.file_type == "bundle_trk" and (ref is None or not ref.exists()):
            return CopyResult(**asdict(task), status="missing_reference_nii", message="ACPC QA dwimap NIfTI reference does not exist")
        return CopyResult(**asdict(task), status="planned", message="dry run")
    if task.file_type == "bundle_trk":
        status, message = convert_tck_to_dsi_voxmm_trk(src, ref, dst, overwrite)  # type: ignore[arg-type]
    else:
        status, message = copy_file(src, dst, overwrite)
    return CopyResult(**asdict(task), status=status, message=message)


def write_manifest(rows: list[CopyResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(CopyResult("", "", "", "", "", "", "", "", "", "", "", "")).keys())
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_subjects_json(rows: list[CopyResult], out_json: Path) -> None:
    fib_rows = [r for r in rows if r.file_type == "fib" and r.status in {"copied", "existing", "planned"}]
    fib_rows.sort(key=lambda r: (r.subject, r.session))
    data = [{"subject": r.subject, "session": r.session, "fib": r.destination} for r in fib_rows]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy official QSIRecon/DSI Studio FIB and QA files and convert official .tck.gz bundles to DSI-compatible pipeline-named .trk files in existing target folders.")
    p.add_argument("--source-root", required=True, type=Path, help="Path to qsirecon-DSIStudio or its parent directory.")
    p.add_argument("--recon-root", required=True, type=Path, help="Existing destination recon root, e.g. /nfs/masi/taylogc1/HBCD_diff_morph/dsi_recon.")
    p.add_argument("--bundle-root", required=True, type=Path, help="Existing destination bundle root, e.g. /nfs/masi/taylogc1/HBCD_diff_morph/DSI_tracts_autotrack.")
    p.add_argument("--subjects", nargs="*", default=None, help="Optional subject filters. Default: all existing target subject folders.")
    p.add_argument("--sessions", nargs="*", default=None, help="Optional session filters. Default: all existing target session folders.")
    p.add_argument("--manifest-csv", type=Path, default=None, help="Copy/convert manifest CSV. Default: <bundle-root>/official_bundle_copy_manifest.csv")
    p.add_argument("--subjects-json", type=Path, default=None, help="subjects.json output path. Default: <recon-root>/subjects.json")
    p.add_argument("--no-subjects-json", action="store_true", help="Do not write subjects.json.")
    p.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    p.add_argument("--dry-run", action="store_true", help="Plan copies/conversions and write manifest, but do not copy or convert files.")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel copy/conversion workers.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qsi_root = resolve_qsirecon_root(args.source_root)
    recon_root = args.recon_root.expanduser().resolve()
    bundle_root = args.bundle_root.expanduser().resolve()
    manifest_csv = args.manifest_csv or (bundle_root / "official_bundle_copy_manifest.csv")
    subjects_json = args.subjects_json or (recon_root / "subjects.json")
    subjects_filter = normalize_filters(args.subjects)
    sessions_filter = normalize_filters(args.sessions)

    if not qsi_root.exists():
        raise FileNotFoundError(f"QSIRecon source root does not exist: {qsi_root}")
    if not recon_root.exists():
        raise FileNotFoundError(f"Destination recon root does not exist: {recon_root}")
    if not bundle_root.exists():
        raise FileNotFoundError(f"Destination bundle root does not exist: {bundle_root}")

    tasks = plan_copy_tasks(qsi_root, recon_root, bundle_root, subjects_filter, sessions_filter)
    if not tasks:
        raise RuntimeError("No copy/conversion tasks were planned. Check existing target folders and optional subject/session filters.")

    results: list[CopyResult] = []
    workers = max(int(args.workers), 1)
    if workers == 1:
        for task in tasks:
            results.append(execute_task(task, args.overwrite, args.dry_run))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(execute_task, task, args.overwrite, args.dry_run): task for task in tasks}
            for fut in as_completed(future_map):
                results.append(fut.result())

    results.sort(key=lambda r: (r.subject, r.session, r.file_type, r.target_tract, r.hemi, r.official_bundle))
    write_manifest(results, manifest_csv)
    if not args.no_subjects_json:
        write_subjects_json(results, subjects_json)

    counts: dict[str, int] = {}
    for row in results:
        counts[row.status] = counts.get(row.status, 0) + 1
    print(f"Wrote manifest: {manifest_csv}")
    if not args.no_subjects_json:
        print(f"Wrote subjects.json: {subjects_json}")
    print("Status counts:", counts)
    print(f"Bundle conversion mode: {CONVERSION_MODE}")


if __name__ == "__main__":
    main()
