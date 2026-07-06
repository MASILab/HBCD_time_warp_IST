#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.spatial import cKDTree

DEFAULT_T2W_GLOB = "**/*_logJ_node_summary.csv"
DEFAULT_TRACT_GLOB = "**/*_mask_syn_node_profile.csv"
DEFAULT_T2W_TRK_GLOB = "**/*_filtered_warped_to_*.trk"
DEFAULT_TRACT_TRK_GLOB = "**/*_mask_syn_filtered_warped_to_*.trk"
DEFAULT_BUNDLEWARP_SUMMARY_GLOB = "**/bundlewarp_registration_summary.csv"
DEFAULT_T2W_BUAN_SUMMARY_GLOB = "**/tract_similarity_summary.csv"
DEFAULT_TRACT_BUAN_SUMMARY_GLOB = "**/mask_syn_registration_summary.csv"
DEFAULT_METRIC_CANDIDATES = [
    "mean_logJ",
    "mean_logj",
    "mean_log_j",
    "mean_mask_syn_logJ",
    "mean_mask_syn_logj",
    "logJ",
    "logj",
]
N_BINS = 10
BIN_LABELS = [f"{i * 10}-{(i + 1) * 10}%" for i in range(N_BINS)]

t2w_label = "Anatomical"
tract_label = "Tractography"
hemi_order = ["L", "R"]
heatmap_test = "ttest"
heatmap_use_fdr = True
heatmap_cmap = "coolwarm"
method_disagreement_test = "ttest"
method_use_fdr = True
method_disagreement_cmap = "viridis"
method_disagreement_max_points_per_bin = 25000
buan_test = "ttest"
buan_use_fdr = True
FDR_SIGNIFICANT_TEXT_COLOR = "black"
NON_FDR_TEXT_COLOR = "black"
bundlewarp_distribution_ncols = 3

def _finite_1d(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]

def _safe_nanmean(values):
    vals = _finite_1d(values)
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))

def _safe_nanmedian(values):
    vals = _finite_1d(values)
    if vals.size == 0:
        return np.nan
    return float(np.median(vals))

def _safe_nanstd(values):
    vals = _finite_1d(values)
    if vals.size == 0:
        return np.nan
    return float(np.std(vals))

def set_minimal_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "white",
        "axes.linewidth": 0.0,
        "axes.grid": False,
        "xtick.color": "0.25",
        "ytick.color": "0.25",
        "text.color": "0.15",
        "axes.labelcolor": "0.20",
        "axes.titlecolor": "0.15",
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 9,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })

def remove_axes_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

def minimal_colorbar(cbar):
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=7, colors="0.30")
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

def normalize_subject_id(subject_id):
    if str(subject_id).startswith("sub-"):
        return str(subject_id)
    return f"sub-{subject_id}"

def strip_known_suffixes(path, suffixes):
    name = Path(path).name
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem

def strip_tract_filename_to_key(path):
    name = Path(path).name
    if name.endswith(".trk"):
        name = name[:-4]
    patterns = [
        r"_mask_syn_filtered_warped_to_[A-Za-z0-9\-]+$",
        r"_mask_syn_unfiltered_warped_to_[A-Za-z0-9\-]+$",
        r"_mask_syn_warped_to_[A-Za-z0-9\-]+$",
        r"_filtered_similarity_warped_to_[A-Za-z0-9\-]+$",
        r"_filtered_warped_to_[A-Za-z0-9\-]+$",
        r"_unfiltered_warped_to_[A-Za-z0-9\-]+$",
        r"_warped_to_[A-Za-z0-9\-]+$",
        r"_bundlewarp_nonlinear_moved_filtered$",
        r"_bundlewarp_linear_moved_filtered$",
        r"_bundlewarp_nonlinear_moved_unfiltered$",
        r"_bundlewarp_linear_moved_unfiltered$",
        r"_bundlewarp_nonlinear_moved$",
        r"_bundlewarp_linear_moved$",
    ]
    for pattern in patterns:
        name = re.sub(pattern, "", name)
    return name

def parse_tract_key(tract_key):
    tokens = tract_key.split("_")
    hemi_tokens = {"L", "R", "all"}
    for i, token in enumerate(tokens):
        if token in hemi_tokens and i > 0:
            family = "_".join(tokens[:i])
            hemi = token
            label = f"{family}_{hemi}"
            return family, hemi, label
    if len(tokens) >= 2 and tokens[-1] in hemi_tokens:
        family = "_".join(tokens[:-1])
        hemi = tokens[-1]
        label = f"{family}_{hemi}"
        return family, hemi, label
    return tract_key, "all", tract_key

def find_subject_dirs(root, subjects=None):
    root = Path(root)
    if subjects:
        dirs = []
        for subject in subjects:
            subject = normalize_subject_id(subject)
            d = root / subject
            if d.exists():
                dirs.append(d)
            else:
                print(f"WARNING: subject directory not found: {d}")
        return dirs
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")])

def choose_metric_column(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested metric column {requested} not found. Columns: {list(df.columns)}")
        return requested
    for col in DEFAULT_METRIC_CANDIDATES:
        if col in df.columns:
            return col
    numeric_cols = [c for c in df.columns if c != "node" and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(
        "Could not infer nodewise metric column. "
        f"Expected one of {DEFAULT_METRIC_CANDIDATES}; found columns {list(df.columns)}"
    )

def read_nodewise_csv(csv_path, subject, pipeline, metric_col=None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    col = choose_metric_column(df, metric_col)
    if "node" not in df.columns:
        df = df.copy()
        df["node"] = np.arange(len(df))
    tract_key = strip_known_suffixes(
        csv_path.name,
        suffixes=[
            "_logJ_node_summary.csv",
            "_mask_syn_node_profile.csv",
            "_node_summary.csv",
            ".csv",
        ],
    )
    family, hemi, tract_label = parse_tract_key(tract_key)
    out = pd.DataFrame({
        "pipeline": pipeline,
        "subject": subject,
        "tract_key": tract_key,
        "family": family,
        "hemi": hemi,
        "tract_label": tract_label,
        "node": pd.to_numeric(df["node"], errors="coerce").astype("Int64"),
        "logJ": pd.to_numeric(df[col], errors="coerce"),
        "source_csv": str(csv_path),
    })
    out = out.dropna(subset=["node", "logJ"])
    out["node"] = out["node"].astype(int)
    return out

def load_pipeline_data(root, pipeline, glob_pattern, metric_col=None, subjects=None):
    rows = []
    for subject_dir in find_subject_dirs(root, subjects=subjects):
        subject = subject_dir.name
        files = sorted(subject_dir.glob(glob_pattern))
        if not files:
            print(f"WARNING: no files found for {pipeline} subject {subject} using pattern {glob_pattern}")
        for f in files:
            try:
                rows.append(read_nodewise_csv(f, subject, pipeline, metric_col=metric_col))
            except Exception as exc:
                print(f"WARNING: skipping {f}: {exc}")
    if not rows:
        raise RuntimeError(f"No nodewise data loaded for pipeline {pipeline} from {root}")
    return pd.concat(rows, ignore_index=True)

def load_pipeline_trk_map(root, pipeline, glob_pattern, subjects=None):
    rows = []
    for subject_dir in find_subject_dirs(root, subjects=subjects):
        subject = subject_dir.name
        files = sorted(subject_dir.glob(glob_pattern))
        if not files:
            print(f"WARNING: no .trk files found for {pipeline} subject {subject} using pattern {glob_pattern}")
        for f in files:
            tract_key = strip_tract_filename_to_key(f)
            family, hemi, tract_label = parse_tract_key(tract_key)
            rows.append({
                "pipeline": pipeline,
                "subject": subject,
                "tract_key": tract_key,
                "family": family,
                "hemi": hemi,
                "tract_label": tract_label,
                "trk_path": str(f),
            })
    if not rows:
        return pd.DataFrame(columns=["pipeline", "subject", "tract_key", "family", "hemi", "tract_label", "trk_path"])
    df = pd.DataFrame(rows)
    duplicates = df.duplicated(subset=["pipeline", "subject", "tract_label"], keep=False)
    if duplicates.any():
        dup = df.loc[duplicates, ["pipeline", "subject", "tract_label", "trk_path"]]
        print("WARNING: duplicate .trk matches found; using the first per subject/tract_label:")
        print(dup.to_string(index=False))
    return df.drop_duplicates(subset=["pipeline", "subject", "tract_label"], keep="first")

def ordered_unique(values, requested_order=None):
    vals = list(dict.fromkeys([v for v in values if pd.notna(v)]))
    if requested_order:
        requested = [x for x in requested_order if x in vals]
        rest = sorted([x for x in vals if x not in requested])
        return requested + rest
    return sorted(vals)

def get_symmetric_norm(values, vlim=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if vlim is None:
        if values.size == 0:
            vmax = 1.0
        else:
            vmax = float(np.nanmax(np.abs(values)))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
    else:
        vmax = float(abs(vlim))
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), vmax

def _arc_length_fraction(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if points.shape[0] == 1:
        return np.zeros((1,), dtype=float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = dist[-1]
    if not np.isfinite(total) or total <= 0:
        return np.zeros((points.shape[0],), dtype=float)
    return dist / total

def _node_values_for_streamline_points(streamline_rasmm, node_values):
    node_values = np.asarray(node_values, dtype=float)
    n_nodes = len(node_values)
    node_x = np.linspace(0.0, 1.0, n_nodes)
    pt_x = _arc_length_fraction(streamline_rasmm)
    return np.interp(pt_x, node_x, node_values).astype(np.float32)

def _interpolate_native_at_fraction(arr_native, fractions, target_fraction):
    target_fraction = float(np.clip(target_fraction, 0.0, 1.0))
    out = np.empty((3,), dtype=np.float32)
    for dim in range(3):
        out[dim] = np.interp(target_fraction, fractions, arr_native[:, dim])
    return out

def _streamline_segment_by_fraction(arr_native, arr_rasmm, start_fraction, end_fraction):
    if arr_native.shape[0] < 2:
        return None
    fractions = _arc_length_fraction(arr_rasmm)
    if fractions.shape[0] != arr_native.shape[0] or fractions[-1] <= fractions[0]:
        return None
    start_fraction = float(np.clip(start_fraction, 0.0, 1.0))
    end_fraction = float(np.clip(end_fraction, 0.0, 1.0))
    if end_fraction <= start_fraction:
        return None
    inner = np.where((fractions > start_fraction) & (fractions < end_fraction))[0]
    pts = [_interpolate_native_at_fraction(arr_native, fractions, start_fraction)]
    for idx in inner:
        pts.append(arr_native[idx].astype(np.float32))
    pts.append(_interpolate_native_at_fraction(arr_native, fractions, end_fraction))
    seg = np.asarray(pts, dtype=np.float32)
    if seg.shape[0] < 2:
        return None
    if np.nanmax(np.linalg.norm(np.diff(seg, axis=0), axis=1)) <= 0:
        return None
    return seg

def _trk_header_rasmm_axis_bounds(header, axis=1):
    dims = np.asarray(header.get("dimensions", [0, 0, 0])[:3], dtype=float)
    if dims.size != 3 or np.any(dims <= 0):
        return None
    affine = None
    for key in ["voxel_to_rasmm", "vox_to_ras"]:
        if key in header:
            affine = np.asarray(header[key], dtype=float)
            break
    if affine is None or affine.shape != (4, 4):
        return None
    corners = np.array([
        [0, 0, 0],
        [dims[0] - 1, 0, 0],
        [0, dims[1] - 1, 0],
        [0, 0, dims[2] - 1],
        [dims[0] - 1, dims[1] - 1, 0],
        [dims[0] - 1, 0, dims[2] - 1],
        [0, dims[1] - 1, dims[2] - 1],
        [dims[0] - 1, dims[1] - 1, dims[2] - 1],
    ], dtype=float)
    ras = nib.affines.apply_affine(affine, corners)
    return float(np.nanmin(ras[:, axis])), float(np.nanmax(ras[:, axis]))

def _flip_streamlines_ap_for_dsi_display(segment_streamlines, affine_to_rasmm, trk_header):
    bounds = _trk_header_rasmm_axis_bounds(trk_header, axis=1)
    if bounds is None:
        pts = []
        for sl in segment_streamlines:
            arr = np.asarray(sl, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] > 0:
                pts.append(nib.affines.apply_affine(affine_to_rasmm, arr))
        if not pts:
            return segment_streamlines
        all_pts = np.vstack(pts)
        bounds = (float(np.nanmin(all_pts[:, 1])), float(np.nanmax(all_pts[:, 1])))
    y_min, y_max = bounds
    inv_affine = np.linalg.inv(affine_to_rasmm)
    flipped = []
    for sl in segment_streamlines:
        arr = np.asarray(sl, dtype=np.float32)
        ras = nib.affines.apply_affine(affine_to_rasmm, arr)
        ras[:, 1] = (y_min + y_max) - ras[:, 1]
        flipped.append(nib.affines.apply_affine(inv_affine, ras).astype(np.float32))
    return flipped

def _bin_node_means(node_values, n_bins=N_BINS):
    node_values = np.asarray(node_values, dtype=float)
    idx_bins = np.array_split(np.arange(len(node_values)), n_bins)
    means = []
    for idx in idx_bins:
        if len(idx) == 0:
            means.append(np.nan)
        else:
            means.append(float(np.nanmean(node_values[idx])))
    return np.asarray(means, dtype=float)

def export_dsi_binned_segment_tractograms(in_trk, out_subdir, tract_label, pipeline_name, node_values, cmap_name, norm, n_bins=N_BINS, flip_ap_for_dsi=False):
    in_trk = Path(in_trk)
    out_subdir = Path(out_subdir)
    out_subdir.mkdir(parents=True, exist_ok=True)
    trk_file = nib.streamlines.load(str(in_trk))
    tractogram = trk_file.tractogram
    affine = tractogram.affine_to_rasmm
    if affine is None:
        affine = np.eye(4)
    cmap = plt.get_cmap(cmap_name)
    bin_means = _bin_node_means(node_values, n_bins=n_bins)
    rows = []
    color_rows_255 = []
    color_rows_01 = []
    for b in range(n_bins):
        start_fraction = b / n_bins
        end_fraction = (b + 1) / n_bins
        segment_streamlines = []
        for sl in tractogram.streamlines:
            arr_native = np.asarray(sl, dtype=np.float32)
            if arr_native.ndim != 2 or arr_native.shape[1] != 3 or arr_native.shape[0] < 2:
                continue
            arr_rasmm = nib.affines.apply_affine(affine, arr_native)
            seg = _streamline_segment_by_fraction(arr_native, arr_rasmm, start_fraction, end_fraction)
            if seg is not None:
                segment_streamlines.append(seg)
        if not segment_streamlines:
            print(f"WARNING: no segment streamlines for {pipeline_name} {tract_label} bin {b + 1:02d}; skipping")
            continue
        if flip_ap_for_dsi:
            segment_streamlines = _flip_streamlines_ap_for_dsi_display(segment_streamlines, affine, trk_file.header)
        value = bin_means[b]
        if np.isfinite(value):
            rgba = cmap(norm([value]))[0]
        else:
            rgba = np.array([0.5, 0.5, 0.5, 1.0])
        rgb_01 = rgba[:3]
        rgb_255 = np.rint(np.clip(rgb_01, 0, 1) * 255).astype(int)
        out_trk = out_subdir / f"{tract_label}_{pipeline_name}_bin{b + 1:02d}.trk"
        out_tg = nib.streamlines.Tractogram(segment_streamlines, affine_to_rasmm=affine)
        header = trk_file.header.copy()
        header["nb_streamlines"] = len(segment_streamlines)
        nib.streamlines.trk.TrkFile(out_tg, header=header).save(str(out_trk))
        color_rows_255.append(rgb_255)
        color_rows_01.append(rgb_01)
        rows.append({
            "pipeline": pipeline_name,
            "tract_label": tract_label,
            "bin": b + 1,
            "bin_label": BIN_LABELS[b] if b < len(BIN_LABELS) else f"bin{b+1}",
            "node_start_fraction": start_fraction,
            "node_end_fraction": end_fraction,
            "mean_logJ": value,
            "color_R_0_255": int(rgb_255[0]),
            "color_G_0_255": int(rgb_255[1]),
            "color_B_0_255": int(rgb_255[2]),
            "color_R_0_1": float(rgb_01[0]),
            "color_G_0_1": float(rgb_01[1]),
            "color_B_0_1": float(rgb_01[2]),
            "output_trk": str(out_trk),
            "n_segment_streamlines": int(len(segment_streamlines)),
        })
    if rows:
        color_255_txt = out_subdir / f"{tract_label}_{pipeline_name}_dsi_colors_0_255.txt"
        color_01_txt = out_subdir / f"{tract_label}_{pipeline_name}_colors_0_1.txt"
        np.savetxt(color_255_txt, np.asarray(color_rows_255, dtype=int), fmt="%d")
        np.savetxt(color_01_txt, np.asarray(color_rows_01, dtype=float), fmt="%.6f")
        for row in rows:
            row["dsi_color_txt_0_255"] = str(color_255_txt)
            row["color_txt_0_1"] = str(color_01_txt)
    return rows

def export_pipeline_dsi_binned_tractograms(df, trk_df, pipeline_name, out_dir, cmap_name="coolwarm", vlim=None, n_bins=N_BINS, flip_ap_for_dsi=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm, vmax = get_symmetric_norm(df["logJ"].values, vlim=vlim)
    trk_lookup = {}
    if trk_df is not None and not trk_df.empty:
        for row in trk_df.to_dict("records"):
            trk_lookup[(row["subject"], row["tract_label"])] = row["trk_path"]
    rows = []
    for keys, g in df.groupby(["subject", "tract_label", "family", "hemi"], dropna=False):
        subject, tract_label, family, hemi = keys
        trk_path = trk_lookup.get((subject, tract_label))
        if trk_path is None:
            print(f"WARNING: missing .trk for {pipeline_name} {subject} {tract_label}; skipping DSI binned .trk export")
            continue
        g = g.sort_values("node")
        node_values = g["logJ"].to_numpy(dtype=float)
        out_subdir = out_dir / pipeline_name / subject / f"{family}_tracts" / tract_label
        try:
            segment_rows = export_dsi_binned_segment_tractograms(
                trk_path,
                out_subdir,
                tract_label=tract_label,
                pipeline_name=pipeline_name,
                node_values=node_values,
                cmap_name=cmap_name,
                norm=norm,
                n_bins=n_bins,
                flip_ap_for_dsi=flip_ap_for_dsi,
            )
            for row in segment_rows:
                row.update({
                    "subject": subject,
                    "family": family,
                    "hemi": hemi,
                    "source_trk": str(trk_path),
                    "source_csv": str(g["source_csv"].iloc[0]),
                    "cmap": cmap_name,
                    "vmin": float(norm.vmin),
                    "vmax": float(norm.vmax),
                    "ap_flipped_for_dsi_display": bool(flip_ap_for_dsi),
                })
            rows.extend(segment_rows)
        except Exception as exc:
            print(f"WARNING: failed DSI binned .trk export for {pipeline_name} {subject} {tract_label}: {exc}")
    if rows:
        manifest = pd.DataFrame(rows)
    else:
        manifest = pd.DataFrame(columns=["pipeline", "subject", "tract_label", "bin", "output_trk"])
    return manifest

def bin_subject_tract_nodes(df, n_bins=N_BINS):
    rows = []
    for keys, g in df.groupby(["pipeline", "subject", "tract_label", "family", "hemi"], dropna=False):
        pipeline, subject, tract_label, family, hemi = keys
        g = g.sort_values("node")
        vals = g["logJ"].to_numpy(dtype=float)
        nodes = g["node"].to_numpy(dtype=int)
        idx_bins = np.array_split(np.arange(len(vals)), n_bins)
        for b, idx in enumerate(idx_bins):
            if len(idx) == 0:
                continue
            rows.append({
                "pipeline": pipeline,
                "subject": subject,
                "tract_label": tract_label,
                "family": family,
                "hemi": hemi,
                "bin": b + 1,
                "bin_label": BIN_LABELS[b] if b < len(BIN_LABELS) else f"bin{b+1}",
                "node_start": int(nodes[idx[0]]),
                "node_end": int(nodes[idx[-1]]),
                "mean_logJ": float(np.nanmean(vals[idx])),
            })
    return pd.DataFrame(rows)

def bin_nodewise_values(df, n_bins=N_BINS):
    rows = []
    for keys, g in df.groupby(["pipeline", "subject", "tract_label", "family", "hemi"], dropna=False):
        pipeline, subject, tract_label, family, hemi = keys
        g = g.sort_values("node").reset_index(drop=True)
        vals = g["logJ"].to_numpy(dtype=float)
        nodes = g["node"].to_numpy(dtype=int)
        idx_bins = np.array_split(np.arange(len(vals)), n_bins)
        for b, idx in enumerate(idx_bins):
            if len(idx) == 0:
                continue
            for node_within_bin, row_idx in enumerate(idx):
                rows.append({
                    "pipeline": pipeline,
                    "subject": subject,
                    "tract_label": tract_label,
                    "family": family,
                    "hemi": hemi,
                    "bin": b + 1,
                    "bin_label": BIN_LABELS[b] if b < len(BIN_LABELS) else f"bin{b+1}",
                    "node": int(nodes[row_idx]),
                    "node_within_bin": int(node_within_bin),
                    "logJ": float(vals[row_idx]),
                })
    return pd.DataFrame(rows)

def make_tract_column_order(labels, family_order=None, hemi_order=None):
    hemi_order = hemi_order or ["L", "R", "all"]
    parsed = []
    for label in labels:
        family, hemi, _ = parse_tract_key(label)
        parsed.append((label, family, hemi))
    families = ordered_unique([x[1] for x in parsed], family_order)
    out = []
    for fam in families:
        fam_labels = [x for x in parsed if x[1] == fam]
        for hemi in hemi_order:
            for label, _, h in sorted(fam_labels):
                if h == hemi and label not in out:
                    out.append(label)
        for label, _, h in sorted(fam_labels):
            if label not in out:
                out.append(label)
    return out

def get_group_binned_heatmap_values(binned):
    if binned.empty:
        return np.asarray([], dtype=float)
    group = (
        binned
        .groupby(["pipeline", "tract_label", "family", "hemi", "bin", "bin_label"], dropna=False)["mean_logJ"]
        .mean()
        .reset_index()
    )
    return group["mean_logJ"].to_numpy(dtype=float)

def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(pvals.shape, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not np.any(finite):
        return qvals
    p = pvals[finite]
    order = np.argsort(p)
    ranked = p[order]
    m = float(len(ranked))
    q_ranked = ranked * m / (np.arange(len(ranked)) + 1.0)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    qvals[finite] = q
    return qvals

def p_to_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def _draw_nonsignificant_cell_cross(ax, row_index, col_index, startoff=0.25, endoff=0.75):
    left = col_index - 0.5
    top = row_index - 0.5
    x_min = left + startoff
    x_max = left + endoff
    y_min = top + startoff
    y_max = top + endoff
    ax.plot([x_min, x_max], [y_min, y_max], color="black", linewidth=1.0, alpha=0.5)
    ax.plot([x_min, x_max], [y_max, y_min], color="black", linewidth=1.0, alpha=0.5)

def _cell_label_from_raw_and_fdr(inference_row, test="ttest"):
    if inference_row is None or len(inference_row) == 0:
        return "", True
    fdr_col = "wilcoxon_q_fdr" if test == "wilcoxon" else "ttest_q_fdr"
    fdr_q = float(inference_row[fdr_col].iloc[0]) if fdr_col in inference_row.columns else np.nan
    fdr_stars = p_to_stars(fdr_q)
    if fdr_stars:
        return fdr_stars, False
    return "", True

def _compute_heatmap_inference_one_pipeline(binned_one_pipeline, pipeline_name, test="ttest"):
    rows = []
    group_cols = ["tract_label", "family", "hemi", "bin", "bin_label"]
    for keys, g in binned_one_pipeline.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(g["mean_logJ"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        median = float(np.median(vals)) if n else np.nan
        t_stat = np.nan
        p_ttest = np.nan
        w_stat = np.nan
        p_wilcoxon = np.nan
        if n >= 2:
            try:
                res = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
                t_stat = float(res.statistic)
                p_ttest = float(res.pvalue)
            except Exception:
                pass
        if n >= 3 and not np.allclose(vals, 0.0):
            try:
                res = stats.wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
                w_stat = float(res.statistic)
                p_wilcoxon = float(res.pvalue)
            except Exception:
                pass
        row = {"pipeline": pipeline_name}
        row.update(dict(zip(group_cols, keys)))
        row.update({
            "n_subjects": n,
            "mean_logJ": mean,
            "median_logJ": median,
            "std_logJ": sd,
            "ttest_stat": t_stat,
            "ttest_p": p_ttest,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_wilcoxon,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["ttest_q_fdr"] = benjamini_hochberg(out["ttest_p"].to_numpy(dtype=float))
    out["wilcoxon_q_fdr"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy(dtype=float))
    out["fdr_scope"] = str(pipeline_name)
    out["n_tests_in_ttest_fdr_scope"] = int(np.sum(np.isfinite(out["ttest_p"].to_numpy(dtype=float))))
    out["n_tests_in_wilcoxon_fdr_scope"] = int(np.sum(np.isfinite(out["wilcoxon_p"].to_numpy(dtype=float))))
    return out

def compute_heatmap_inference(binned, test="ttest"):
    frames = []
    for pipeline_name, sub in binned.groupby("pipeline", dropna=False):
        frames.append(_compute_heatmap_inference_one_pipeline(sub.copy(), pipeline_name, test=test))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def summarize_heatmap_significance_counts(inference):
    if inference is None or inference.empty:
        return pd.DataFrame()
    rows = []
    for pipeline, g in inference.groupby("pipeline", dropna=False):
        row = {"pipeline": pipeline, "n_cells": int(len(g))}
        for col in ["ttest_p", "ttest_q_fdr", "wilcoxon_p", "wilcoxon_q_fdr"]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                row[f"{col}_n_finite"] = int(vals.size)
                row[f"{col}_n_lt_0_05"] = int(np.sum(vals < 0.05))
                row[f"{col}_min"] = float(np.min(vals)) if vals.size else np.nan
                row[f"{col}_median"] = float(np.median(vals)) if vals.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def _heatmap_p_column(test, use_fdr=True):
    if test == "wilcoxon":
        return "wilcoxon_q_fdr" if use_fdr else "wilcoxon_p"
    return "ttest_q_fdr" if use_fdr else "ttest_p"

def plot_group_heatmap(binned, pipeline_name, out_file, cmap_name="coolwarm", vlim=None, column_order=None, family_order=None, hemi_order=None, title=None, inference=None, heatmap_test="ttest", use_fdr=True):
    sub = binned[binned["pipeline"] == pipeline_name].copy()
    if sub.empty:
        raise RuntimeError(f"No binned data for pipeline {pipeline_name}")
    group = sub.groupby(["tract_label", "family", "hemi", "bin", "bin_label"], dropna=False)["mean_logJ"].mean().reset_index()
    inf = pd.DataFrame()
    if inference is not None and not inference.empty:
        inf = inference[inference["pipeline"] == pipeline_name].copy()
    cols = column_order or make_tract_column_order(group["tract_label"].unique(), family_order=family_order, hemi_order=hemi_order)
    rows = list(range(1, len(BIN_LABELS) + 1))
    mat = np.full((len(rows), len(cols)), np.nan, dtype=float)
    cell_text = [["" for _ in cols] for _ in rows]
    cell_draw_cross = [[False for _ in cols] for _ in rows]
    for i, b in enumerate(rows):
        for j, label in enumerate(cols):
            val = group[(group["tract_label"] == label) & (group["bin"] == b)]["mean_logJ"]
            if len(val):
                mat[i, j] = float(val.iloc[0])
                star = ""
                draw_cross = False
                if not inf.empty:
                    row = inf[(inf["tract_label"] == label) & (inf["bin"] == b)]
                    star, draw_cross = _cell_label_from_raw_and_fdr(row, test=heatmap_test)
                if draw_cross:
                    cell_draw_cross[i][j] = True
                    cell_text[i][j] = ""
                else:
                    cell_text[i][j] = f"{mat[i, j]:.3f}{star}"
    norm, _ = get_symmetric_norm(mat.ravel(), vlim=vlim)
    fig_w = max(8, 0.55 * len(cols) + 2.5)
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap=cmap_name, norm=norm, aspect="auto", interpolation="nearest")
    remove_axes_box(ax)
    ax.set_title(title or f"Group mean log det $J$ by tract decile: {pipeline_name} warps", fontsize=12, pad=28)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(BIN_LABELS, fontsize=8)
    ax.set_ylabel("tract decile", fontsize=9)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([])
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)

    for j, label in enumerate(cols):
        hemi = parse_tract_key(label)[1]
        ax.text(j, -0.62, hemi, ha="center", va="center", fontsize=8, clip_on=False)

    start = 0
    while start < len(cols):
        fam = parse_tract_key(cols[start])[0]
        end = start + 1
        while end < len(cols) and parse_tract_key(cols[end])[0] == fam:
            end += 1
        center = 0.5 * (start + end - 1)
        ax.text(center, -1.25, fam, ha="center", va="center", fontsize=8, fontweight="bold", clip_on=False)
        if end - 1 > start:
            ax.plot([start - 0.45, end - 1 + 0.45], [-0.92, -0.92], color="0.60", linewidth=0.5, clip_on=False)
        start = end

    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                if cell_draw_cross[i][j]:
                    _draw_nonsignificant_cell_cross(ax, i, j)
                elif cell_text[i][j]:
                    ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=6, color="black", fontweight="normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("mean binned log det $J$", fontsize=8, color="0.25")
    minimal_colorbar(cbar)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file

def _trk_bin_point_clouds(in_trk, n_bins=N_BINS, max_points_per_bin=25000):
    trk_file = nib.streamlines.load(str(in_trk))
    tractogram = trk_file.tractogram
    affine = tractogram.affine_to_rasmm
    if affine is None:
        affine = np.eye(4)
    bins = [[] for _ in range(n_bins)]
    for sl in tractogram.streamlines:
        arr_native = np.asarray(sl, dtype=np.float32)
        if arr_native.ndim != 2 or arr_native.shape[1] != 3 or arr_native.shape[0] < 2:
            continue
        arr_rasmm = nib.affines.apply_affine(affine, arr_native)
        for b in range(n_bins):
            seg_native = _streamline_segment_by_fraction(
                arr_native,
                arr_rasmm,
                b / n_bins,
                (b + 1) / n_bins,
            )
            if seg_native is None:
                continue
            seg_rasmm = nib.affines.apply_affine(affine, seg_native)
            bins[b].append(seg_rasmm)
    out = []
    rng = np.random.default_rng(12345)
    for chunks in bins:
        if not chunks:
            out.append(np.empty((0, 3), dtype=np.float32))
            continue
        pts = np.vstack(chunks).astype(np.float32)
        if pts.shape[0] > max_points_per_bin:
            idx = rng.choice(pts.shape[0], size=max_points_per_bin, replace=False)
            pts = pts[np.sort(idx)]
        out.append(pts)
    return out

def _symmetric_nn_distance(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    tree_b = cKDTree(B)
    d_ab, _ = tree_b.query(A, k=1)
    tree_a = cKDTree(A)
    d_ba, _ = tree_a.query(B, k=1)
    return float(0.5 * (np.nanmean(d_ab) + np.nanmean(d_ba)))

def _centroid_distance(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    return float(np.linalg.norm(np.nanmean(A, axis=0) - np.nanmean(B, axis=0)))

def compute_method_disagreement_by_bin(t2w_trks, tract_trks, n_bins=N_BINS, max_points_per_bin=25000):
    if t2w_trks.empty or tract_trks.empty:
        return pd.DataFrame()
    left = t2w_trks.rename(columns={"trk_path": "t2w_trk_path"})
    right = tract_trks.rename(columns={"trk_path": "tract_trk_path"})
    merged = left.merge(
        right[["subject", "tract_label", "tract_trk_path"]],
        on=["subject", "tract_label"],
        how="inner",
    )
    rows = []
    cache = {}
    for row in merged.to_dict("records"):
        key_t2w = row["t2w_trk_path"]
        key_tract = row["tract_trk_path"]
        try:
            if key_t2w not in cache:
                cache[key_t2w] = _trk_bin_point_clouds(key_t2w, n_bins=n_bins, max_points_per_bin=max_points_per_bin)
            if key_tract not in cache:
                cache[key_tract] = _trk_bin_point_clouds(key_tract, n_bins=n_bins, max_points_per_bin=max_points_per_bin)
            t2w_bins = cache[key_t2w]
            tract_bins = cache[key_tract]
        except Exception as exc:
            print(f"WARNING: failed loading tract pair for method disagreement: {key_t2w} / {key_tract}: {exc}")
            continue
        for b in range(n_bins):
            A = t2w_bins[b]
            B = tract_bins[b]
            rows.append({
                "subject": row["subject"],
                "tract_label": row["tract_label"],
                "family": row["family"],
                "hemi": row["hemi"],
                "bin": b + 1,
                "bin_label": BIN_LABELS[b] if b < len(BIN_LABELS) else f"bin{b+1}",
                "t2w_trk_path": key_t2w,
                "tract_trk_path": key_tract,
                "n_points_t2w": int(A.shape[0]),
                "n_points_tractography": int(B.shape[0]),
                "centroid_distance_mm": _centroid_distance(A, B),
                "symmetric_nn_distance_mm": _symmetric_nn_distance(A, B),
            })
    return pd.DataFrame(rows)

def compute_method_disagreement_inference(disagreement, metric="symmetric_nn_distance_mm", test="ttest"):
    rows = []
    if disagreement.empty or metric not in disagreement.columns:
        return pd.DataFrame()
    group_cols = ["tract_label", "family", "hemi", "bin", "bin_label"]
    for keys, g in disagreement.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        mean = float(np.mean(vals)) if n else np.nan
        median = float(np.median(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        t_stat = np.nan
        p_ttest = np.nan
        w_stat = np.nan
        p_wilcoxon = np.nan
        if n >= 2:
            try:
                res = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
                t_stat = float(res.statistic)
                p_ttest = float(res.pvalue)
            except Exception:
                pass
        if n >= 3 and not np.allclose(vals, 0.0):
            try:
                res = stats.wilcoxon(vals, zero_method="wilcox", alternative="greater")
                w_stat = float(res.statistic)
                p_wilcoxon = float(res.pvalue)
            except Exception:
                pass
        row = dict(zip(group_cols, keys))
        row.update({
            "metric": metric,
            "n_subjects": n,
            "mean": mean,
            "median": median,
            "std": sd,
            "ttest_stat": t_stat,
            "ttest_p": p_ttest,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_wilcoxon,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_q_fdr"] = benjamini_hochberg(out["ttest_p"].to_numpy(dtype=float))
    out["wilcoxon_q_fdr"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy(dtype=float))
    return out

def compute_endpoint_core_disagreement_tests(disagreement, metric="symmetric_nn_distance_mm", test="ttest", endpoint_bins=(1, 2, 9, 10), core_bins=(5, 6)):
    rows = []
    if disagreement.empty or metric not in disagreement.columns:
        return pd.DataFrame()
    for keys, g in disagreement.groupby(["tract_label", "family", "hemi"], dropna=False):
        tract_label, family, hemi = keys
        subj_rows = []
        for subject, sg in g.groupby("subject"):
            endpoint_vals = pd.to_numeric(sg.loc[sg["bin"].isin(endpoint_bins), metric], errors="coerce").to_numpy(dtype=float)
            core_vals = pd.to_numeric(sg.loc[sg["bin"].isin(core_bins), metric], errors="coerce").to_numpy(dtype=float)
            endpoint_vals = endpoint_vals[np.isfinite(endpoint_vals)]
            core_vals = core_vals[np.isfinite(core_vals)]
            if endpoint_vals.size == 0 or core_vals.size == 0:
                continue
            subj_rows.append({
                "subject": subject,
                "endpoint_mean": float(np.mean(endpoint_vals)),
                "core_mean": float(np.mean(core_vals)),
                "endpoint_minus_core": float(np.mean(endpoint_vals) - np.mean(core_vals)),
            })
        comp = pd.DataFrame(subj_rows)
        vals = comp["endpoint_minus_core"].to_numpy(dtype=float) if not comp.empty else np.asarray([], dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        t_stat = np.nan
        p_ttest = np.nan
        w_stat = np.nan
        p_wilcoxon = np.nan
        if n >= 2:
            try:
                res = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
                t_stat = float(res.statistic)
                p_ttest = float(res.pvalue)
            except Exception:
                pass
        if n >= 3 and not np.allclose(vals, 0.0):
            try:
                res = stats.wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
                w_stat = float(res.statistic)
                p_wilcoxon = float(res.pvalue)
            except Exception:
                pass
        rows.append({
            "tract_label": tract_label,
            "family": family,
            "hemi": hemi,
            "metric": metric,
            "endpoint_bins": ",".join(str(x) for x in endpoint_bins),
            "core_bins": ",".join(str(x) for x in core_bins),
            "n_subjects": n,
            "mean_endpoint_minus_core": float(np.mean(vals)) if n else np.nan,
            "median_endpoint_minus_core": float(np.median(vals)) if n else np.nan,
            "std_endpoint_minus_core": float(np.std(vals, ddof=1)) if n > 1 else np.nan,
            "ttest_stat": t_stat,
            "ttest_p": p_ttest,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_wilcoxon,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_q_fdr"] = benjamini_hochberg(out["ttest_p"].to_numpy(dtype=float))
    out["wilcoxon_q_fdr"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy(dtype=float))
    return out

def _method_test_p_column(test, use_fdr=True):
    if test == "wilcoxon":
        return "wilcoxon_q_fdr" if use_fdr else "wilcoxon_p"
    return "ttest_q_fdr" if use_fdr else "ttest_p"

def plot_method_disagreement_heatmap(disagreement, out_file, metric="symmetric_nn_distance_mm", column_order=None, family_order=None, hemi_order=None, cmap_name="viridis", inference=None, test="ttest", use_fdr=True, title=None):
    group = disagreement.groupby(["tract_label", "family", "hemi", "bin", "bin_label"], dropna=False)[metric].mean().reset_index()
    cols = column_order or make_tract_column_order(group["tract_label"].unique(), family_order=family_order, hemi_order=hemi_order)
    rows = list(range(1, len(BIN_LABELS) + 1))
    mat = np.full((len(rows), len(cols)), np.nan, dtype=float)
    txt = [["" for _ in cols] for _ in rows]
    txt_draw_cross = [[False for _ in cols] for _ in rows]
    for i, b in enumerate(rows):
        for j, label in enumerate(cols):
            vals = group[(group["tract_label"] == label) & (group["bin"] == b)][metric]
            if len(vals):
                mat[i, j] = float(vals.iloc[0])
                star = ""
                draw_cross = False
                if inference is not None and not inference.empty:
                    row = inference[(inference["tract_label"] == label) & (inference["bin"] == b)]
                    star, draw_cross = _cell_label_from_raw_and_fdr(row, test=test)
                if draw_cross:
                    txt_draw_cross[i][j] = True
                    txt[i][j] = ""
                else:
                    txt[i][j] = f"{mat[i, j]:.2f}{star}"
    finite = mat[np.isfinite(mat)]
    vmax = float(np.nanpercentile(finite, 95)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig_w = max(8, 0.55 * len(cols) + 2.5)
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, cmap=cmap_name, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    remove_axes_box(ax)
    ax.set_title(title or "Method disagreement by tract decile: Anatomical vs Tractography warps", fontsize=12, pad=42)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(BIN_LABELS, fontsize=8)
    ax.set_ylabel("tract decile", fontsize=9)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([])
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    for j, label in enumerate(cols):
        hemi = parse_tract_key(label)[1]
        ax.text(j, -0.62, hemi, ha="center", va="center", fontsize=8, clip_on=False)
    start = 0
    while start < len(cols):
        fam = parse_tract_key(cols[start])[0]
        end = start + 1
        while end < len(cols) and parse_tract_key(cols[end])[0] == fam:
            end += 1
        center = 0.5 * (start + end - 1)
        ax.text(center, -1.25, fam, ha="center", va="center", fontsize=8, fontweight="bold", clip_on=False)
        if end - 1 > start:
            ax.plot([start - 0.45, end - 1 + 0.45], [-0.92, -0.92], color="0.60", linewidth=0.5, clip_on=False)
        start = end
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                if txt_draw_cross[i][j]:
                    _draw_nonsignificant_cell_cross(ax, i, j)
                elif txt[i][j]:
                    ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=6, color="black", fontweight="normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("mean symmetric NN distance (mm)", fontsize=8, color="0.25")
    minimal_colorbar(cbar)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file

def plot_method_disagreement_profile_grid(disagreement, out_file, metric="symmetric_nn_distance_mm", family_order=None, hemi_order=None, endpoint_core_tests=None, test="ttest", use_fdr=True):
    families = ordered_unique(disagreement["family"].unique(), family_order)
    hemis = ordered_unique(disagreement["hemi"].unique(), hemi_order or ["L", "R", "all"])
    n_rows = max(1, len(families))
    n_cols = max(1, len(hemis))
    fig_w = max(6, 3.2 * n_cols)
    fig_h = max(4, 2.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    x = np.arange(1, N_BINS + 1)
    p_col = _method_test_p_column(test, use_fdr=use_fdr)
    for r, family in enumerate(families):
        for c, hemi in enumerate(hemis):
            ax = axes[r, c]
            remove_axes_box(ax)
            sub = disagreement[(disagreement["family"] == family) & (disagreement["hemi"] == hemi)]
            if sub.empty:
                ax.axis("off")
                continue
            grp = sub.groupby("bin")[metric]
            means = grp.mean().reindex(x).to_numpy(dtype=float)
            medians = grp.median().reindex(x).to_numpy(dtype=float)
            counts = grp.count().reindex(x).to_numpy(dtype=float)
            stds = grp.std().reindex(x).to_numpy(dtype=float)
            sem = stds / np.sqrt(np.maximum(counts, 1))
            ax.plot(x, means, marker="o", linewidth=1.25, markersize=3, color="0.18", label="mean")
            ax.plot(x, medians, marker="s", linewidth=1.0, markersize=2.5, linestyle="--", color="0.48", label="median")
            ax.fill_between(x, means - sem, means + sem, color="0.70", alpha=0.28, linewidth=0)
            ax.axvspan(0.5, 2.5, color="0.94", zorder=0)
            ax.axvspan(8.5, 10.5, color="0.94", zorder=0)
            ax.axvspan(4.5, 6.5, color="0.985", zorder=0)
            ax.grid(True, axis="y", color="0.92", linewidth=0.5)
            ax.set_axisbelow(True)
            star = ""
            diff_txt = ""
            if endpoint_core_tests is not None and not endpoint_core_tests.empty and p_col in endpoint_core_tests.columns:
                row = endpoint_core_tests[(endpoint_core_tests["family"] == family) & (endpoint_core_tests["hemi"] == hemi)]
                if len(row):
                    pval = float(row[p_col].iloc[0])
                    star = p_to_stars(pval)
                    diff = row["mean_endpoint_minus_core"].iloc[0]
                    if np.isfinite(diff):
                        diff_txt = f" endpoint-core={diff:.2f} mm{star}"
            ax.set_title(f"{family}_{hemi}{diff_txt}", fontsize=8, pad=6)
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in x], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            if r == n_rows - 1:
                ax.set_xlabel("decile bin", fontsize=8)
            if c == 0:
                ax.set_ylabel("symmetric NN distance (mm)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=8, ncol=len(handles), bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Method-disagreement profiles: Anatomical vs Tractography warps", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file

def _normalize_threshold_label(value):
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1e-8:
            return f"{int(round(f))}mm"
        return f"{f:g}mm"
    except Exception:
        return str(value)

def _threshold_numeric(threshold_label):
    m = re.search(r"([-+]?\d*\.?\d+)", str(threshold_label))
    if not m:
        return np.nan
    return float(m.group(1))

def extract_buan_columns(df, allowed_prefix=None):
    out = []
    for col in df.columns:
        low = col.lower()
        if "buan" not in low or "threshold" not in low or "improvement" in low:
            continue
        if "aligned" in low or "deformed" in low:
            continue
        if allowed_prefix is not None and allowed_prefix.lower() not in low:
            continue
        m = re.search(r"threshold_([-+]?\d*\.?\d+)mm", low)
        if not m:
            continue
        out.append((col, _normalize_threshold_label(m.group(1)), _threshold_numeric(m.group(1))))
    out.sort(key=lambda x: (np.inf if not np.isfinite(x[2]) else x[2], x[0]))
    return out

def load_buan_summaries(root, pipeline, glob_pattern, subjects=None, allowed_prefix=None):
    rows = []
    for subject_dir in find_subject_dirs(root, subjects=subjects):
        subject = subject_dir.name
        files = sorted(subject_dir.glob(glob_pattern))
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception as exc:
                print(f"WARNING: skipping BUAN summary {f}: {exc}")
                continue
            buan_cols = extract_buan_columns(df, allowed_prefix=allowed_prefix)
            if not buan_cols:
                print(f"WARNING: no BUAN threshold columns found in {f}")
                continue
            for _, row in df.iterrows():
                tract_key = row.get("tract", row.get("tract_key", row.get("younger_tract", "")))
                if pd.isna(tract_key) or str(tract_key).strip() == "":
                    continue
                family, hemi, tract_label = parse_tract_key(str(tract_key))
                for col, threshold_label, threshold_mm in buan_cols:
                    value = pd.to_numeric(row.get(col, np.nan), errors="coerce")
                    if not np.isfinite(value):
                        continue
                    rows.append({
                        "pipeline": pipeline,
                        "subject": subject,
                        "tract_key": str(tract_key),
                        "family": family,
                        "hemi": hemi,
                        "tract_label": tract_label,
                        "threshold": threshold_label,
                        "threshold_mm": threshold_mm,
                        "buan_similarity": float(value),
                        "source_column": col,
                        "source_csv": str(f),
                    })
    if not rows:
        return pd.DataFrame(columns=["pipeline", "subject", "tract_key", "family", "hemi", "tract_label", "threshold", "threshold_mm", "buan_similarity", "source_column", "source_csv"])
    return pd.DataFrame(rows)

def summarize_buan_population(buan_df):
    if buan_df.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in buan_df.groupby(["pipeline", "family", "hemi", "tract_label", "threshold", "threshold_mm"], dropna=False):
        pipeline, family, hemi, tract_label, threshold, threshold_mm = keys
        vals = pd.to_numeric(g["buan_similarity"], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append({
            "pipeline": pipeline,
            "family": family,
            "hemi": hemi,
            "tract_label": tract_label,
            "threshold": threshold,
            "threshold_mm": threshold_mm,
            "n_subjects": int(vals.size),
            "mean_buan_similarity": _safe_nanmean(vals),
            "median_buan_similarity": _safe_nanmedian(vals),
            "std_buan_similarity": _safe_nanstd(vals),
        })
    return pd.DataFrame(rows)

def compute_buan_pipeline_tests(buan_df, pipeline_a="Anatomical", pipeline_b="Tractography", test="ttest"):
    if buan_df.empty:
        return pd.DataFrame()
    rows = []
    grouped = buan_df.groupby(["family", "hemi", "tract_label", "threshold", "threshold_mm"], dropna=False)
    for keys, g in grouped:
        family, hemi, tract_label, threshold, threshold_mm = keys
        a = g[g["pipeline"] == pipeline_a][["subject", "buan_similarity"]].rename(columns={"buan_similarity": "a"})
        b = g[g["pipeline"] == pipeline_b][["subject", "buan_similarity"]].rename(columns={"buan_similarity": "b"})
        merged = pd.merge(a, b, on="subject", how="inner")
        avals = pd.to_numeric(merged["a"], errors="coerce").to_numpy(dtype=float)
        bvals = pd.to_numeric(merged["b"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(avals) & np.isfinite(bvals)
        avals = avals[valid]
        bvals = bvals[valid]
        diffs = bvals - avals
        if diffs.size >= 2 and np.nanstd(diffs) > 0:
            t_stat, t_p = stats.ttest_rel(bvals, avals, nan_policy="omit")
        else:
            t_stat, t_p = np.nan, np.nan
        if diffs.size >= 1 and np.any(np.abs(diffs) > 0):
            try:
                w_stat, w_p = stats.wilcoxon(diffs)
            except Exception:
                w_stat, w_p = np.nan, np.nan
        else:
            w_stat, w_p = np.nan, np.nan
        rows.append({
            "family": family,
            "hemi": hemi,
            "tract_label": tract_label,
            "threshold": threshold,
            "threshold_mm": threshold_mm,
            "n_subjects": int(diffs.size),
            f"mean_{pipeline_b}_minus_{pipeline_a}": _safe_nanmean(diffs),
            "ttest_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "ttest_p": float(t_p) if np.isfinite(t_p) else np.nan,
            "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
            "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
        })
    out = pd.DataFrame(rows)
    for col in ["ttest_p", "wilcoxon_p"]:
        if col in out.columns:
            valid = out[col].notna()
            q = np.full(len(out), np.nan, dtype=float)
            if valid.any():
                q[valid.to_numpy()] = benjamini_hochberg(out.loc[valid, col].to_numpy(dtype=float))
            out[col.replace("_p", "_q_fdr")] = q
    return out

def threshold_shade_colors(thresholds, cmap_name):
    thresholds = sorted(thresholds, key=lambda x: _threshold_numeric(x))
    cmap = plt.get_cmap(cmap_name)
    if len(thresholds) == 1:
        vals = [0.70]
    else:
        vals = np.linspace(0.45, 0.90, len(thresholds))
    return {thr: cmap(v) for thr, v in zip(thresholds, vals)}

def plot_buan_similarity_by_tract(buan_df, tests_df, out_file, pipeline_a="Anatomical", pipeline_b="Tractography", family_order=None, hemi_order=None, use_fdr=True, test="ttest"):
    if buan_df.empty:
        return None
    labels = make_tract_column_order(sorted(buan_df["tract_label"].unique()), family_order=family_order, hemi_order=hemi_order)
    thresholds = sorted(buan_df["threshold"].unique(), key=lambda x: _threshold_numeric(x))
    t2w_colors = threshold_shade_colors(thresholds, "Blues")
    tract_colors = threshold_shade_colors(thresholds, "Oranges")
    pipeline_colors = {pipeline_a: t2w_colors, pipeline_b: tract_colors}
    n_thr = max(len(thresholds), 1)
    fig_w = max(10, 0.72 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    xbase = np.arange(len(labels), dtype=float)
    offsets = {}
    width = 0.36
    for p_i, pipe in enumerate([pipeline_a, pipeline_b]):
        base_shift = -width / 2 if p_i == 0 else width / 2
        for t_i, thr in enumerate(thresholds):
            within = 0.0 if n_thr == 1 else (t_i - (n_thr - 1) / 2) * (width / max(n_thr, 1)) * 0.78
            offsets[(pipe, thr)] = base_shift + within
    rng = np.random.default_rng(12345)
    for i, label in enumerate(labels):
        sub = buan_df[buan_df["tract_label"] == label]
        for pipe in [pipeline_a, pipeline_b]:
            for thr in thresholds:
                vals = sub[(sub["pipeline"] == pipe) & (sub["threshold"] == thr)]["buan_similarity"].dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                x = xbase[i] + offsets[(pipe, thr)]
                jitter = rng.normal(0, 0.008, size=vals.size)
                ax.scatter(np.full(vals.size, x) + jitter, vals, s=11, alpha=0.35, color=pipeline_colors[pipe][thr], edgecolors="none")
                ax.plot([x - 0.035, x + 0.035], [np.nanmean(vals), np.nanmean(vals)], color=pipeline_colors[pipe][thr], linewidth=1.8)
    p_col = f"{test}_q_fdr" if use_fdr else f"{test}_p"
    if tests_df is not None and not tests_df.empty and p_col in tests_df.columns:
        y_top = np.nanmax(buan_df["buan_similarity"].to_numpy(dtype=float))
        if not np.isfinite(y_top):
            y_top = 1.0
        y_step = 0.035 * max(y_top, 1.0)
        for i, label in enumerate(labels):
            for t_i, thr in enumerate(thresholds):
                row = tests_df[(tests_df["tract_label"] == label) & (tests_df["threshold"] == thr)]
                if len(row) == 0:
                    continue
                pval = row[p_col].iloc[0]
                stars = p_to_stars(pval)
                if not stars:
                    continue
                y = y_top + (t_i + 1) * y_step
                ax.text(xbase[i], y, stars, ha="center", va="bottom", fontsize=7, color=FDR_SIGNIFICANT_TEXT_COLOR)
    ax.set_xticks(xbase)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("BUAN similarity", fontsize=9)
    ax.set_title("BUAN similarity by tract: Anatomical and Tractography pipelines", fontsize=13, pad=12)
    ax.grid(True, axis="y", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    remove_axes_box(ax)
    # family separators/top labels
    if labels:
        ylim = ax.get_ylim()
        y_label = ylim[1] + 0.04 * (ylim[1] - ylim[0])
        families = [parse_tract_key(l)[0] for l in labels]
        start = 0
        for fam in sorted(set(families), key=families.index):
            idxs = [j for j, f in enumerate(families) if f == fam]
            if not idxs:
                continue
            ax.text(np.mean(idxs), y_label, fam, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(ylim[0], y_label + 0.04 * (ylim[1] - ylim[0]))
    from matplotlib.lines import Line2D
    handles = []
    for pipe, cmap_name in [(pipeline_a, "Blues"), (pipeline_b, "Oranges")]:
        colors = pipeline_colors[pipe]
        for thr in thresholds:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=5, color=colors[thr], label=f"{pipe} {thr}"))
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=min(len(handles), 4), loc="lower center", bbox_to_anchor=(0.5, -0.34))
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file

def load_bundlewarp_summaries(root, glob_pattern=DEFAULT_BUNDLEWARP_SUMMARY_GLOB, subjects=None):
    rows = []
    for subject_dir in find_subject_dirs(root, subjects=subjects):
        subject = subject_dir.name
        files = sorted(subject_dir.glob(glob_pattern))
        if not files:
            print(f"WARNING: no BundleWarp summary CSV found for subject {subject} using pattern {glob_pattern}")
        for f in files:
            try:
                df = pd.read_csv(f)
                if "subject" not in df.columns:
                    df.insert(0, "subject", subject)
                else:
                    df["subject"] = df["subject"].fillna(subject)
                if "tract" in df.columns:
                    tract_keys = df["tract"].astype(str)
                elif "tract_label" in df.columns:
                    tract_keys = df["tract_label"].astype(str)
                else:
                    print(f"WARNING: BundleWarp summary lacks tract column, skipping {f}")
                    continue
                parsed = tract_keys.apply(parse_tract_key)
                df["family"] = [x[0] for x in parsed]
                df["hemi"] = [x[1] for x in parsed]
                df["tract_label"] = [x[2] for x in parsed]
                df["source_csv"] = str(f)
                rows.append(df)
            except Exception as exc:
                print(f"WARNING: failed reading BundleWarp summary {f}: {exc}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


BUNDLEWARP_METRIC_GROUPS = {
    "higher_is_better": [
        "mean_centroid_improvement",
        "percent_centroid_improvement",
        "mean_node_residual_improvement",
        "percent_nodes_improved",
        "fraction_younger_streamlines_matched",
        "fraction_older_streamlines_matched",
        "buan_improvement_threshold_5mm",
        "buan_improvement_threshold_7mm",
        "buan_improvement_threshold_10mm",
    ],
    "lower_is_better": [
        "mean_node_residual_post",
        "mean_centroid_post_distance",
        "mean_spread_residual_to_older",
        "match_reuse_rate",
        "mean_tangent_angle_change_deg",
        "p95_tangent_angle_change_deg",
        "p95_abs_log_tangent_stretch",
        "endpoint_displacement_start",
        "endpoint_displacement_end",
        "first10_mean_displacement",
        "last10_mean_displacement",
    ],
    "deformation_magnitude_or_direction": [
        "mean_bundlewarp_displacement",
        "core_mean_displacement",
        "mean_perpendicular_displacement",
        "mean_parallel_displacement",
        "median_abs_parallel_displacement",
        "mean_log_tangent_stretch",
        "mean_spread_change",
    ],
}

def bundlewarp_metric_group(metric):
    for group, metrics in BUNDLEWARP_METRIC_GROUPS.items():
        if metric in metrics:
            return group
    return "other"

def bundlewarp_metric_group_label(group):
    labels = {
        "higher_is_better": "Higher = better",
        "lower_is_better": "Lower = better",
        "deformation_magnitude_or_direction": "Magnitude/direction",
        "other": "Other",
    }
    return labels.get(group, group.replace("_", " "))

def default_bundlewarp_metrics():
    out = []
    for group in ["higher_is_better", "lower_is_better", "deformation_magnitude_or_direction"]:
        out.extend(BUNDLEWARP_METRIC_GROUPS[group])
    return out

def choose_bundlewarp_metrics(df, requested=None):
    if df is None or df.empty:
        return []
    metrics = requested or default_bundlewarp_metrics()
    return [m for m in metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]

def summarize_bundlewarp_population(df, metrics):
    rows = []
    if df.empty or not metrics:
        return pd.DataFrame()
    for keys, g in df.groupby(["tract_label", "family", "hemi"], dropna=False):
        tract_label, family, hemi = keys
        row = {"tract_label": tract_label, "family": family, "hemi": hemi, "n_subjects": int(g["subject"].nunique())}
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{metric}_median"] = float(np.median(vals)) if vals.size else np.nan
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
            row[f"{metric}_n"] = int(vals.size)
        rows.append(row)
    return pd.DataFrame(rows)

def metric_display_name(metric):
    replacements = {
        "mean_bundlewarp_displacement": "BW displacement",
        "core_mean_displacement": "Core displacement",
        "mean_perpendicular_displacement": "Perpendicular displacement",
        "mean_parallel_displacement": "Parallel displacement",
        "median_abs_parallel_displacement": "Abs. parallel displacement",
        "mean_log_tangent_stretch": "Log tangent stretch",
        "p95_abs_log_tangent_stretch": "P95 abs. log stretch",
        "mean_tangent_angle_change_deg": "Tangent angle change",
        "p95_tangent_angle_change_deg": "P95 tangent angle change",
        "mean_centroid_improvement": "Centroid improvement",
        "percent_centroid_improvement": "% centroid improvement",
        "mean_centroid_post_distance": "Post centroid distance",
        "mean_node_residual_improvement": "Residual improvement",
        "percent_nodes_improved": "% nodes improved",
        "mean_node_residual_post": "Post residual",
        "mean_spread_change": "Spread change",
        "mean_spread_residual_to_older": "Spread residual",
        "fraction_younger_streamlines_matched": "V02 match fraction",
        "fraction_older_streamlines_matched": "V03 match fraction",
        "match_reuse_rate": "Match reuse rate",
        "endpoint_displacement_start": "Start endpoint displacement",
        "endpoint_displacement_end": "End endpoint displacement",
        "first10_mean_displacement": "First decile displacement",
        "last10_mean_displacement": "Last decile displacement",
        "buan_improvement_threshold_5mm": "BUAN improvement 5 mm",
        "buan_improvement_threshold_7mm": "BUAN improvement 7 mm",
        "buan_improvement_threshold_10mm": "BUAN improvement 10 mm",
    }
    return replacements.get(metric, metric.replace("_", " "))

def plot_bundlewarp_metric_distributions(df, metrics, out_file, family_order=None, hemi_order=None, n_cols=4):
    if df.empty or not metrics:
        return None
    labels = make_tract_column_order(sorted(df["tract_label"].dropna().unique()), family_order=family_order, hemi_order=hemi_order)
    n_metrics = len(metrics)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig_w = max(12, 0.32 * len(labels) * n_cols + 2.0)
    fig_h = max(2.8 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    rng = np.random.default_rng(12345)
    for i, metric in enumerate(metrics):
        ax = axes[i // n_cols, i % n_cols]
        remove_axes_box(ax)
        for x, label in enumerate(labels):
            vals = pd.to_numeric(df.loc[df["tract_label"] == label, metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            jitter = rng.normal(0, 0.045, size=vals.size)
            ax.scatter(np.full(vals.size, x) + jitter, vals, s=13, alpha=0.55, linewidths=0, color="0.25")
            med = np.median(vals)
            ax.plot([x - 0.22, x + 0.22], [med, med], color="tab:blue", linewidth=1.5)
        ax.axhline(0, color="0.75", linewidth=0.6)
        ax.grid(True, axis="y", color="0.92", linewidth=0.5)
        group_label = bundlewarp_metric_group_label(bundlewarp_metric_group(metric))
        ax.set_title(f"{group_label}: {metric_display_name(metric)}", fontsize=9, pad=6)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for j in range(n_metrics, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")
    fig.suptitle("Population-level BundleWarp metrics by tract", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_file

def parse_args():
    p = argparse.ArgumentParser(description="Export DSI-compatible decile-segment .trk files and create group heatmaps, method-disagreement plots, and BundleWarp summaries.")
    p.add_argument("--t2w-output-root", required=True, help="Directory containing subject folders from the Anatomical registration pipeline.")
    p.add_argument("--tract-output-root", required=True, help="Directory containing subject folders from the Tractography pipeline.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--subjects", nargs="+", default=None, help="Optional subject IDs to include. Can be sub-* or bare IDs.")
    p.add_argument("--t2w-glob", default=DEFAULT_T2W_GLOB)
    p.add_argument("--tract-glob", default=DEFAULT_TRACT_GLOB)
    p.add_argument("--t2w-trk-glob", default=DEFAULT_T2W_TRK_GLOB, help="Glob for warped Anatomical-pipeline .trk files inside each subject folder.")
    p.add_argument("--tract-trk-glob", default=DEFAULT_TRACT_TRK_GLOB, help="Glob for warped Tractography .trk files inside each subject folder.")
    p.add_argument("--t2w-metric-col", default=None)
    p.add_argument("--tract-metric-col", default=None)
    p.add_argument("--family-order", nargs="+", default=None)
    p.add_argument("--fig-format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--heatmap-vlim", type=float, default=None, help="Symmetric color limit for Figure 2 heatmaps. Default: max absolute binned mean log$J$ across both pipelines.")
    p.add_argument("--dsi-bin-vlim", type=float, default=None, help="Symmetric color limit for DSI-compatible decile-segment .trk colors. Default: max absolute subject-level decile mean log$J$ across both pipelines.")
    p.add_argument("--method-disagreement-metric", choices=["symmetric_nn_distance_mm", "centroid_distance_mm"], default="symmetric_nn_distance_mm", help="Spatial disagreement metric for Figure 3.")
    p.add_argument("--t2w-buan-summary-glob", default=DEFAULT_T2W_BUAN_SUMMARY_GLOB, help="Glob for Anatomical tract similarity summary CSVs inside each subject folder.")
    p.add_argument("--tract-buan-summary-glob", default=DEFAULT_TRACT_BUAN_SUMMARY_GLOB, help="Glob for Tractography mask-SyN summary CSVs inside each subject folder.")
    p.add_argument("--bundlewarp-summary-glob", default=DEFAULT_BUNDLEWARP_SUMMARY_GLOB, help="Glob for BundleWarp summary CSVs inside each Tractography subject folder.")
    p.add_argument("--bundlewarp-metrics", nargs="+", default=None, help="Optional BundleWarp summary metrics to visualize. Defaults to a curated set of key metrics if present.")
    return p.parse_args()

def main():
    args = parse_args()
    set_minimal_plot_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t2w = load_pipeline_data(args.t2w_output_root, t2w_label, args.t2w_glob, metric_col=args.t2w_metric_col, subjects=args.subjects)
    tract = load_pipeline_data(args.tract_output_root, tract_label, args.tract_glob, metric_col=args.tract_metric_col, subjects=args.subjects)
    all_data = pd.concat([t2w, tract], ignore_index=True)

    t2w_trks = load_pipeline_trk_map(args.t2w_output_root, t2w_label, args.t2w_trk_glob, subjects=args.subjects)
    tract_trks = load_pipeline_trk_map(args.tract_output_root, tract_label, args.tract_trk_glob, subjects=args.subjects)

    binned = bin_subject_tract_nodes(all_data, n_bins=N_BINS)

    heatmap_inference = compute_heatmap_inference(binned, test=heatmap_test)
    heatmap_inference.to_csv(out_dir / "figure2_heatmap_logJ_one_sample_tests.csv", index=False)
    heatmap_sig_counts = summarize_heatmap_significance_counts(heatmap_inference)

    _, dsi_bin_vmax = get_symmetric_norm(binned["mean_logJ"].values, vlim=args.dsi_bin_vlim)
    print(f"Using symmetric color scale for DSI-compatible decile-segment .trk colors: ±{dsi_bin_vmax:.6g}")
    dsi_binned_trk_dir = out_dir / "dsi_compatible_decile_metric_trks"
    t2w_dsi_manifest = export_pipeline_dsi_binned_tractograms(
        t2w,
        t2w_trks,
        t2w_label,
        dsi_binned_trk_dir,
        cmap_name=heatmap_cmap,
        vlim=dsi_bin_vmax,
        n_bins=N_BINS,
        flip_ap_for_dsi=True,
    )
    tract_dsi_manifest = export_pipeline_dsi_binned_tractograms(
        tract,
        tract_trks,
        tract_label,
        dsi_binned_trk_dir,
        cmap_name=heatmap_cmap,
        vlim=dsi_bin_vmax,
        n_bins=N_BINS,
    )
    dsi_manifest = pd.concat([t2w_dsi_manifest, tract_dsi_manifest], ignore_index=True)
    dsi_color_metadata = {
        "cmap": heatmap_cmap,
        "vmin": -float(dsi_bin_vmax),
        "vcenter": 0.0,
        "vmax": float(dsi_bin_vmax),
        "n_bins": N_BINS,
        "bin_labels": BIN_LABELS,
        "scale_basis": "subject-level decile mean log$J$ values across both pipelines",
    }
    with open(out_dir / "dsi_compatible_decile_metric_trk_color_metadata.json", "w") as f:
        json.dump(dsi_color_metadata, f, indent=2)

    all_labels = sorted(binned["tract_label"].unique())
    column_order = make_tract_column_order(all_labels, family_order=args.family_order, hemi_order=hemi_order)
    heatmap_display_values = get_group_binned_heatmap_values(binned)
    _, heatmap_vmax = get_symmetric_norm(heatmap_display_values, vlim=args.heatmap_vlim)
    print(f"Using symmetric color scale for Figure 2 heatmaps only, based on displayed group mean binned values: ±{heatmap_vmax:.6g}")

    plot_group_heatmap(binned, t2w_label, out_dir / f"figure2_group_heatmap_{t2w_label}.{args.fig_format}", cmap_name=heatmap_cmap, vlim=heatmap_vmax, column_order=column_order, family_order=args.family_order, hemi_order=hemi_order, title=f"Group mean log det $J$ by tract decile: {t2w_label} warps", inference=heatmap_inference, heatmap_test=heatmap_test, use_fdr=heatmap_use_fdr)
    plot_group_heatmap(binned, tract_label, out_dir / f"figure2_group_heatmap_{tract_label}.{args.fig_format}", cmap_name=heatmap_cmap, vlim=heatmap_vmax, column_order=column_order, family_order=args.family_order, hemi_order=hemi_order, title=f"Group mean log det $J$ by tract decile: {tract_label} warps", inference=heatmap_inference, heatmap_test=heatmap_test, use_fdr=heatmap_use_fdr)
    heatmap_color_metadata = {
        "cmap": heatmap_cmap,
        "vmin": -float(heatmap_vmax),
        "vcenter": 0.0,
        "vmax": float(heatmap_vmax),
        "scale_basis": "subject-level decile-binned mean log$J$ values across both pipelines",
        "heatmap_test": heatmap_test,
        "heatmap_significance_annotation": "pipeline-specific FDR q-values",
    }
    with open(out_dir / "figure2_heatmap_color_metadata.json", "w") as f:
        json.dump(heatmap_color_metadata, f, indent=2)

    method_disagreement = compute_method_disagreement_by_bin(
        t2w_trks,
        tract_trks,
        n_bins=N_BINS,
        max_points_per_bin=method_disagreement_max_points_per_bin,
    )
    method_disagreement_tests = compute_method_disagreement_inference(
        method_disagreement,
        metric=args.method_disagreement_metric,
        test=method_disagreement_test,
    )
    method_disagreement_tests.to_csv(out_dir / "figure3_method_disagreement_cell_tests.csv", index=False)
    endpoint_core_tests = compute_endpoint_core_disagreement_tests(
        method_disagreement,
        metric=args.method_disagreement_metric,
        test=method_disagreement_test,
    )
    endpoint_core_tests.to_csv(out_dir / "figure3_method_disagreement_endpoint_vs_core_tests.csv", index=False)
    method_heatmap_file = plot_method_disagreement_heatmap(
        method_disagreement,
        out_dir / f"figure3_method_disagreement_heatmap.{args.fig_format}",
        metric=args.method_disagreement_metric,
        column_order=column_order,
        family_order=args.family_order,
        hemi_order=hemi_order,
        cmap_name=method_disagreement_cmap,
        inference=method_disagreement_tests,
        test=method_disagreement_test,
        use_fdr=method_use_fdr,
        title="Method disagreement by tract decile: Anatomical vs Tractography warps",
    )
    method_profile_file = plot_method_disagreement_profile_grid(
        method_disagreement,
        out_dir / f"figure3_method_disagreement_profiles.{args.fig_format}",
        metric=args.method_disagreement_metric,
        family_order=args.family_order,
        hemi_order=hemi_order,
        endpoint_core_tests=endpoint_core_tests,
        test=method_disagreement_test,
        use_fdr=method_use_fdr,
    )

    t2w_buan = load_buan_summaries(
        args.t2w_output_root,
        t2w_label,
        args.t2w_buan_summary_glob,
        subjects=args.subjects,
        allowed_prefix=None,
    )
    tract_buan = load_buan_summaries(
        args.tract_output_root,
        tract_label,
        args.tract_buan_summary_glob,
        subjects=args.subjects,
        allowed_prefix="buan_mask_syn",
    )
    buan_long = pd.concat([t2w_buan, tract_buan], ignore_index=True)
    if not buan_long.empty:
        buan_summary = summarize_buan_population(buan_long)
        buan_tests = compute_buan_pipeline_tests(
            buan_long,
            pipeline_a=t2w_label,
            pipeline_b=tract_label,
            test=buan_test,
        )
        buan_tests.to_csv(out_dir / "figure6_buan_similarity_pipeline_tests.csv", index=False)
        plot_buan_similarity_by_tract(
            buan_long,
            buan_tests,
            out_dir / f"figure6_buan_similarity_by_tract.{args.fig_format}",
            pipeline_a=t2w_label,
            pipeline_b=tract_label,
            family_order=args.family_order,
            hemi_order=hemi_order,
            use_fdr=buan_use_fdr,
            test=buan_test,
        )
    else:
        print("WARNING: no BUAN similarity summaries loaded; skipping BUAN similarity figure.")

    bw = load_bundlewarp_summaries(args.tract_output_root, glob_pattern=args.bundlewarp_summary_glob, subjects=args.subjects)
    if not bw.empty:
        bw_metrics = choose_bundlewarp_metrics(bw, requested=args.bundlewarp_metrics)
        if bw_metrics:
            bw_summary = summarize_bundlewarp_population(bw, bw_metrics)
            bw_summary.to_csv(out_dir / "bundlewarp_population_metric_summary.csv", index=False)
            plot_bundlewarp_metric_distributions(
                bw,
                bw_metrics,
                out_dir / f"figure4_bundlewarp_metric_distributions.{args.fig_format}",
                family_order=args.family_order,
                hemi_order=hemi_order,
                n_cols=bundlewarp_distribution_ncols,
            )
        else:
            print("WARNING: no requested/default BundleWarp metrics found for population plots.")
    else:
        print("WARNING: no BundleWarp summaries loaded; skipping BundleWarp population plots.")

    print("Done.")
    print(f"Wrote outputs to: {out_dir}")

if __name__ == "__main__":
    main()
