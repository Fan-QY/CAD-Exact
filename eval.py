"""
GeomCAD-Eval v2.0 — Industrial-Precision-First Evaluation Protocol
==================================================================

Three tiers of metrics:

  Tier 1 — Industrial Precision (paper headline)
    Mesh-level (canonical space, bbox diagonal = 1000):
      - Hausdorff Distance (max worst-case point deviation)
      - Tolerance Pass Rate @ τ ∈ {1, 5, 10, 20}
      - Volume Error (% of GT volume)
    Program-level (parameter vocabulary [1, 1000]):
      - Parameter Exact Match (PEM)
      - Parameter Tolerant Match @ ±k, k ∈ {1, 5, 10}
      - Param Count Match Rate

  Tier 2 — Similarity (cross-paper compatibility)
      - IoU (volumetric, point-in-volume estimate)
      - CD ×10³ (mean of squared symmetric distances, in canonical space)
      - F-Score @ τ_relative = 2%   [Tatarchenko et al., CVPR 2019]

  Tier 3 — Generation
      - Validity Rate (% of pred samples whose .py executes + produces non-empty mesh)

Coordinate system:
  We use the GT bounding box to define a canonical space where bbox diagonal
  = 1000 canonical units. The pred mesh is transformed using the SAME center+
  scale derived from GT — this preserves any scale-mismatch signal (so a model
  whose pred is twice the size of GT will have huge Hausdorff, not zero).

Aggregation:
  Median (primary) and mean (secondary). All Tier 1/2 metrics are conditional
  on per-method validity; n_valid is reported.

Usage:
  python geomcad_eval.py \\
      --gt_dir   SOURCE/extrude_test_1500 \\
      --pred_dir output_extrude_cadllm \\
      --out_json results/extrude_cadllm.json \\
      --name "CADLLM-extrude" \\
      [--limit 50]   # for dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm


# ============================================================================
# Constants
# ============================================================================

CANONICAL_DIAGONAL: float = 1000.0  # bbox diagonal scaled to 1000 canonical units
# Surface-sample count for CD / Hausdorff / F-Score. Overridable with
# --n_surface_points; the module-level value is what an unconfigured import sees.
#
# This is not a free parameter: every distance-based metric has a noise floor set
# by it. Two independent samplings of the SAME mesh (i.e. a perfect prediction)
# still differ, because the points do not land in the same places. Measured on
# canonical meshes with diagonal 1000:
#
# Measured max gap between two samplings of the same canonical mesh, and the
# tolerance bands that sit above it (i.e. the ones a perfect prediction can pass):
#
#      8192 pts   floor 21.9   usable: none      0.03 s/sample
#     32768 pts   floor 11.9   usable: 20        0.12 s/sample
#     65536 pts   floor  8.9   usable: 10, 20    0.25 s/sample
#    131072 pts   floor  6.3   usable: 10, 20    0.62 s/sample
#
# 65536 is the knee: it recovers tau=10 for ~2 h across all 20 table cells, which
# is nothing next to inference. tau=1 and tau=5 are unreachable at any practical
# density (a floor under 5 needs ~260k points, under 1 needs ~6M) and tau=1 is in
# any case finer than the STL tessellation the ground truth is stored as -- they
# should not be reported as achievable rates.
N_POINTS: int = 65536
N_VOLUME_PTS: int = 10000           # volume samples for IoU
TOLERANCE_BANDS: Tuple[int, ...] = (1, 5, 10, 20)
PARAM_TOLERANCE_BANDS: Tuple[int, ...] = (1, 5, 10)
F_SCORE_TAU_RATIO: float = 0.02     # 2% of canonical diagonal


# ============================================================================
# Tier 1 — Program-level: parameter extraction & PEM
# ============================================================================

# Mirrors the regex used in CAD-DataGen/generate_*_dataset_1m.py:transform_script.
# Numbers not embedded inside identifiers (so "w0" -> w, not 0).
_NUM_RE = re.compile(r'(?<![a-zA-Z_])(-?\d+\.?\d*)(?![a-zA-Z_])')


def extract_numerical_params(py_text: str) -> List[float]:
    """Extract every numerical literal from a CadQuery .py program."""
    return [float(m.group(1)) for m in _NUM_RE.finditer(py_text)]


def integerize_to_geomcad_range(values: List[float]) -> List[int]:
    """
    Map a list of float values into [1, 1000] integers via min-max normalization.
    Matches the integerization that built the GeomCAD GT corpus, so float-output
    models (e.g., CAD-Recode) can be compared against integer GT in the same
    vocabulary.
    """
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    rng = vmax - vmin
    if rng < 1e-3:
        return [500] * len(values)
    return [
        max(1, min(1000, int(round((v - vmin) / rng * 999 + 1))))
        for v in values
    ]


def parameter_match(pred_py: str, gt_py: str,
                    renormalize_pred: bool = False) -> Optional[Dict[str, Any]]:
    """
    Compute PEM and Tolerant Match @ ±{1, 5, 10}.

    Both sides are rounded and clamped into the GeomCAD [1, 1000] integer
    vocabulary, and nothing else is done to them. This must stay symmetric: a
    model that emits exactly the ground-truth integers has to score PEM 1.0.

    renormalize_pred exists for baselines that emit continuous coordinates
    instead of the GeomCAD vocabulary (base CAD-Recode does). It min-max maps
    the prediction into [1, 1000] so the two are at least on the same scale.
    It is OFF by default because applying it to a model that already predicts
    the right integers destroys them: the map is defined by the prediction's own
    min and max, so 938 becomes 1000 and 54 becomes 58. Evaluating ground truth
    against itself with it enabled scores PEM 8.7% instead of 100% -- measured,
    and the reason this default changed. Any number produced with it on is a
    scale-invariant approximation, not an exact-match rate, and must be labelled
    as such.

    Returns None if either side has no params; otherwise a dict with a
    count_match flag -- when False, PEM/tolerant values cover only the leading
    min(len) params and should be read with caution.
    """
    gt_vals = extract_numerical_params(gt_py)
    pr_vals = extract_numerical_params(pred_py)
    if not gt_vals or not pr_vals:
        return None

    gt_int = [max(1, min(1000, int(round(v)))) for v in gt_vals]
    if renormalize_pred:
        pr_int = integerize_to_geomcad_range(pr_vals)
    else:
        pr_int = [max(1, min(1000, int(round(v)))) for v in pr_vals]

    n = min(len(gt_int), len(pr_int))
    g = np.array(gt_int[:n])
    p = np.array(pr_int[:n])
    diffs = np.abs(g - p)

    out: Dict[str, Any] = {
        'pem': float(np.mean(diffs == 0)),
        'gt_param_count': len(gt_int),
        'pred_param_count': len(pr_int),
        'count_match': len(gt_int) == len(pr_int),
    }
    for k in PARAM_TOLERANCE_BANDS:
        out[f'tolerant_{k}'] = float(np.mean(diffs <= k))
    return out


# ============================================================================
# Mesh canonicalization
# ============================================================================

def canonicalize(mesh: trimesh.Trimesh,
                 ref_bounds: Optional[np.ndarray] = None,
                 ) -> trimesh.Trimesh:
    """
    Center mesh at origin and scale bbox diagonal to CANONICAL_DIAGONAL.

    If ref_bounds (shape (2, 3)) is provided, use those bounds to derive the
    transform — this is how we put pred and GT into the SAME canonical frame.
    A pred whose actual extent differs from GT will then exhibit absolute-
    units mismatch (which is exactly what industrial-precision evaluation
    should expose).
    """
    if ref_bounds is None:
        bmin, bmax = mesh.bounds
    else:
        bmin, bmax = ref_bounds[0], ref_bounds[1]
    center = (bmin + bmax) / 2.0
    diag = float(np.linalg.norm(bmax - bmin))
    if diag < 1e-9:
        return mesh.copy()
    out = mesh.copy()
    out.apply_translation(-center)
    out.apply_scale(CANONICAL_DIAGONAL / diag)
    return out


# ============================================================================
# Tier 1 — Mesh-level industrial metrics
# ============================================================================

def hausdorff_and_tolerance(pred_pts: np.ndarray,
                            gt_pts: np.ndarray,
                            ) -> Dict[str, float]:
    """
    Symmetric Hausdorff distance (in canonical units) plus binary
    Tolerance-Pass indicators at each band τ.
    """
    tree_g = cKDTree(gt_pts)
    tree_p = cKDTree(pred_pts)
    d_p2g, _ = tree_g.query(pred_pts, k=1)
    d_g2p, _ = tree_p.query(gt_pts, k=1)
    hausdorff = float(max(d_p2g.max(), d_g2p.max()))
    out: Dict[str, float] = {'hausdorff_abs': hausdorff}
    for tau in TOLERANCE_BANDS:
        out[f'tol_pass_{tau}'] = float(hausdorff <= tau)
    return out


def volume_error_pct(pred_mesh: trimesh.Trimesh,
                     gt_mesh: trimesh.Trimesh,
                     ) -> float:
    """|V_pred − V_gt| / V_gt — relative error of total enclosed volume."""
    v_gt = float(abs(gt_mesh.volume))
    v_pred = float(abs(pred_mesh.volume))
    if v_gt < 1e-12:
        return float('nan')
    return abs(v_pred - v_gt) / v_gt


# ============================================================================
# Tier 2 — Similarity metrics
# ============================================================================

def chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Standard symmetric squared CD."""
    tree_g = cKDTree(gt_pts)
    tree_p = cKDTree(pred_pts)
    d_p2g, _ = tree_g.query(pred_pts, k=1)
    d_g2p, _ = tree_p.query(gt_pts, k=1)
    return float(np.mean(d_p2g ** 2) + np.mean(d_g2p ** 2))


def f_score(pred_pts: np.ndarray, gt_pts: np.ndarray, tau: float
            ) -> Dict[str, float]:
    """F-Score @ τ in canonical units. Tatarchenko et al., CVPR 2019."""
    tree_g = cKDTree(gt_pts)
    tree_p = cKDTree(pred_pts)
    d_p2g, _ = tree_g.query(pred_pts, k=1)
    d_g2p, _ = tree_p.query(gt_pts, k=1)
    p = float(np.mean(d_p2g < tau))
    r = float(np.mean(d_g2p < tau))
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {'precision': p, 'recall': r, 'f_score': f}


def iou_pointwise(pred_mesh: trimesh.Trimesh,
                  gt_mesh: trimesh.Trimesh,
                  rng: np.random.Generator,
                  n_points: int = N_VOLUME_PTS,
                  ) -> float:
    """
    Volumetric IoU via uniform point sampling in the joint AABB.
    O(n) and avoids the slow + fragile boolean intersection path.
    """
    aabb_min = np.minimum(pred_mesh.bounds[0], gt_mesh.bounds[0])
    aabb_max = np.maximum(pred_mesh.bounds[1], gt_mesh.bounds[1])
    pts = rng.uniform(aabb_min, aabb_max, size=(n_points, 3))
    try:
        in_pred = pred_mesh.contains(pts)
        in_gt = gt_mesh.contains(pts)
    except Exception:
        return float('nan')
    inter = int((in_pred & in_gt).sum())
    union = int((in_pred | in_gt).sum())
    return float(inter / union) if union > 0 else 0.0


# ============================================================================
# Per-sample evaluation
# ============================================================================

def evaluate_sample(gt_mesh_path: Path,
                    gt_py_path: Path,
                    pred_mesh_path: Path,
                    pred_py_path: Path,
                    rng: np.random.Generator,
                    renormalize_pred: bool = False,
                    n_surface_points: int = N_POINTS,
                    n_volume_points: int = N_VOLUME_PTS,
                    ) -> Dict[str, Any]:
    """
    Evaluate one (gt, pred) pair. Returns a flat dict of metrics plus a
    'valid' boolean and (when invalid) a 'failure' reason tag.
    """
    out: Dict[str, Any] = {'valid': False}

    # --- existence + load ---
    if not pred_mesh_path.exists() or not pred_py_path.exists():
        out['failure'] = 'pred_missing'
        return out
    try:
        gt_mesh = trimesh.load(str(gt_mesh_path), force='mesh', process=False)
        pred_mesh = trimesh.load(str(pred_mesh_path), force='mesh', process=False)
    except Exception as e:
        out['failure'] = f'load_error:{type(e).__name__}'
        return out
    if pred_mesh is None or len(getattr(pred_mesh, 'faces', [])) == 0:
        out['failure'] = 'pred_empty'
        return out
    if gt_mesh is None or len(getattr(gt_mesh, 'faces', [])) == 0:
        out['failure'] = 'gt_empty'
        return out

    # --- read .py text (program-level metrics; geometry-independent) ---
    try:
        gt_py = gt_py_path.read_text() if gt_py_path.exists() else ''
        pred_py = pred_py_path.read_text()
    except Exception:
        gt_py = ''
        pred_py = ''
    pmatch = (parameter_match(pred_py, gt_py, renormalize_pred)
              if (gt_py and pred_py) else None)
    if pmatch is not None:
        for k, v in pmatch.items():
            out[f'param_{k}'] = v

    # --- canonicalize using GT bbox; pred uses SAME transform ---
    try:
        gt_bounds = np.array(gt_mesh.bounds)
        gt_canon = canonicalize(gt_mesh)
        pred_canon = canonicalize(pred_mesh, ref_bounds=gt_bounds)
    except Exception as e:
        out['failure'] = f'canon_error:{type(e).__name__}'
        return out

    # --- surface point sampling ---
    try:
        gt_pts, _ = trimesh.sample.sample_surface(gt_canon, n_surface_points)
        pred_pts, _ = trimesh.sample.sample_surface(pred_canon, n_surface_points)
    except Exception as e:
        out['failure'] = f'sample_error:{type(e).__name__}'
        return out

    # --- Tier 1 mesh-level ---
    out.update(hausdorff_and_tolerance(pred_pts, gt_pts))
    out['volume_error_pct'] = volume_error_pct(pred_canon, gt_canon)

    # --- Tier 2 similarity ---
    out['cd_x1000'] = chamfer_distance(pred_pts, gt_pts) * 1000.0
    fs = f_score(pred_pts, gt_pts, tau=F_SCORE_TAU_RATIO * CANONICAL_DIAGONAL)
    for k, v in fs.items():
        out[f'fscore_{k}'] = v
    out['iou'] = iou_pointwise(pred_canon, gt_canon, rng, n_points=n_volume_points)

    out['valid'] = True
    return out


# ============================================================================
# Aggregation
# ============================================================================

def _stats(records: List[Dict[str, Any]], key: str) -> Optional[Dict[str, float]]:
    vals = []
    for r in records:
        if key not in r:
            continue
        v = r[key]
        if isinstance(v, float) and np.isnan(v):
            continue
        vals.append(float(v))
    if not vals:
        return None
    arr = np.asarray(vals)
    return {
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'n': int(arr.size),
    }


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(records)
    valid = [r for r in records if r.get('valid')]
    n_valid = len(valid)
    validity = n_valid / n_total if n_total else 0.0

    summary: Dict[str, Any] = {
        'n_total': n_total,
        'n_valid': n_valid,
        'validity_rate': validity,
        'tier_1_industrial': {},
        'tier_2_similarity': {},
        'tier_3_generation': {'validity_rate': validity},
    }
    if n_valid == 0:
        # also count failure modes for diagnosis
        modes: Dict[str, int] = {}
        for r in records:
            modes[r.get('failure', 'unknown')] = modes.get(r.get('failure', 'unknown'), 0) + 1
        summary['failure_mode_counts'] = modes
        return summary

    # --- Tier 1: mesh ---
    summary['tier_1_industrial']['hausdorff_abs'] = _stats(valid, 'hausdorff_abs')
    for tau in TOLERANCE_BANDS:
        summary['tier_1_industrial'][f'tol_pass_{tau}'] = _stats(valid, f'tol_pass_{tau}')
    summary['tier_1_industrial']['volume_error_pct'] = _stats(valid, 'volume_error_pct')

    # --- Tier 1: program ---
    summary['tier_1_industrial']['param_pem'] = _stats(valid, 'param_pem')
    for k in PARAM_TOLERANCE_BANDS:
        summary['tier_1_industrial'][f'param_tolerant_{k}'] = _stats(valid, f'param_tolerant_{k}')
    summary['tier_1_industrial']['param_count_match_rate'] = _stats(valid, 'param_count_match')

    # --- Tier 2 ---
    summary['tier_2_similarity']['iou'] = _stats(valid, 'iou')
    summary['tier_2_similarity']['cd_x1000'] = _stats(valid, 'cd_x1000')
    summary['tier_2_similarity']['fscore_f_score'] = _stats(valid, 'fscore_f_score')
    summary['tier_2_similarity']['fscore_precision'] = _stats(valid, 'fscore_precision')
    summary['tier_2_similarity']['fscore_recall'] = _stats(valid, 'fscore_recall')

    # diagnostic: failure breakdown across all records
    modes: Dict[str, int] = {}
    for r in records:
        if not r.get('valid'):
            modes[r.get('failure', 'unknown')] = modes.get(r.get('failure', 'unknown'), 0) + 1
    summary['failure_mode_counts'] = modes
    return summary


def _fmt_pct(stat: Optional[Dict[str, float]]) -> str:
    if stat is None:
        return '   n/a'
    return f"{stat['median']*100:6.2f}% (mean {stat['mean']*100:.2f}%)"


def _fmt_num(stat: Optional[Dict[str, float]], scale: float = 1.0,
             unit: str = '') -> str:
    if stat is None:
        return '   n/a'
    return f"{stat['median']*scale:8.4f}{unit} (mean {stat['mean']*scale:.4f}{unit})"


def print_summary(summary: Dict[str, Any], name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"Validity Rate          {summary['validity_rate']*100:6.2f}%   "
          f"(n_valid {summary['n_valid']}/{summary['n_total']})")

    t1 = summary['tier_1_industrial']
    t2 = summary['tier_2_similarity']

    print(f"\n--- Tier 1: Industrial Precision ---")
    print(f"PEM                    {_fmt_pct(t1.get('param_pem'))}")
    for k in PARAM_TOLERANCE_BANDS:
        print(f"Param Tolerant @ ±{k:<3}  {_fmt_pct(t1.get(f'param_tolerant_{k}'))}")
    print(f"Param-count match      {_fmt_pct(t1.get('param_count_match_rate'))}")
    print(f"Hausdorff (canonical)  {_fmt_num(t1.get('hausdorff_abs'))}")
    for tau in TOLERANCE_BANDS:
        print(f"Tol Pass @ τ={tau:<3}      {_fmt_pct(t1.get(f'tol_pass_{tau}'))}")
    print(f"Volume Error           {_fmt_pct(t1.get('volume_error_pct'))}")

    print(f"\n--- Tier 2: Similarity ---")
    print(f"IoU                    {_fmt_num(t2.get('iou'))}")
    print(f"CD ×10³                {_fmt_num(t2.get('cd_x1000'))}")
    print(f"F-Score @ 2%           {_fmt_num(t2.get('fscore_f_score'))}")

    fm = summary.get('failure_mode_counts', {})
    if fm:
        print(f"\n--- Failure modes ---")
        for k, v in sorted(fm.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<30} {v}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt_dir', required=True,
                    help='GT test set dir; expects subdirs <id>/<id>.{stl,py}')
    ap.add_argument('--pred_dir', required=True,
                    help='Prediction dir; same layout as gt_dir')
    ap.add_argument('--out_json', required=True,
                    help='Path to write aggregated metrics JSON')
    ap.add_argument('--per_sample_json', default=None,
                    help='Optional: also dump per-sample records here')
    ap.add_argument('--limit', type=int, default=None,
                    help='Limit sample count (use --limit 50 for dry-run)')
    ap.add_argument('--use_pred_ids', action='store_true',
                    help='Iterate IDs from pred_dir instead of gt_dir (useful for dry-run on '
                         'legacy outputs where gt_dir is the full 1M training pool)')
    ap.add_argument('--n_surface_points', type=int, default=N_POINTS,
                    help="surface samples per mesh for Hausdorff/CD/F-Score. Sets the "
                         "noise floor of every distance metric: at 8192 two samplings of "
                         "an identical mesh already differ by a max gap of 22 canonical "
                         "units, so tolerance bands below that can never be passed.")
    ap.add_argument('--n_volume_points', type=int, default=N_VOLUME_PTS,
                    help="volume samples per mesh for IoU")
    ap.add_argument('--renormalize_pred', action='store_true',
                    help="min-max map predicted numbers into [1,1000] before comparing. "
                         "Only for baselines that emit continuous coordinates instead of "
                         "the GeomCAD vocabulary. It rescales by the prediction's own min "
                         "and max, so a model that already predicts the correct integers "
                         "is penalised: ground truth scored against itself drops from "
                         "PEM 100%% to 8.7%%. Any number produced with this flag is a "
                         "scale-invariant approximation, not an exact-match rate.")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--name', default='unnamed_run',
                    help='Run name (recorded in output JSON)')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    if not gt_dir.is_dir():
        raise SystemExit(f"GT dir not found: {gt_dir}")
    if not pred_dir.is_dir():
        raise SystemExit(f"Pred dir not found: {pred_dir}")

    # Every subdirectory is taken to be one sample, so anything else living in the
    # GT root -- a shard index built for split inference, a scratch folder -- silently
    # inflates the denominator and shows up as pred_missing. Names starting with "_"
    # or "." are therefore skipped; keep helper directories outside the root or
    # prefixed, and never inside it unprefixed.
    id_source_dir = pred_dir if args.use_pred_ids else gt_dir
    sample_ids = sorted([d.name for d in id_source_dir.iterdir()
                         if d.is_dir() and not d.name.startswith(('_', '.'))])
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    print(f"GeomCAD-Eval v2.0  |  {args.name}")
    print(f"  GT   : {gt_dir}  ({len(sample_ids)} samples)")
    print(f"  Pred : {pred_dir}")
    print(f"  Out  : {args.out_json}")
    print(f"  Canonical diagonal = {CANONICAL_DIAGONAL}, "
          f"surface pts = {args.n_surface_points}, volume pts = {args.n_volume_points}")

    records: List[Dict[str, Any]] = []
    for sid in tqdm(sample_ids, desc='evaluating'):
        rec: Dict[str, Any] = {'sample_id': sid}
        # Some inference outputs use `.txt` instead of `.py` for the predicted CAD program;
        # fall back to `.txt` if `.py` is absent.
        pred_py_default = pred_dir / sid / f'{sid}.py'
        pred_py_path = pred_py_default if pred_py_default.exists() else (pred_dir / sid / f'{sid}.txt')
        rec.update(evaluate_sample(
            gt_mesh_path=gt_dir / sid / f'{sid}.stl',
            gt_py_path=gt_dir / sid / f'{sid}.py',
            pred_mesh_path=pred_dir / sid / f'{sid}.stl',
            pred_py_path=pred_py_path,
            rng=rng,
            renormalize_pred=args.renormalize_pred,
            n_surface_points=args.n_surface_points,
            n_volume_points=args.n_volume_points,
        ))
        records.append(rec)

    summary = aggregate(records)
    summary['run_name'] = args.name
    summary['gt_dir'] = str(gt_dir)
    summary['pred_dir'] = str(pred_dir)
    summary['canonical_diagonal'] = CANONICAL_DIAGONAL
    summary['n_surface_points'] = args.n_surface_points
    summary['n_volume_points'] = args.n_volume_points

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Aggregate metrics → {out_path}")

    if args.per_sample_json:
        per_path = Path(args.per_sample_json)
        per_path.parent.mkdir(parents=True, exist_ok=True)
        with open(per_path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"✓ Per-sample records → {per_path}")

    print_summary(summary, args.name)


if __name__ == '__main__':
    main()
