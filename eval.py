"""
GeomCAD-Eval v2.1 — Industrial-Precision-First Evaluation Protocol
==================================================================

Three tiers of metrics:

  Tier 1 — Industrial Precision (paper headline)
    Program-level (parameter vocabulary [1, 1000], 0 allowed as semantic marker):
      - PEM strict           (position-wise sequence exact match)
      - rPEM                 (multiset recall, denominator = len(gt))
      - rPEM@k, k ∈ {1,5,10} (multiset recall under |Δ| ≤ k)
      - MFM_op (gen4)        (Counter over {extrude, revolve, sweep, loft})
    Mesh-level (canonical space, bbox diagonal = 1000):
      - TVF@δ, δ ∈ {1%, 5%}  (relative volume error ≤ δ)
      - Hausdorff Distance (canonical units, legacy)

  Tier 2 — Similarity (cross-paper compatibility)
      - IoU (manifold3d exact boolean; falls back to point-in-volume)
      - CD% (symmetric linear Chamfer, % of canonical diagonal)
      - HD% (symmetric Hausdorff, % of canonical diagonal)
      - F-Score @ τ = 2% of canonical diagonal
      - VGF@5% (F@2% gated by |ΔV|/V_gt ≤ 5%)

  Tier 3 — Generation
      - Validity Rate (STL loads with non-empty triangle set after the harness's
        30s exec_preds run)

Coordinate frames:
  By default the pred mesh is transformed using the SAME center+scale derived
  from the GT bbox, preserving scale-mismatch signal in TVF/CD/HD/IoU. With
  --pred_post, the pred is canonicalized by its OWN bbox (shape-only reporting;
  matches +post baseline convention in the CAD-Recode paper).

Aggregation (main-table columns):
  - Rates over valid predictions (TVF@δ, PEM, rPEM, rPEM@10, MFMop, IoU, F@2%) : mean
  - Rates over the full split (VGF@5%, Validity)                                 : mean
  - CD% and HD%                                                                   : median
    (per-sample distributions are heavy-tailed)
  - Surface sampling is seeded per sample from --seed + crc32(sample_id), so a
    given sample gets the same sampling seed across every method's evaluation.

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
import io
import json
import os
import re
import tokenize
import zlib
from multiprocessing import get_context
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

# Contexts in which a leading '-' should be read as a unary minus attached to
# the following NUMBER, rather than a binary subtraction. Anything that expects
# a value on its right side counts. Empty prev_op (start of stream) also counts.
_UNARY_MINUS_AFTER = {None, '', '(', ',', '=', '[', '{', ':', '*', '/', '+', '-',
                      '**', '//', '%', '&', '|', '^', '<', '>', '<=', '>=',
                      '==', '!=', 'return'}


def extract_numerical_params(py_text: str) -> List[float]:
    """Return every numerical literal from a CadQuery program.

    Uses Python's tokenizer, which handles the cases a regex misses:
      * identifiers with digits (``w12``, ``sketch_20``) never contribute
        digits to the output;
      * multi-character numeric syntax (``1e-3``, ``.5``, ``1_000``) is parsed
        as a single NUMBER token, not split;
      * numbers inside comments and string literals are skipped.

    Unary minus is recovered by pairing a preceding ``-`` operator with the
    next NUMBER whenever the ``-`` sits in a context that expects a value
    (start of stream, after ``(``/``,``/``=``, etc.). Subtraction between two
    values (``x - 1``) leaves the number positive.
    """
    if not py_text:
        return []
    out: List[float] = []
    prev = None  # last non-trivia token string ('' at start)
    pending_neg = False
    try:
        toks = tokenize.generate_tokens(io.StringIO(py_text).readline)
        for tok in toks:
            t, s = tok.type, tok.string
            if t in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
                     tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
                continue
            if t == tokenize.OP and s == '-' and prev in _UNARY_MINUS_AFTER:
                pending_neg = True
                # Do not update prev; a chain "= - -1" would then re-arm.
                continue
            if t == tokenize.NUMBER:
                try:
                    v = float(s.replace('_', ''))
                except ValueError:
                    pending_neg = False
                    prev = s
                    continue
                out.append(-v if pending_neg else v)
                pending_neg = False
                prev = s
                continue
            # Anything else clears any pending unary minus.
            pending_neg = False
            prev = s
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        # Fall through: whatever we did parse before the error still counts.
        pass
    return out


def multiset_recall(pred: List[int], gt: List[int], k: int = 0) -> float:
    """Greedy multiset recall on sorted lists, one-to-one matching, no double count.

    Denominator is len(gt) (paper wording: recall). A pred literal matches a gt
    literal iff |pred - gt| <= k, and each is consumed at most once. Order
    invariant, unlike position-wise partial credit.
    """
    if not gt:
        return 0.0
    p, g = sorted(pred), sorted(gt)
    i = j = m = 0
    while i < len(p) and j < len(g):
        if abs(p[i] - g[j]) <= k:
            m += 1; i += 1; j += 1
        elif p[i] < g[j]:
            i += 1
        else:
            j += 1
    return m / len(g)


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
    if not gt_vals:
        return None
    pr_vals = extract_numerical_params(pred_py or '')

    # 0 is a semantic marker (revolve start angle / sweep origin) so it must
    # not be clamped up to 1: doing so would let GT 0 alias with predicted 1.
    gt_int = [max(0, min(1000, int(round(v)))) for v in gt_vals]

    # Empty pred (valid mesh but zero literals): score 0 across the board rather than
    # drop the sample. Dropping inflates rPEM / PEM by removing failure cases from the
    # denominator.
    if not pr_vals:
        out: Dict[str, Any] = {
            'pem': 0.0,
            'pem_strict': 0.0,
            'rpem': 0.0,
            'rpem_precision': 0.0,
            'gt_param_count': len(gt_int),
            'pred_param_count': 0,
            'count_match': False,
        }
        for k in PARAM_TOLERANCE_BANDS:
            out[f'tolerant_{k}'] = 0.0
            out[f'rpem_{k}'] = 0.0
        return out

    if renormalize_pred:
        pr_int = integerize_to_geomcad_range(pr_vals)
    else:
        pr_int = [max(0, min(1000, int(round(v)))) for v in pr_vals]

    n = min(len(gt_int), len(pr_int))
    g = np.array(gt_int[:n])
    p = np.array(pr_int[:n])
    diffs = np.abs(g - p)

    count_match = len(gt_int) == len(pr_int)
    out: Dict[str, Any] = {
        'pem': float(np.mean(diffs == 0)),
        'pem_strict': float(1.0 if (count_match and (diffs == 0).all()) else 0.0),
        'gt_param_count': len(gt_int),
        'pred_param_count': len(pr_int),
        'count_match': count_match,
    }
    for k in PARAM_TOLERANCE_BANDS:
        out[f'tolerant_{k}'] = float(np.mean(diffs <= k))
    out['rpem'] = multiset_recall(pr_int, gt_int, k=0)
    out['rpem_precision'] = multiset_recall(gt_int, pr_int, k=0)
    for k in PARAM_TOLERANCE_BANDS:
        out[f'rpem_{k}'] = multiset_recall(pr_int, gt_int, k=k)
    return out


# ============================================================================
# Tier 3 / structural gating — op-sequence match and gated parameter match
# ============================================================================

# Matches `.name(` for every method call in the CadQuery chain. Ignores static
# `cq.Workplane(...)` calls (no leading dot), so we get exactly the ops used to
# build the shape.
_OP_CALL_RE = re.compile(r'\.([a-zA-Z_][a-zA-Z_0-9]*)\s*\(')

# Op-sequence F1 threshold above which the gated parameter match trusts
# position-indexed number matching. Below the threshold the numbers may be
# aligned by coincidence rather than by semantic correspondence, so gated
# metrics get 0 for that sample. 0.8 admits small structural drift (a couple
# of extra .translate() or missing chamfer) while blocking wildly different
# call chains.
GATED_OP_F1_THRESHOLD: float = 0.8


def extract_op_sequence(py_text: str) -> List[str]:
    """The ordered sequence of `.method(` calls in the CadQuery program."""
    return _OP_CALL_RE.findall(py_text)


def op_sequence_match(pred_py: str, gt_py: str) -> Dict[str, float]:
    """Multiset op-level P/R/F1 + exact-sequence flag.

    Uses multiset intersection (Counter & Counter): a repeated op matches gt
    only up to gt's count, so a pred that spams the same op does not get
    false credit and a pred that skips ops loses recall. F1 blends the two.
    op_seq_exact is the strictest form: full ordered equality.
    """
    from collections import Counter
    pred_ops = extract_op_sequence(pred_py or '')
    gt_ops = extract_op_sequence(gt_py)
    if not gt_ops:
        empty_match = 1.0 if not pred_ops else 0.0
        return {'op_precision': 0.0, 'op_recall': 0.0, 'op_f1': 0.0,
                'op_seq_exact': empty_match,
                'mfm_op_strict': empty_match,
                'mfm_op_gen4': empty_match,
                'op_count_pred': float(len(pred_ops)),
                'op_count_gt': 0.0}
    pc = Counter(pred_ops)
    gc = Counter(gt_ops)
    matches = sum((pc & gc).values())
    n_pred = sum(pc.values())
    n_gt = sum(gc.values())
    p = matches / n_pred if n_pred else 0.0
    r = matches / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    _GEN4 = {'extrude', 'revolve', 'sweep', 'loft'}
    pc4 = Counter({k: v for k, v in pc.items() if k in _GEN4})
    gc4 = Counter({k: v for k, v in gc.items() if k in _GEN4})
    mfm_gen4 = 1.0 if pc4 == gc4 else 0.0
    return {
        'op_precision':  float(p),
        'op_recall':     float(r),
        'op_f1':         float(f1),
        'op_seq_exact':  1.0 if pred_ops == gt_ops else 0.0,
        'mfm_op_strict': 1.0 if pc == gc else 0.0,
        'mfm_op_gen4':   float(mfm_gen4),
        'op_count_pred': float(n_pred),
        'op_count_gt':   float(n_gt),
    }


def gated_parameter_match(pred_py: str, gt_py: str,
                          renormalize_pred: bool = False,
                          op_f1_threshold: float = GATED_OP_F1_THRESHOLD,
                          ) -> Optional[Dict[str, Any]]:
    """PEM / Tolerant scores gated on op-sequence F1 threshold.

    parameter_match compares numbers by POSITION. If pred's op sequence
    differs from gt's, positional indices no longer refer to the same
    semantic slots, and a raw PEM/Tolerant can hit high values by
    coincidence (e.g. a box output can score Tol@±10 100% against a loft
    gt if its three literals happen to sit within 10 of gt's first three).

    Gating: samples whose op-sequence F1 is below op_f1_threshold get
    gated_pem = gated_tolerant_k = 0.0. Samples that clear the gate get the
    unmodified parameter_match values. gate_pass reports which side each
    sample fell on.

    Returns None when parameter_match returns None.
    """
    pm = parameter_match(pred_py, gt_py, renormalize_pred)
    if pm is None:
        return None
    om = op_sequence_match(pred_py, gt_py)
    gate = om['op_f1'] >= op_f1_threshold
    out: Dict[str, Any] = {'gate_pass': 1.0 if gate else 0.0,
                            'op_f1_used': om['op_f1']}
    for k in ['pem'] + [f'tolerant_{tt}' for tt in PARAM_TOLERANCE_BANDS]:
        if k in pm:
            out[f'gated_{k}'] = float(pm[k]) if gate else 0.0
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


def chamfer_pct(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Symmetric linear Chamfer as % of canonical diagonal (paper CD%)."""
    tree_g = cKDTree(gt_pts)
    tree_p = cKDTree(pred_pts)
    d_p2g, _ = tree_g.query(pred_pts, k=1)
    d_g2p, _ = tree_p.query(gt_pts, k=1)
    return float((np.mean(d_p2g) + np.mean(d_g2p)) / 2.0 / CANONICAL_DIAGONAL * 100.0)


def hausdorff_pct(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Symmetric Hausdorff as % of canonical diagonal (paper HD%)."""
    tree_g = cKDTree(gt_pts)
    tree_p = cKDTree(pred_pts)
    d_p2g, _ = tree_g.query(pred_pts, k=1)
    d_g2p, _ = tree_p.query(gt_pts, k=1)
    return float(max(np.max(d_p2g), np.max(d_g2p)) / CANONICAL_DIAGONAL * 100.0)


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
    Volumetric IoU. Prefers manifold3d exact boolean when both meshes convert
    cleanly (works even without pyembree/rtree). Falls back to point-in-volume
    via trimesh.contains, then to NaN.

    Fallback samples the JOINT bbox of pred and GT (expanded by 10%), so a
    pred that pokes outside the GT box does not have its outside-mass silently
    ignored.
    """
    # --- primary: manifold3d exact IoU ---
    # manifold3d.Manifold does not raise on non-manifold input; it silently
    # returns an empty Manifold whose volume is 0. That path used to be logged
    # as IoU=0 (v_inter=0, v_union=v_gt>eps). Reject empty operands explicitly
    # so the sample falls back to point-in-volume instead of scoring a false 0.
    try:
        import manifold3d
        def _tri_to_m(tri):
            m = manifold3d.Mesh(
                vert_properties=np.asarray(tri.vertices, dtype=np.float32),
                tri_verts=np.asarray(tri.faces, dtype=np.uint32),
            )
            mm = manifold3d.Manifold(m)
            if mm.is_empty() or float(mm.volume()) <= 1e-9:
                raise ValueError("manifold3d could not build a solid from mesh")
            return mm
        m_pred = _tri_to_m(pred_mesh)
        m_gt   = _tri_to_m(gt_mesh)
        v_inter = (m_pred ^ m_gt).volume()   # intersection
        v_union = (m_pred + m_gt).volume()   # union
        if v_union > 1e-9:
            return float(max(0.0, min(1.0, v_inter / v_union)))
    except Exception:
        pass

    # --- fallback: point sampling with trimesh.contains ---
    if not pred_mesh.is_watertight:
        try:
            pred_mesh = pred_mesh.copy()
            trimesh.repair.fill_holes(pred_mesh)
        except Exception:
            pass
    if not gt_mesh.is_watertight:
        try:
            gt_mesh = gt_mesh.copy()
            trimesh.repair.fill_holes(gt_mesh)
        except Exception:
            pass
    lo = np.minimum(gt_mesh.bounds[0], pred_mesh.bounds[0])
    hi = np.maximum(gt_mesh.bounds[1], pred_mesh.bounds[1])
    pad = 0.1 * (hi - lo)
    pts = rng.uniform(lo - pad, hi + pad, size=(n_points, 3))
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
                    pred_post: bool = False,
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
        # process=True merges duplicate vertices in STL so meshes are
        # topologically watertight, which manifold3d needs to compute volume.
        # process=False leaves each face's 3 vertices independent, making
        # is_watertight=False and IoU degenerate to NaN.
        gt_mesh = trimesh.load(str(gt_mesh_path), force='mesh', process=True)
        pred_mesh = trimesh.load(str(pred_mesh_path), force='mesh', process=True)
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
    except Exception:
        gt_py = ''
    try:
        pred_py = pred_py_path.read_text() if pred_py_path.exists() else ''
    except Exception:
        pred_py = ''
    pmatch = (parameter_match(pred_py, gt_py, renormalize_pred)
              if gt_py else None)
    if pmatch is not None:
        for k, v in pmatch.items():
            out[f'param_{k}'] = v

    # --- op-sequence match (Tier 3) + gated parameter match ---
    if gt_py:
        omatch = op_sequence_match(pred_py, gt_py)
        out.update(omatch)
        gmatch = gated_parameter_match(pred_py, gt_py, renormalize_pred)
        if gmatch is not None:
            out.update(gmatch)

    # --- canonicalize: default uses GT bbox for both (frame-alignment eval);
    #     pred_post=True lets pred use its own bbox (symmetric-inverse +post,
    #     matches CAD-Recode paper's shape-only reporting for baselines). ---
    try:
        gt_canon = canonicalize(gt_mesh)
        if pred_post:
            pred_canon = canonicalize(pred_mesh)
        else:
            gt_bounds = np.array(gt_mesh.bounds)
            pred_canon = canonicalize(pred_mesh, ref_bounds=gt_bounds)
    except Exception as e:
        out['failure'] = f'canon_error:{type(e).__name__}'
        return out

    # --- surface point sampling (seeded so numbers are reproducible) ---
    try:
        seed_g, seed_p = (int(s) for s in rng.integers(2**31, size=2))
        gt_pts, _ = trimesh.sample.sample_surface(gt_canon, n_surface_points, seed=seed_g)
        pred_pts, _ = trimesh.sample.sample_surface(pred_canon, n_surface_points, seed=seed_p)
    except Exception as e:
        out['failure'] = f'sample_error:{type(e).__name__}'
        return out

    # --- Tier 1 mesh-level ---
    out.update(hausdorff_and_tolerance(pred_pts, gt_pts))
    out['volume_error_pct'] = volume_error_pct(pred_canon, gt_canon)

    # --- Tier 2 similarity ---
    out['cd_x1000'] = chamfer_distance(pred_pts, gt_pts) * 1000.0
    out['cd_pct']   = chamfer_pct(pred_pts, gt_pts)
    out['hd_pct']   = hausdorff_pct(pred_pts, gt_pts)
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
    # v2.1 additions: true multiset recall (paper rPEM definition), full lists
    summary['tier_1_industrial']['param_rpem'] = _stats(valid, 'param_rpem')
    summary['tier_1_industrial']['param_rpem_precision'] = _stats(valid, 'param_rpem_precision')
    for k in PARAM_TOLERANCE_BANDS:
        summary['tier_1_industrial'][f'param_rpem_{k}'] = _stats(valid, f'param_rpem_{k}')

    # --- Tier 1 (structurally-gated PEM/Tol) ---
    summary['tier_1_industrial']['gate_pass_rate'] = _stats(valid, 'gate_pass')
    summary['tier_1_industrial']['gated_pem']     = _stats(valid, 'gated_pem')
    for k in PARAM_TOLERANCE_BANDS:
        summary['tier_1_industrial'][f'gated_tolerant_{k}'] = _stats(valid, f'gated_tolerant_{k}')

    # --- Tier 3 (op-sequence match) ---
    summary['tier_3_generation']['op_precision']       = _stats(valid, 'op_precision')
    summary['tier_3_generation']['op_recall']          = _stats(valid, 'op_recall')
    summary['tier_3_generation']['op_f1']              = _stats(valid, 'op_f1')
    summary['tier_3_generation']['op_seq_exact_rate']  = _stats(valid, 'op_seq_exact')

    # --- Tier 2 ---
    summary['tier_2_similarity']['iou'] = _stats(valid, 'iou')
    summary['tier_2_similarity']['cd_x1000'] = _stats(valid, 'cd_x1000')
    summary['tier_2_similarity']['cd_pct']   = _stats(valid, 'cd_pct')
    summary['tier_2_similarity']['hd_pct']   = _stats(valid, 'hd_pct')
    summary['tier_2_similarity']['fscore_f_score'] = _stats(valid, 'fscore_f_score')
    summary['tier_2_similarity']['fscore_precision'] = _stats(valid, 'fscore_precision')
    summary['tier_2_similarity']['fscore_recall'] = _stats(valid, 'fscore_recall')

    # --- Main-table 12-metric additions ---
    # Strict PEM: per-sample 1.0 iff every literal exactly matches AND count matches
    summary['tier_1_industrial']['pem_strict'] = _stats(valid, 'param_pem_strict')
    # Strict MFM_op: per-sample 1.0 iff Counter(pred_ops) == Counter(gt_ops)
    summary['tier_1_industrial']['mfm_op_strict'] = _stats(valid, 'mfm_op_strict')
    # gen4 MFM_op: per-sample 1.0 iff Counter of extrude/revolve/sweep/loft matches
    summary['tier_1_industrial']['mfm_op_gen4']   = _stats(valid, 'mfm_op_gen4')

    # TVF@delta: fraction of valid predictions with |V_pred - V_gt|/V_gt <= delta
    vol_errs = np.array([r['volume_error_pct'] for r in valid
                         if 'volume_error_pct' in r
                         and not (isinstance(r['volume_error_pct'], float) and np.isnan(r['volume_error_pct']))])
    if vol_errs.size > 0:
        for delta_pct in (1, 5, 10, 20):
            pass_rate = float(np.mean(vol_errs <= delta_pct / 100.0))
            summary['tier_1_industrial'][f'tvf_at_{delta_pct}'] = {
                'median': pass_rate, 'mean': pass_rate,
                'std': 0.0, 'n': int(vol_errs.size),
            }

    # VGF@5%: unconditional over all n_total, invalid samples contribute 0
    vgf_scores: List[float] = []
    for r in records:
        if not r.get('valid'):
            vgf_scores.append(0.0)
            continue
        f = float(r.get('fscore_f_score', 0.0) or 0.0)
        ve = r.get('volume_error_pct', float('inf'))
        try:
            ve = float(ve)
        except Exception:
            ve = float('inf')
        if np.isnan(ve) or ve > 0.05:
            vgf_scores.append(0.0)
        else:
            vgf_scores.append(f)
    if vgf_scores:
        arr_vgf = np.asarray(vgf_scores)
        summary['tier_2_similarity']['vgf_at_5'] = {
            'median': float(np.median(arr_vgf)),
            'mean': float(np.mean(arr_vgf)),
            'std': float(np.std(arr_vgf)),
            'n': int(arr_vgf.size),
        }

    # diagnostic: failure breakdown across all records
    modes: Dict[str, int] = {}
    for r in records:
        if not r.get('valid'):
            modes[r.get('failure', 'unknown')] = modes.get(r.get('failure', 'unknown'), 0) + 1
    summary['failure_mode_counts'] = modes
    return summary


def _fmt_pct(stat: Optional[Dict[str, float]]) -> str:
    """Rate-style metric: print the paper-primary aggregate (mean) first."""
    if stat is None:
        return '   n/a'
    return f"{stat['mean']*100:6.2f}% (median {stat['median']*100:.2f}%)"


def _fmt_med_pct(stat: Optional[Dict[str, float]]) -> str:
    """Distance-style metric shown as %: paper reports median."""
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

    print(f"\n--- Tier 1: Industrial Precision (main-table keys marked *) ---")
    print(f"PEM (strict)         * {_fmt_pct(t1.get('pem_strict'))}")
    print(f"rPEM                 * {_fmt_pct(t1.get('param_rpem'))}")
    print(f"rPEM@10              * {_fmt_pct(t1.get('param_rpem_10'))}")
    print(f"MFMop (gen4)         * {_fmt_pct(t1.get('mfm_op_gen4'))}")
    print(f"(legacy) pem pos-wise  {_fmt_pct(t1.get('param_pem'))}")
    for k in PARAM_TOLERANCE_BANDS:
        print(f"(legacy) tol pos @ ±{k:<2} {_fmt_pct(t1.get(f'param_tolerant_{k}'))}")
    print(f"(legacy) MFMop strict  {_fmt_pct(t1.get('mfm_op_strict'))}")
    print(f"Param-count match      {_fmt_pct(t1.get('param_count_match_rate'))}")
    print(f"Hausdorff (canonical)  {_fmt_num(t1.get('hausdorff_abs'))}")
    for tau in TOLERANCE_BANDS:
        print(f"Tol Pass @ τ={tau:<3}      {_fmt_pct(t1.get(f'tol_pass_{tau}'))}")
    print(f"Volume Error           {_fmt_pct(t1.get('volume_error_pct'))}")

    print(f"\n--- Tier 1: Structurally-gated (op_f1 >= {GATED_OP_F1_THRESHOLD}) ---")
    print(f"Gate Pass Rate         {_fmt_pct(t1.get('gate_pass_rate'))}")
    print(f"Gated PEM              {_fmt_pct(t1.get('gated_pem'))}")
    for k in PARAM_TOLERANCE_BANDS:
        print(f"Gated Tolerant @ ±{k:<3} {_fmt_pct(t1.get(f'gated_tolerant_{k}'))}")

    print(f"\n--- Tier 2: Similarity (main-table keys marked *) ---")
    print(f"IoU                  * {_fmt_pct(t2.get('iou'))}")
    print(f"CD%                  * {_fmt_med_pct({'median': (t2.get('cd_pct') or {}).get('median', 0)/100, 'mean': (t2.get('cd_pct') or {}).get('mean', 0)/100})}")
    print(f"HD%                  * {_fmt_med_pct({'median': (t2.get('hd_pct') or {}).get('median', 0)/100, 'mean': (t2.get('hd_pct') or {}).get('mean', 0)/100})}")
    print(f"F-Score @ 2%         * {_fmt_pct(t2.get('fscore_f_score'))}")
    print(f"(legacy) CD x1000      {_fmt_num(t2.get('cd_x1000'))}")

    t3 = summary.get('tier_3_generation', {})
    print(f"\n--- Tier 3: Generation ---")
    print(f"Validity Rate          {_fmt_pct({'median': summary['validity_rate'], 'mean': summary['validity_rate']})}")
    print(f"Op Sequence F1         {_fmt_pct(t3.get('op_f1'))}")
    print(f"Op Precision           {_fmt_pct(t3.get('op_precision'))}")
    print(f"Op Recall              {_fmt_pct(t3.get('op_recall'))}")
    print(f"Op Sequence Exact      {_fmt_pct(t3.get('op_seq_exact_rate'))}")

    fm = summary.get('failure_mode_counts', {})
    if fm:
        print(f"\n--- Failure modes ---")
        for k, v in sorted(fm.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<30} {v}")


# ============================================================================
# Main
# ============================================================================

def _eval_one_worker(task):
    """Per-sample worker for the multiprocessing pool. Top-level so spawn can pickle it."""
    (sid, gt_dir, pred_dir, seed, renormalize_pred, pred_post,
     n_surface_points, n_volume_points) = task
    gt_dir = Path(gt_dir); pred_dir = Path(pred_dir)
    rng = np.random.default_rng(seed)
    pred_py_default = pred_dir / sid / f'{sid}.py'
    pred_py_path = (pred_py_default if pred_py_default.exists()
                    else pred_dir / sid / f'{sid}.txt')
    rec: Dict[str, Any] = {'sample_id': sid}
    rec.update(evaluate_sample(
        gt_mesh_path=gt_dir / sid / f'{sid}.stl',
        gt_py_path=gt_dir / sid / f'{sid}.py',
        pred_mesh_path=pred_dir / sid / f'{sid}.stl',
        pred_py_path=pred_py_path,
        rng=rng,
        renormalize_pred=renormalize_pred,
        pred_post=pred_post,
        n_surface_points=n_surface_points,
        n_volume_points=n_volume_points,
    ))
    return rec


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
    ap.add_argument('--pred_post', action='store_true',
                    help="Mesh-level +post: canonicalize pred with its OWN bbox instead "
                         "of the GT bbox, so shape-similarity metrics (TVF, CD, HD, F@2, "
                         "IoU) reflect symmetric-inverse comparison (as reported by the "
                         "CAD-Recode paper for baselines).")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--name', default='unnamed_run',
                    help='Run name (recorded in output JSON)')
    ap.add_argument('--workers', type=int, default=1,
                    help='Parallel worker processes for per-sample eval. Default 1 = serial. '
                         'CPU-bound; set close to allocated cores (e.g. 16, '
                         '32-56 on the local 64-core box).')
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

    print(f"GeomCAD-Eval v2.1  |  {args.name}")
    print(f"  GT   : {gt_dir}  ({len(sample_ids)} samples)")
    print(f"  Pred : {pred_dir}")
    print(f"  Out  : {args.out_json}")
    print(f"  Canonical diagonal = {CANONICAL_DIAGONAL}, "
          f"surface pts = {args.n_surface_points}, volume pts = {args.n_volume_points}")

    # Seed each sample from its own ID (not its list-index) so a method that
    # skips samples, --limit, or a different sample_ids ordering does not shift
    # every subsequent seed and reintroduce cross-method noise on identical GT.
    tasks = [(sid, str(gt_dir), str(pred_dir),
              args.seed + zlib.crc32(sid.encode()),
              args.renormalize_pred, args.pred_post,
              args.n_surface_points, args.n_volume_points)
             for sid in sample_ids]

    records: List[Dict[str, Any]] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc='evaluating'):
            records.append(_eval_one_worker(task))
    else:
        # spawn: manifold3d + trimesh + OCCT are not fork-safe. Fresh worker per task
        # avoids OCCT memory accumulation over long runs.
        ctx = get_context('spawn')
        with ctx.Pool(args.workers, maxtasksperchild=50) as pool:
            for rec in tqdm(pool.imap_unordered(_eval_one_worker, tasks, chunksize=1),
                            total=len(tasks), desc=f'evaluating (w={args.workers})'):
                records.append(rec)

    # Sort records so aggregation (mean/std) sums in a deterministic order;
    # multiprocessing.Pool.imap_unordered otherwise reorders results between
    # runs and shifts the last few float bits of every mean.
    records.sort(key=lambda r: r.get('sample_id', ''))
    summary = aggregate(records)
    summary['run_name'] = args.name
    summary['gt_dir'] = str(gt_dir)
    summary['pred_dir'] = str(pred_dir)
    summary['canonical_diagonal'] = CANONICAL_DIAGONAL
    summary['n_surface_points'] = args.n_surface_points
    summary['n_volume_points'] = args.n_volume_points
    summary['pred_post'] = bool(args.pred_post)
    summary['renormalize_pred'] = bool(args.renormalize_pred)

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
