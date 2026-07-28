"""GeomCAD-Eval: compute Tier-1 metrics for extrude test split.

For each (predicted .stl, GT .stl) pair, compute:
  - Hausdorff distance (bidirectional, canonical units)
  - Tolerance pass rate at tau in {1, 5, 10, 20}
  - Volume error (relative %)
  - F-score at tau_rel = 2% of GT bbox diagonal
  - PEM (integer-multiset match between predicted .txt and GT .py), if available

Usage:
  python compute_extrude_metrics.py <pred_dir> <gt_dir> <out_csv>
  # Example pred_dirs:
  #   0_Extude_Result/extrude_test_1500_extrude_100w           (per-sample subdirs N/N.{stl,txt})
  #   0_Extude_Result/extrude_test_1500_CADRecode_orig          (per-sample subdirs N/N.stl only)
  #   0_Extude_Result/Output_DeepCAD                            (flat: N.stl in same dir)
"""
import os, sys, json, csv, re, ast, time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

M = 8192  # surface points per mesh

def safe_load_stl(path):
    try:
        m = trimesh.load(path, force='mesh', process=False)
        if m is None or m.is_empty or len(m.vertices) == 0:
            return None
        return m
    except Exception:
        return None

def sample_surface(mesh, n=M, seed=42):
    """Uniform-area-proportional surface sampling."""
    rng = np.random.RandomState(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, sample_color=False)
    return np.asarray(pts)

def metric_pair(pred_pts, gt_pts):
    """Return dict with hausdorff, cd, f@2%, plus pred/gt one-way distances."""
    tree_g = cKDTree(gt_pts)
    tree_p = cKDTree(pred_pts)
    d_p2g, _ = tree_g.query(pred_pts, k=1)  # pred -> gt
    d_g2p, _ = tree_p.query(gt_pts, k=1)    # gt -> pred
    hd = float(max(d_p2g.max(), d_g2p.max()))
    cd = float((d_p2g**2).mean() + (d_g2p**2).mean())
    bbox_diag_gt = float(np.linalg.norm(gt_pts.max(axis=0) - gt_pts.min(axis=0)))
    tau_rel = 0.02 * bbox_diag_gt
    P_tau = float((d_p2g <= tau_rel).mean())
    R_tau = float((d_g2p <= tau_rel).mean())
    f_score = (2 * P_tau * R_tau / (P_tau + R_tau)) if (P_tau + R_tau) > 0 else 0.0
    return dict(hausdorff=hd, cd=cd, fscore_rel2pct=f_score, bbox_diag_gt=bbox_diag_gt)

def volume_error(pred_mesh, gt_mesh):
    try:
        vp, vg = float(pred_mesh.volume), float(gt_mesh.volume)
        if vg <= 0:
            return None
        return abs(vp - vg) / vg
    except Exception:
        return None

INT_LITERAL_RE = re.compile(r'(?<![A-Za-z_.])(-?\d+)(?!\.)\b')

def extract_integer_literals(code: str):
    """Return sorted list of integer literals in the source string.
    Excludes negatives that are unary minus on identifier/.attr (handled by lookbehind).
    Floats (123.0) are excluded by the negative lookahead.
    """
    matches = INT_LITERAL_RE.findall(code)
    return sorted([int(x) for x in matches])

def pem_compare(pred_txt: str, gt_py: str):
    """Return 1 if integer-multiset matches exactly, else 0. Returns None on read fail."""
    try:
        with open(pred_txt) as f:
            pcode = f.read()
        with open(gt_py) as f:
            gcode = f.read()
        pints = extract_integer_literals(pcode)
        gints = extract_integer_literals(gcode)
        return 1 if pints == gints else 0
    except Exception:
        return None

def discover_pred_path(pred_root: Path, idx: int):
    """Try several layouts: flat (N.stl) or per-sample subdir (N/N.stl)."""
    flat = pred_root / f"{idx}.stl"
    if flat.exists():
        return flat, pred_root / f"{idx}.txt" if (pred_root / f"{idx}.txt").exists() else None
    sub_stl = pred_root / str(idx) / f"{idx}.stl"
    if sub_stl.exists():
        sub_txt = pred_root / str(idx) / f"{idx}.txt"
        return sub_stl, (sub_txt if sub_txt.exists() else None)
    return None, None

def discover_gt_paths(gt_root: Path, idx: int):
    """GT layout: gt_root/<idx>/<idx>.stl and <idx>.py"""
    sub_stl = gt_root / str(idx) / f"{idx}.stl"
    sub_py = gt_root / str(idx) / f"{idx}.py"
    return (sub_stl if sub_stl.exists() else None,
            sub_py if sub_py.exists() else None)

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    pred_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])
    out_csv = Path(sys.argv[3])
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 1500

    rows = []
    t0 = time.time()
    for i in range(n_samples):
        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{n_samples}] elapsed {time.time()-t0:.1f}s", flush=True)

        pred_stl, pred_txt = discover_pred_path(pred_dir, i)
        gt_stl, gt_py = discover_gt_paths(gt_dir, i)

        row = dict(idx=i, pred_stl_exists=int(pred_stl is not None),
                   gt_stl_exists=int(gt_stl is not None))

        if pred_stl is None or gt_stl is None:
            row['exec_ok'] = 0
            rows.append(row); continue

        gt_mesh = safe_load_stl(gt_stl)
        pred_mesh = safe_load_stl(pred_stl)
        if gt_mesh is None or pred_mesh is None:
            row['exec_ok'] = 0
            rows.append(row); continue

        row['exec_ok'] = 1
        try:
            gt_pts = sample_surface(gt_mesh)
            pr_pts = sample_surface(pred_mesh)
            geom = metric_pair(pr_pts, gt_pts)
            row.update(geom)
            for tau in (1.0, 5.0, 10.0, 20.0):
                row[f'tp_tau{int(tau)}'] = int(geom['hausdorff'] <= tau)
            ve = volume_error(pred_mesh, gt_mesh)
            row['vol_err'] = ve
        except Exception as e:
            row['err'] = str(e)[:120]

        # PEM
        if pred_txt is not None and gt_py is not None:
            row['pem'] = pem_compare(pred_txt, gt_py)
        else:
            row['pem'] = None

        rows.append(row)

    # write CSV
    fields = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    # summary
    valid = [r for r in rows if r.get('exec_ok') == 1 and 'hausdorff' in r]
    n = len(rows); nv = len(valid)
    if nv == 0:
        print(f"\nNo valid predictions; wrote {out_csv}")
        return
    def med(key, default=np.nan):
        v = [r[key] for r in valid if isinstance(r.get(key), (int, float))]
        return float(np.median(v)) if v else default
    def mean(key, default=np.nan):
        v = [r[key] for r in valid if isinstance(r.get(key), (int, float))]
        return float(np.mean(v)) if v else default

    print()
    print(f"=== Summary: {pred_dir.name} ===")
    print(f"  Validity (exec_ok / total)  : {nv}/{n} = {nv/n*100:.2f}%")
    print(f"  Hausdorff median   : {med('hausdorff'):.3f}   mean: {mean('hausdorff'):.3f}")
    print(f"  TP @ tau=1   : {mean('tp_tau1')*100:.2f}%")
    print(f"  TP @ tau=5   : {mean('tp_tau5')*100:.2f}%")
    print(f"  TP @ tau=10  : {mean('tp_tau10')*100:.2f}%")
    print(f"  TP @ tau=20  : {mean('tp_tau20')*100:.2f}%")
    print(f"  VolErr median: {med('vol_err')*100 if med('vol_err')==med('vol_err') else float('nan'):.3f}%   mean: {mean('vol_err')*100:.3f}%")
    print(f"  CD median    : {med('cd'):.5f}")
    print(f"  F@2pct median: {med('fscore_rel2pct'):.4f}")
    pem_vals = [r['pem'] for r in valid if r.get('pem') is not None]
    if pem_vals:
        print(f"  PEM (n={len(pem_vals)}) : {sum(pem_vals)/len(pem_vals)*100:.2f}%")
    print(f"  CSV: {out_csv}")
    print(f"  Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
