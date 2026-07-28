"""Compute voxel IoU (200-sample subset, res=16) for all (op, method) combos
   with at least 100 non-empty STLs. Output: verify_csv/run_5x4/iou_<op>_<method>.json"""
import json, sys, time, os
import numpy as np
import trimesh
from pathlib import Path
from statistics import median

ROOT = Path("/path/to/workspace")
GT_BY_OP = {
    "extrude": ROOT/"CADLLM/SOURCE/test/extrude_test_1500",
    "revolve": ROOT/"CADLLM/SOURCE/test/revolve_test_1500",
    "sweep":   ROOT/"CADLLM/SOURCE/test/sweep_test_1500",
    "loft":    ROOT/"CADLLM/SOURCE/test/loft_test_1500",
}
RES_ROOT = {
    "extrude": ROOT/"0_Results/0_Extude_Result",
    "revolve": ROOT/"0_Results/1_Revolve_Result",
    "sweep":   ROOT/"0_Results/2_Sweep_Result",
    "loft":    ROOT/"0_Results/3_Loft_Result",
}
SUFFIX = {
    "stage1": "extrude_100w",
    "stage2": "extrude_revolve_100w",
    "stage3": "extrude_revolve_sweep_100w",
    "stage4": "extrude_revolve_sweep_loft_100w",
    "crorig": "orig",  # extrude special-cased below
}
OUT_DIR = ROOT/"CADLLM/verify_csv/run_5x4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pred_dir(op, method):
    suf = SUFFIX[method]
    if op == "extrude" and method == "crorig":
        suf = "CADRecode_orig"
    return RES_ROOT[op] / f"{op}_test_1500_{suf}"

def voxel_iou(pred_path, gt_path, res=16):
    try:
        m_p = trimesh.load(str(pred_path), force='mesh', process=False)
        m_g = trimesh.load(str(gt_path), force='mesh', process=False)
        if m_p is None or m_g is None or len(m_p.vertices)==0 or len(m_g.vertices)==0:
            return None
        bb_min = np.minimum(m_p.bounds[0], m_g.bounds[0])
        bb_max = np.maximum(m_p.bounds[1], m_g.bounds[1])
        if (bb_max - bb_min).max() <= 0: return None
        rng = np.random.RandomState(42)
        pts = rng.uniform(bb_min, bb_max, size=(res**3, 3))
        in_p = m_p.contains(pts)
        in_g = m_g.contains(pts)
        inter = int(np.logical_and(in_p, in_g).sum())
        union = int(np.logical_or(in_p, in_g).sum())
        return float(inter)/float(union) if union>0 else None
    except Exception:
        return None

def count_real_stls(d):
    if not d.exists(): return 0
    n=0
    for sub in d.iterdir():
        if sub.is_dir():
            stl = sub/f"{sub.name}.stl"
            if stl.exists() and stl.stat().st_size > 0:
                n+=1
    return n

def run_combo(op, method, N_max=200, threshold=100):
    d = pred_dir(op, method)
    n_real = count_real_stls(d)
    out_path = OUT_DIR / f"iou_{op}_{method}.json"
    if n_real < threshold:
        out = {"op":op, "method":method, "n_real_stls":n_real, "n_iou":0,
               "skipped":True, "reason": f"only {n_real} non-empty STLs (<{threshold})"}
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[{op}/{method}] SKIP ({n_real} real STLs)", flush=True)
        return

    print(f"[{op}/{method}] N_real={n_real}, sampling {N_max}...", flush=True)
    gt_root = GT_BY_OP[op]
    ious = []
    t0 = time.time()
    for i in range(N_max):
        pred = d / str(i) / f"{i}.stl"
        if not pred.exists() or pred.stat().st_size == 0:
            continue
        gt = gt_root / str(i) / f"{i}.stl"
        if not gt.exists():
            continue
        iou = voxel_iou(pred, gt)
        if iou is not None:
            ious.append(iou)
        if (i+1) % 50 == 0:
            print(f"  [{op}/{method}] {i+1}/{N_max} t={time.time()-t0:.0f}s n={len(ious)}", flush=True)
    out = {"op":op, "method":method, "n_real_stls":n_real, "n_iou":len(ious),
           "mean": float(np.mean(ious)) if ious else None,
           "median": float(median(ious)) if ious else None,
           "skipped": False, "elapsed_sec": time.time()-t0}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{op}/{method}] mean={out['mean']:.4f if out['mean'] else 0} median={out['median'] or 0} n={out['n_iou']} t={out['elapsed_sec']:.0f}s", flush=True)

def main():
    for op in ["extrude", "revolve", "sweep", "loft"]:
        for method in ["stage1", "stage2", "stage3", "stage4", "crorig"]:
            try:
                run_combo(op, method)
            except Exception as e:
                print(f"  ERR [{op}/{method}]: {e}", flush=True)
    print("ALL DONE")

if __name__ == "__main__":
    main()
