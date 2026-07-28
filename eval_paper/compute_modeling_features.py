"""Modeling-feature match rate (Metric A): count CAD elements in predicted .txt vs GT .py.

For each (pred_code, gt_code) pair, count regex matches for:
  - .extrude(, .revolve(, .sweep(, .loft(, .cylinder(, .box(, .sphere(
  - .union(, .cut(, .intersect(
  - .segment(, .arc(, .lineTo(, .threePointArc(, .radiusArc(, .circle(
  - .sketch(, .workplane(, .moveTo(, .close(

Output per-sample CSV + summary of % samples whose feature counts EXACTLY match GT.
"""
import os, sys, re, csv
from pathlib import Path

FEATURES = {
    'op_extrude': r'\.extrude\(',
    'op_revolve': r'\.revolve\(',
    'op_sweep':   r'\.sweep\(',
    'op_loft':    r'\.loft\(',
    'op_cylinder':r'\.cylinder\(',
    'op_box':     r'\.box\(',
    'bool_union': r'\.union\(',
    'bool_cut':   r'\.cut\(',
    'sk_segment': r'\.segment\(',
    'sk_arc':     r'\.arc\(',
    'sk_lineTo':  r'\.lineTo\(',
    'sk_circle':  r'\.circle\(',
    'sk_close':   r'\.close\(',
    'sk_sketch':  r'\.sketch\(',
    'workplane':  r'\.[Ww]orkplane\(|cq\.Workplane\(',
    'moveTo':     r'\.moveTo\(',
}

def feature_counts(code: str):
    return {k: len(re.findall(p, code)) for k, p in FEATURES.items()}

def discover(pred_root, gt_root, idx):
    pred_txt = pred_root / str(idx) / f"{idx}.txt"
    if not pred_txt.exists():
        pred_txt = pred_root / f"{idx}.txt"
    gt_py = gt_root / str(idx) / f"{idx}.py"
    return (pred_txt if pred_txt.exists() else None,
            gt_py if gt_py.exists() else None)

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    pred_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])
    out_csv = Path(sys.argv[3])
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 1500

    rows = []
    feature_match_counts = {k: 0 for k in FEATURES}
    n_compared = 0

    for i in range(n_samples):
        pt, gp = discover(pred_dir, gt_dir, i)
        row = {'idx': i, 'has_pred': int(pt is not None), 'has_gt': int(gp is not None)}
        if pt is None or gp is None:
            rows.append(row); continue
        try:
            with open(pt) as f: pcode = f.read()
            with open(gp) as f: gcode = f.read()
        except Exception:
            rows.append(row); continue

        pc = feature_counts(pcode)
        gc = feature_counts(gcode)

        n_compared += 1
        for k in FEATURES:
            row[f'pred_{k}'] = pc[k]
            row[f'gt_{k}'] = gc[k]
            row[f'match_{k}'] = int(pc[k] == gc[k])
            if pc[k] == gc[k]:
                feature_match_counts[k] += 1

        # summary scores per sample
        all_match = all(pc[k] == gc[k] for k in FEATURES)
        any_op_match = (pc['op_extrude'] == gc['op_extrude']) and (pc['op_revolve'] == gc['op_revolve']) and \
                       (pc['op_sweep'] == gc['op_sweep']) and (pc['op_loft'] == gc['op_loft'])
        bool_match = (pc['bool_union'] == gc['bool_union']) and (pc['bool_cut'] == gc['bool_cut'])
        sk_total_pred = sum(pc[k] for k in ['sk_segment','sk_arc','sk_lineTo','sk_circle'])
        sk_total_gt = sum(gc[k] for k in ['sk_segment','sk_arc','sk_lineTo','sk_circle'])
        row['all_features_match'] = int(all_match)
        row['op_count_match'] = int(any_op_match)
        row['bool_count_match'] = int(bool_match)
        row['sk_total_diff'] = abs(sk_total_pred - sk_total_gt)
        row['sk_total_match'] = int(sk_total_pred == sk_total_gt)

        rows.append(row)

    # CSV
    fields = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, '') for k in fields})

    if n_compared == 0:
        print(f"No comparable pairs in {pred_dir.name} (likely no .txt files)")
        return

    print(f"\n=== Modeling-feature match: {pred_dir.name} ===")
    print(f"  Comparable samples: {n_compared}/{n_samples}")
    print(f"  All-features-match:   {sum(r.get('all_features_match',0) for r in rows)/n_compared*100:.2f}%")
    print(f"  Op-count match (4 ops):    {sum(r.get('op_count_match',0) for r in rows)/n_compared*100:.2f}%")
    print(f"  Bool-count match (union/cut): {sum(r.get('bool_count_match',0) for r in rows)/n_compared*100:.2f}%")
    print(f"  Sketch-primitive total match: {sum(r.get('sk_total_match',0) for r in rows)/n_compared*100:.2f}%")
    print(f"  Per-feature match rates:")
    for k in FEATURES:
        print(f"    {k:14s}: {feature_match_counts[k]/n_compared*100:6.2f}%")

if __name__ == "__main__":
    main()
