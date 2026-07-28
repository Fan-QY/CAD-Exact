"""Aggregate 5x4 metrics into 4 per-op LaTeX tables (15 metrics x 5 methods).

Inputs (under verify_csv/run_5x4/):
  metrics_<op>_<method>.csv    surface metrics from compute_extrude_metrics.py
  mf_<op>_<method>.csv         modeling features from compute_modeling_features.py
  iou_<op>_<method>.json       voxel IoU summary

Outputs:
  table_<op>.tex               5-method 15-metric LaTeX table per op
  summary_5x4_v2.json          machine-readable cell-by-cell numbers
"""
import csv, json, math
from pathlib import Path
from statistics import median

ROOT = Path("/path/to/verify_csv/run_5x4")
OPS = ["extrude", "revolve", "sweep", "loft"]
METHODS = ["stage1", "stage2", "stage3", "stage4", "crorig"]
METHOD_LABEL = {
    "stage1": "CAD-Exact stage 1", "stage2": "CAD-Exact stage 2",
    "stage3": "CAD-Exact stage 3", "stage4": "CAD-Exact stage 4 final",
    "crorig": "CAD-Recode v1.5 (+post)",
}

def fnum(x):
    if x is None: return None
    try: v = float(x)
    except (ValueError, TypeError): return None
    return None if math.isnan(v) or math.isinf(v) else v

def load_csv(p):
    if not p.exists(): return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if v in ("", "None"):
                    row[k] = None
                else:
                    fn = fnum(v)
                    row[k] = fn if fn is not None else v
            rows.append(row)
    return rows

def aggregate_one(op, method):
    surf = load_csv(ROOT / f"metrics_{op}_{method}.csv")
    mf   = load_csv(ROOT / f"mf_{op}_{method}.csv")
    iou_p = ROOT / f"iou_{op}_{method}.json"
    iou = json.loads(iou_p.read_text()) if iou_p.exists() else {}

    n_total = len(surf) if surf else 1500
    valid = [r for r in surf if r.get("exec_ok") == 1.0]
    n_valid = len(valid)
    o = {}
    o["n_total"] = n_total; o["n_valid"] = n_valid
    o["validity"] = 100.0 * n_valid / n_total if n_total else None

    def _med(rows, key):
        vs = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return median(vs) if vs else None

    def _pct_le(rows, key, thr):
        ok = sum(1 for r in rows if isinstance(r.get(key), (int, float)) and r[key] <= thr)
        return 100.0 * ok / n_total if n_total else None

    # Tier-1
    o["tvf_1"]  = _pct_le(valid, "vol_err", 0.01)
    o["tvf_5"]  = _pct_le(valid, "vol_err", 0.05)
    o["tvf_50"] = _pct_le(valid, "vol_err", 0.50)
    pem_n = sum(1 for r in valid if r.get("pem") == 1.0)
    o["pem"]    = 100.0 * pem_n / n_total if n_total else None
    # MFM/MFS
    if mf:
        op_match  = sum(1 for r in mf if r.get("op_count_match") == 1.0)
        all_match = sum(1 for r in mf if r.get("all_features_match") == 1.0)
        bool_match = sum(1 for r in mf if r.get("bool_count_match") == 1.0)
        sk_match = sum(1 for r in mf if r.get("sk_total_match") == 1.0)
        o["mfm_op"]  = 100.0 * op_match / n_total
        o["mfm_all"] = 100.0 * all_match / n_total
        # MFS = (op + bool + sk) / 3 (3-axis composite)
        o["mfs"] = (o["mfm_op"] + 100.0*bool_match/n_total + 100.0*sk_match/n_total) / 3.0
    else:
        o["mfm_op"] = o["mfm_all"] = o["mfs"] = None

    # Tier-2
    o["cd_med"]    = _med(valid, "cd")
    o["fscore_med"] = _med(valid, "fscore_rel2pct")
    o["hd_med"]    = _med(valid, "hausdorff")
    o["iou_mean"]  = iou.get("mean")

    # VGF@5%: % all with valid AND vol_err<=5% AND fscore>=0.95
    vgf_n = sum(1 for r in valid
                if isinstance(r.get("vol_err"), (int, float)) and r["vol_err"] <= 0.05
                and isinstance(r.get("fscore_rel2pct"), (int, float)) and r["fscore_rel2pct"] >= 0.95)
    o["vgf_5"] = 100.0 * vgf_n / n_total if n_total else None

    # Tier-3
    for d, k in [(0.05, "uv_5"), (0.10, "uv_10"), (0.50, "uv_50")]:
        ok = sum(1 for r in valid if isinstance(r.get("vol_err"), (int, float)) and r["vol_err"] <= d)
        o[k] = 100.0 * ok / n_total if n_total else None
    return o

def fmt_num(v, decimals=1, percent=False):
    if v is None: return "n/a"
    if percent: return f"{v:.{decimals}f}"
    if abs(v) < 0.01 and v != 0: return f"{v:.4f}"
    if abs(v) >= 1000: return f"{int(round(v))}"
    return f"{v:.{decimals}f}"

def main():
    summary = {}
    for op in OPS:
        summary[op] = {m: aggregate_one(op, m) for m in METHODS}
    out = ROOT / "summary_5x4_v2.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")

    # Generate per-op LaTeX
    metric_rows = [
        ("\\textit{Tier-1: Precision-aware geometry}", None, None),
        ("\\quad TVF@1\\,\\% (\\%, $\\uparrow$)",  "tvf_1", 1),
        ("\\quad TVF@5\\,\\% (\\%, $\\uparrow$)",  "tvf_5", 1),
        ("\\quad TVF@50\\,\\% (\\%, $\\uparrow$)", "tvf_50", 1),
        ("\\quad MFS (\\%, $\\uparrow$)",          "mfs", 1),
        ("\\quad MFM op-count (\\%, $\\uparrow$)", "mfm_op", 1),
        ("\\quad MFM all-feat (\\%, $\\uparrow$)", "mfm_all", 1),
        ("\\quad PEM (\\%, $\\uparrow$)",          "pem", 1),
        ("\\textit{Tier-2: Shape similarity}", None, None),
        ("\\quad CD median ($\\downarrow$)",        "cd_med", 1),
        ("\\quad F@2\\% median ($\\uparrow$)",      "fscore_med", 4),
        ("\\quad Hausdorff median ($\\downarrow$)", "hd_med", 1),
        ("\\quad IoU mean ($\\uparrow$)",           "iou_mean", 3),
        ("\\quad VGF@5\\,\\% (\\%, $\\uparrow$)",   "vgf_5", 1),
        ("\\textit{Tier-3: Program validity}", None, None),
        ("\\quad Plain validity (\\%, $\\uparrow$)","validity", 1),
        ("\\quad UV@5\\,\\% (\\%, $\\uparrow$)",    "uv_5", 1),
        ("\\quad UV@10\\,\\% (\\%, $\\uparrow$)",   "uv_10", 1),
        ("\\quad UV@50\\,\\% (\\%, $\\uparrow$)",   "uv_50", 1),
    ]

    for op in OPS:
        rows_tex = []
        # Find best (max) per metric — except CD/Hausdorff (min)
        for label, key, dec in metric_rows:
            if key is None:
                rows_tex.append(f"\\multicolumn{{6}}{{l}}{{{label}}} \\\\")
                continue
            vals = [summary[op][m].get(key) for m in METHODS]
            valid_vals = [(i, v) for i, v in enumerate(vals) if v is not None]
            if not valid_vals:
                cells = ["n/a"] * 5
            else:
                if key in ("cd_med", "hd_med"):
                    best_i = min(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_i = max(valid_vals, key=lambda x: x[1])[0]
                cells = []
                for i, v in enumerate(vals):
                    if v is None:
                        cells.append("n/a")
                    else:
                        s = fmt_num(v, decimals=dec)
                        cells.append(f"\\textbf{{{s}}}" if i == best_i else s)
            rows_tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
        body = "\n".join(rows_tex)
        cap_op = op.capitalize()
        tex = f"""% Auto-generated from compute_*_metrics.py + voxel_iou (200-sample subset)
\\begin{{table}}[!ht]\\centering\\small
\\caption{{\\textbf{{Unified evaluation on the 1500-sample {op} test split.}} 5 methods, 15 metrics. \\textbf{{Bold}} marks the highest value per row (lower is better for $\\downarrow$-marked rows).}}
\\label{{tab:main-{op}}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l ccccc}}
\\toprule
\\textbf{{Metric}} & \\shortstack{{CAD-Exact\\\\stage 1}} & \\shortstack{{CAD-Exact\\\\stage 2}} & \\shortstack{{CAD-Exact\\\\stage 3}} & \\shortstack{{CAD-Exact\\\\stage 4 final}} & \\shortstack{{CAD-Recode\\\\v1.5 (+post)}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
        op_tex = ROOT / f"table_{op}.tex"
        op_tex.write_text(tex)
        print(f"Wrote {op_tex}")

    print("\n=== Compact preview ===")
    for op in OPS:
        print(f"\n--- {op} ---")
        keys = ["validity", "tvf_5", "mfm_op", "iou_mean", "uv_50"]
        print(f"{'metric':<12}" + "".join(f"{m:>10}" for m in METHODS))
        for k in keys:
            vs = [summary[op][m].get(k) for m in METHODS]
            print(f"{k:<12}" + "".join(f"{('%.2f'%v) if isinstance(v,(int,float)) else '-':>10}" for v in vs))

if __name__ == "__main__":
    main()
