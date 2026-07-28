#!/usr/bin/env python3
"""
Qualitative grid: predicted mesh beside ground truth, for the four case types
Reviewer 1 asks about — successes, valid-but-inaccurate, invalid programs, and
operation-specific failures.

Cases are chosen by Hausdorff quantile rather than by eye, so the figure cannot be
accused of showing only flattering examples; the selection rule and the metric of
each panel are printed on the figure itself.

Rendering goes through matplotlib's 3D triangle plotting rather than an offscreen
GL context: trimesh's viewer wants pyglet or OSMesa, neither of which is reliably
available in the container, and the meshes here are small (median ~500 triangles)
so the slower path costs seconds.

  singularity exec --no-home -B <ws>:/ws CADLLM2c.sif \
      python /ws/CAD-Exact-HPC/eval_paper/render_qualitative.py \
      --metrics /ws/results_rebuttal/eval/metrics_extrude_stage1new.csv \
      --gt /ws/testsets_indexed/extrude \
      --pred /ws/results_rebuttal/predictions_indexed/stage1_extrude__test_extrude \
      --op extrude --out /ws/results_rebuttal/qualitative
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def as_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def pick_cases(rows: list[dict], n_each: int) -> dict[str, list[dict]]:
    """Quantile bands across the whole valid population, plus failures.

    Showing p5 / p50 / p95 rather than a "good" and a "bad" bucket keeps the figure
    from being a curated selection: the reader sees where the distribution actually
    sits, including its tail.
    """
    valid = [r for r in rows if as_float(r.get("exec_ok")) == 1.0
             and np.isfinite(as_float(r.get("hausdorff")))]
    valid.sort(key=lambda r: as_float(r["hausdorff"]))
    invalid = [r for r in rows if as_float(r.get("exec_ok")) != 1.0]
    if not valid:
        return {"invalid": invalid[:n_each]}

    def band(q):
        i = int(q * (len(valid) - 1))
        lo = max(0, i - n_each // 2)
        return valid[lo: lo + n_each]

    return {
        "p05": band(0.05),
        "p50": band(0.50),
        "p95": band(0.95),
        "invalid": invalid[:n_each],
    }


def draw_mesh(ax, mesh: trimesh.Trimesh | None, title: str, colour: str) -> None:
    ax.set_axis_off()
    ax.set_title(title, fontsize=8, pad=2)
    if mesh is None or len(mesh.faces) == 0:
        ax.text2D(0.5, 0.5, "no mesh", ha="center", va="center",
                  transform=ax.transAxes, fontsize=8, color="0.5")
        return
    v = mesh.vertices
    coll = Poly3DCollection(v[mesh.faces], alpha=0.9, linewidths=0.05)
    coll.set_facecolor(colour)
    coll.set_edgecolor("0.35")
    ax.add_collection3d(coll)
    c = (v.min(0) + v.max(0)) / 2
    r = float(np.abs(v - c).max()) or 1.0
    for setter, ci in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(c[ci] - r, c[ci] + r)
    ax.view_init(elev=22, azim=135)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:      # older matplotlib
        pass


def load(path: Path) -> trimesh.Trimesh | None:
    if not path.exists():
        return None
    try:
        m = trimesh.load(path, force="mesh")
        return m if len(getattr(m, "faces", [])) else None
    except Exception:
        return None


def render(cases: dict[str, list[dict]], gt_root: Path, pred_root: Path,
           op: str, out: Path) -> None:
    labels = {
        "p05": "5th percentile Hausdorff",
        "p50": "median Hausdorff",
        "p95": "95th percentile Hausdorff",
        "invalid": "program did not execute",
    }
    rows = [(k, v) for k, v in cases.items() if v]
    if not rows:
        print(f"{op}: nothing to render")
        return

    n_col = max(len(v) for _, v in rows)
    fig = plt.figure(figsize=(2.4 * n_col * 2, 2.6 * len(rows)))
    fig.suptitle(
        f"{op}: prediction (right) vs ground truth (left).  "
        f"Cases picked by Hausdorff quantile, not by inspection.",
        fontsize=11, y=0.995)

    for r, (kind, items) in enumerate(rows):
        for c, row in enumerate(items):
            idx = row["idx"]
            gt = load(gt_root / idx / f"{idx}.stl")
            pr = load(pred_root / idx / f"{idx}.stl")
            hd = as_float(row.get("hausdorff"))
            note = f"HD {hd:.1f}" if np.isfinite(hd) else "invalid"

            for j, (mesh, name, colour) in enumerate(
                    ((gt, "GT", "#8fb8de"), (pr, "pred", "#e8a87c"))):
                pos = r * n_col * 2 + c * 2 + j + 1
                if j == 1 and mesh is None:
                    # An empty 3D axis says nothing; the program that failed does.
                    ax = fig.add_subplot(len(rows), n_col * 2, pos)
                    ax.set_axis_off()
                    ax.set_title(f"#{idx} pred  (did not execute)", fontsize=8, pad=2)
                    txt = (pred_root / idx / f"{idx}.txt")
                    raw = txt.read_text() if txt.exists() else "(no program)"
                    # matplotlib's wrap= overlaps long unbroken lines; wrap explicitly
                    # and cap the height so the panel stays readable.
                    lines = []
                    for ln in raw.splitlines():
                        lines.extend(textwrap.wrap(ln, 58) or [""])
                    body = "\n".join(lines[:14])
                    if len(lines) > 14:
                        body += "\n..."
                    ax.text(0.0, 0.98, body, transform=ax.transAxes, fontsize=4.4,
                            family="monospace", va="top", ha="left", linespacing=1.35)
                    continue
                ax = fig.add_subplot(len(rows), n_col * 2, pos, projection="3d")
                head = f"#{idx} {name}" + (f"  {note}" if j == 1 else "")
                draw_mesh(ax, mesh, head, colour)
                if c == 0 and j == 0:
                    ax.text2D(-0.18, 0.5, labels[kind], transform=ax.transAxes,
                              rotation=90, va="center", ha="center", fontsize=9)

    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"qualitative_{op}.png"
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.01,
                        wspace=0.02, hspace=0.10)
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4, help="cases per category")
    args = ap.parse_args()

    rows = load_rows(Path(args.metrics))
    cases = pick_cases(rows, args.n)
    for k, v in cases.items():
        print(f"  {k}: {len(v)} cases")
    render(cases, Path(args.gt), Path(args.pred), args.op, Path(args.out))


if __name__ == "__main__":
    main()
