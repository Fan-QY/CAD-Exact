"""Check that two containers turn the same CAD program into the same mesh.

cadquery runs in exactly one place -- infer_batched.py executes a predicted
program and tessellates it into the STL every metric is then computed from. So
the geometry kernel does not affect ground truth (that comes straight out of the
parquet), but it does affect every prediction. If two arms of the table were
meshed by different OCCT bindings, a difference in their scores could come from
the kernel rather than the model.

The old image carries cadquery 2.5.0.dev0 / cadquery-ocp 7.7.2, an unreleased
build that cannot be pinned exactly, so equality has to be measured rather than
assumed. This reproduces infer_batched.py's execution path exactly and prints one
hash per program:

    singularity exec -B /scratch:/scratch OLD.sif python container/compare_cadquery.py \
        --parquet_root /scratch/$USER/GeomCAD --n 200 > /tmp/old.txt
    singularity exec -B /scratch:/scratch NEW.sif python container/compare_cadquery.py \
        --parquet_root /scratch/$USER/GeomCAD --n 200 > /tmp/new.txt
    diff /tmp/old.txt /tmp/new.txt && echo "geometry identical"

Any diff means: train wherever you like, but run inference in one image only.
"""
import argparse
import hashlib
import sys
from typing import Any, Dict

import numpy as np
import pyarrow.parquet as pq
import trimesh

import cadquery as cq  # noqa: F401  -- programs reference it as `cq`


def mesh_hash(py_string: str) -> str:
    """Execute one program the way infer_batched.py does and hash the result.

    Vertices are rounded to 1e-6 before hashing: a bit-exact float comparison
    would flag differences far below the tolerance any metric here can see, and
    the point is to catch a changed tessellation, not changed rounding.
    """
    local_ns: Dict[str, Any] = {}
    exec(py_string, globals(), local_ns)
    compound = local_ns["r"].val()
    vertices, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
    v = np.round(np.asarray(mesh.vertices, dtype=np.float64), 6)
    f = np.asarray(mesh.faces, dtype=np.int64)
    h = hashlib.sha256()
    h.update(v.tobytes())
    h.update(f.tobytes())
    return f"{h.hexdigest()[:24]} nv={len(v)} nf={len(f)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_root", required=True)
    ap.add_argument("--ops", default="extrude,loft,revolve,sweep")
    ap.add_argument("--n", type=int, default=100, help="programs per operation")
    args = ap.parse_args()

    import cadquery
    print(f"# cadquery {getattr(cadquery, '__version__', '?')}", file=sys.stderr)

    for op in args.ops.split(","):
        op = op.strip()
        path = f"{args.parquet_root}/data/{op}/test-00000-of-00001.parquet"
        cols = pq.read_table(path, columns=["id", "program"]).to_pydict()
        for i in range(min(args.n, len(cols["id"]))):
            idx = cols["id"][i]
            try:
                print(f"{op} {idx} {mesh_hash(cols['program'][i])}")
            except Exception as e:
                # A program that fails in both images is fine; one that fails in
                # only one is exactly the drift this is looking for, so the
                # failure text is part of the compared output.
                print(f"{op} {idx} ERROR {type(e).__name__}: {str(e)[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
