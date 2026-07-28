"""
extract_test_from_parquet.py
============================

从 GeomCAD parquet 里抽 test split, 落回 per-sample 目录 + JSON 索引,
好让老 infer_batched.py + eval.py 无改动地用.

对每个 op:
  {out}/{op}/<id>/<id>.py                  CadQuery 源码
  {out}/{op}/<id>/<id>.stl                 对应 mesh
  {out}/{op}/test_mesh_paths.json          {"<id>": ".../<id>.stl", ...}
  {out}/{op}/test_text_paths.json          {"<id>": ".../<id>.py",  ...}

多 GPU 推理时可额外分片:
  {out}/{op}/shard0/test_mesh_paths.json ...
  {out}/{op}/shard1/test_mesh_paths.json ...

Usage:
  python extract_test_from_parquet.py \
      --parquet_root /scratch/$USER/GeomCAD \
      --ops extrude,loft,revolve,sweep \
      --out /scratch/$USER/test_extracted \
      --shards 2
"""
import argparse
import json
import sys
from pathlib import Path

# pyarrow, not pandas: the container ships the former and not the latter, and
# reading a parquet file is all this needs.
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Reuse the trainer's decoder rather than reimplementing the byte layout. The
# ground-truth mesh an evaluation compares against must be byte-identical to the
# one training sampled points from; two copies of "float32, reshape(-1, 3)"
# would be one edit away from disagreeing.
from local.parquet_io import decode_mesh

# The packed schema. The previous version of this script guessed at column names
# ('code' or 'py', 'stl' or 'mesh'), found none of them, and silently skipped
# every row -- it wrote empty index files and exited 0.
COLUMNS = ['id', 'program', 'mesh_vertices', 'mesh_faces']


def _extract_op(parquet_root: Path, op: str, out_root: Path, shards: int):
    op_dir = parquet_root / 'data' / op
    test_files = sorted(op_dir.glob('test-*.parquet'))
    if not test_files:
        raise SystemExit(f'no test parquet for op {op} in {op_dir}')

    op_out = out_root / op
    op_out.mkdir(parents=True, exist_ok=True)

    ids = []
    for pf in test_files:
        table = pq.read_table(pf, columns=COLUMNS)
        cols = {c: table.column(c).to_pylist() for c in COLUMNS}
        for i in range(table.num_rows):
            idx = str(cols['id'][i])
            mesh = decode_mesh(cols['mesh_vertices'][i], cols['mesh_faces'][i])
            sample_dir = op_out / idx
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / f'{idx}.py').write_text(cols['program'][i])
            mesh.export(sample_dir / f'{idx}.stl')
            ids.append(idx)

    if not ids:
        raise SystemExit(f'{op}: extracted 0 samples -- refusing to write an empty index')
    print(f'[extract] {op}: {len(ids):,} samples')

    mesh_map = {i: str(op_out / i / f'{i}.stl') for i in ids}
    text_map = {i: str(op_out / i / f'{i}.py') for i in ids}
    (op_out / 'test_mesh_paths.json').write_text(json.dumps(mesh_map))
    (op_out / 'test_text_paths.json').write_text(json.dumps(text_map))

    # 分片 (为多 GPU 并行推理)
    if shards > 1:
        for s in range(shards):
            shard_ids = ids[s::shards]
            shard_dir = op_out / f'shard{s}'
            shard_dir.mkdir(exist_ok=True)
            (shard_dir / 'test_mesh_paths.json').write_text(
                json.dumps({i: mesh_map[i] for i in shard_ids}))
            (shard_dir / 'test_text_paths.json').write_text(
                json.dumps({i: text_map[i] for i in shard_ids}))
            print(f'[extract]   shard{s}: {len(shard_ids):,}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet_root', required=True, help='GeomCAD root dir')
    ap.add_argument('--ops', default='extrude,loft,revolve,sweep',
                    help='comma-separated ops to extract')
    ap.add_argument('--out', required=True, help='output root')
    ap.add_argument('--shards', type=int, default=1,
                    help='额外为每 op 生成 N 个 shard 供多 GPU 推理')
    args = ap.parse_args()

    root = Path(args.parquet_root)
    out = Path(args.out)
    for op in args.ops.split(','):
        _extract_op(root, op.strip(), out, args.shards)

    print(f'[extract] done. root: {out}')


if __name__ == '__main__':
    main()
