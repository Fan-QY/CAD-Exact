"""
Reading side of the packed GeomCAD parquet corpus.

Only the decode step is kept in this release; the streaming training dataset that
lived alongside it is not part of the inference path.
"""

import numpy as np
import trimesh


def decode_mesh(vertices_blob: bytes, faces_blob: bytes) -> trimesh.Trimesh:
    """Rebuild the mesh from the indexed representation stored in the parquet.

    v[f] reproduces the original STL triangle array byte for byte -- the packer
    deduplicated vertices on exact float32 bit patterns, with no welding
    tolerance, and STL coordinates are float32 to begin with.
    """
    v = np.frombuffer(vertices_blob, dtype=np.float32).reshape(-1, 3)
    f = np.frombuffer(faces_blob, dtype=np.int32).reshape(-1, 3)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)
