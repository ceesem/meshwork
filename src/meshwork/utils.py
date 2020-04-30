import blosc
import numpy as np
DEFAULT_VOXEL_RESOLUTION = [4, 4, 40]


class InputError(Exception):
    def __init__(self, message):
        self.message = message


def unique_column_name(base_name, suffix, df):
    if base_name is not None:
        col_name = f"{base_name}_{suffix}"
    else:
        col_name = suffix
    if col_name in df.columns:
        ii = 0
        while True:
            test_col_name = f"{col_name}_{ii}"
            ii += 1
            if test_col_name not in df.columns:
                col_name = test_col_name
                break
    return col_name


def _compress_mesh_data(mesh, cname='lz4'):
    zvs = blosc.compress(mesh.vertices.tostring(), typesize=8, cname=cname)
    zfs = blosc.compress(mesh.faces.tostring(), typesize=8, cname=cname)
    zes = blosc.compress(mesh.link_edges.tostring(), typesize=8, cname=cname)
    return zvs, zfs, zes


def _decompress_mesh_data(zvs, zfs, zes):
    vs = np.frombuffer(blosc.decompress(zvs)).reshape(-1, 3)
    fs = np.frombuffer(blosc.decompress(zfs)).reshape(-1, 3)
    es = np.frombuffer(blosc.decompress(zes)).reshape(-1, 2)
    return trimesh_io.Mesh(vs, fs, link_edges=es)


class MaskedMeshMemory(object):
    def __init__(self, mesh):
        self.node_mask = mesh.node_mask.copy()
        self.map_indices_to_unmasked = mesh.map_indices_to_unmasked
        self.map_boolean_to_unmasked = mesh.map_boolean_to_unmasked
        self.filter_unmasked_boolean = mesh.filter_unmasked_boolean
        self.filter_unmasked_indices = mesh.filter_unmasked_indices
        self.filter_unmasked_indices_padded = mesh.filter_unmasked_indices_padded
        self._voxel_scaling = mesh.voxel_scaling
        self._kdtree = mesh.kdtree

    @property
    def kdtree(self):
        return self._kdtree

    @property
    def voxel_scaling(self):
        return self._voxel_scaling
