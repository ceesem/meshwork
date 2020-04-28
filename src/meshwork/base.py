from meshparty import trimesh_io, skeleton_io, skeleton
import pandas as pd
import numpy as np
from .utils import InputError, unique_column_name, DEFAULT_VOXEL_RESOLUTION


class MeshLinkedData(object):
    def __init__(self,
                 name,
                 data,
                 mesh,
                 point_column=None,
                 max_distance=np.inf,
                 index_column=None,
                 valid_column=None,
                 voxel_resolution=None,
                 ):
        self._name = name
        self._data = data
        self._original_columns = data.columns
        if point_column is None and index_column is None:
            raise InputError(
                "Either a point column or an index column must be specified")

        self._point_column = point_column
        if index_column is None:
            index_column = unique_column_name(point_column, 'mesh_index', data)
        self._index_column = index_column

        if valid_column is None:
            valid_column = unique_column_name(index_column, 'valid', data)
        self._valid_column = valid_column

        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION
        if mesh.voxel_scaling is not None:
            voxel_resolution = voxel_resolution * mesh.voxel_scaling

        self._voxel_resolution = voxel_resolution
        self._attach_points(mesh, max_distance, voxel_resolution)

    @property
    def index_column(self):
        return self._index_column

    @property
    def point_column(self):
        return self._point_column

    @property
    def valid_column(self):
        return self._valid_column

    @property
    def data(self):
        return self._data[self._original_columns]

    @property
    def data_all(self):
        return self._data

    @property
    def name(self):
        return self._name

    @property
    def mesh_indices(self):
        return self._data[self.index_column]

    @property
    def valid_indices(self):
        return self._data[self.valid_column]

    @property
    def voxel_resolution(self):
        return self._voxel_resolution

    @voxel_resolution.setter
    def voxel_resolution(self, new_res):
        self._voxel_resolution = np.array(new_res).reshape((1, 3))
        print(self._voxel_resolution)

    @property
    def voxels(self):
        if self.point_column is None:
            return None
        else:
            return np.vstack(self.data[self.point_column].values)

    @property
    def points(self):
        if self.point_column is None:
            return None
        else:
            return self.voxels * self.voxel_resolution

    def _attach_points(self, mesh, max_distance, voxel_resolution):
        dist, minds_basic = mesh.kdtree.query(self.points)
        minds = mesh.map_indices_to_unmasked(minds_basic)
        self._data[self.index_column] = minds

        if self.valid_column not in self.data.columns:
            self._data[self.valid_column] = dist < max_distance
        else:
            self._data[self.valid_column] = np.logical_or(
                dist < max_distance, self._data[self.valid_column])


class Meshwork(object):
    def __init__(self, seg_id=None, mesh=None, skeleton=None, annotations=None):
        self._seg_id = seg_id
        self._original_mesh = mesh
        self._skeleton = skeleton

    @property
    def seg_id(self):
        return self._seg_id
