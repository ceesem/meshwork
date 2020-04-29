import blosc
from meshparty import trimesh_io, skeleton_io, skeleton
import pandas as pd
import numpy as np
from .utils import InputError, unique_column_name, DEFAULT_VOXEL_RESOLUTION


class MaskedMeshMemory(object):
    def __init__(self, mesh):
        self.node_mask = mesh.node_mask
        self.map_indices_to_unmasked = mesh.map_indices_to_unmasked
        self.map_boolean_to_unmasked = mesh.map_boolean_to_unmasked
        self.filter_unmasked_boolean = mesh.filter_unmasked_boolean
        self.filter_unmasked_indices = mesh.filter_unmasked_indices
        self.filter_unmasked_indices_padded = mesh.filter_unmasked_indices_padded


class LinkedAnnotationHolder(object):
    def __init__(self,
                 mesh=None,
                 voxel_resolution=None,
                 ):
        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION
        self._voxel_resolution = np.array(voxel_resolution).reshape((1, 3))
        self._link_mesh = mesh
        self._filter_mesh = None
        self._data_tables = {}

    def __getitem__(self, key):
        return self._data_tables[key]

    def __repr__(self):
        return f'Data tables: {self.table_names}'

    def __len__(self):
        return len(self._data_tables)

    def items(self):
        return self._data_tables.items()

    @property
    def table_names(self):
        return list(self._data_tables.keys())

    @property
    def voxel_resolution(self):
        return self._voxel_resolution

    @voxel_resolution.setter
    def voxel_resolution(self, new_res):
        self._voxel_resolution = np.array(new_res).reshape((1, 3))
        for tn in self.table_names:
            self._data_tables[tn].voxel_resolution = self._voxel_resolution

    def add_annotations(self,
                        name,
                        data,
                        link=True,
                        point_column=None,
                        max_distance=np.inf,
                        index_column=None,
                        overwrite=False,
                        ):
        if name in self.table_names and overwrite is False:
            raise ValueError(
                'Table name already taken. Overwrite or choose a different name.')

        self._data_tables[name] = MeshLinkedAnnotation(name,
                                                       data,
                                                       self._link_mesh,
                                                       point_column=point_column,
                                                       link_to_mesh=link,
                                                       max_distance=max_distance,
                                                       index_column=index_column,
                                                       voxel_resolution=self.voxel_resolution,
                                                       )

    def filter_annotations(self, new_mesh):
        self._filter_mesh = MaskedMeshMemory(new_mesh)
        for tn in self.table_names:
            self._data_tables[tn]._filter_data(self._filter_mesh)

    def reset_filters(self):
        self._filter_mesh = None
        for tn in self.table_names:
            self._data_tables[tn]._reset_data()

    def remove_annotations(self, name):
        self._data_tables.pop(name, None)

    def link_annotation(self, name):
        self._data_tables[name]._link_to_mesh(self._link_mesh)
        if self._filter_mesh is not None:
            self._data_tables[name]._filter_data(self._filter_mesh)

    def update_linked_mesh(self, new_mesh):
        self._link_mesh = new_mesh

        for name in self.table_names:
            table = self._data_tables[name]
            if table._linked is False:
                continue

            if table._index_column_base in table._original_columns:
                index_column = table._index_column_base
            else:
                index_column = None

            if table._point_column in table._original_columns:
                point_column = table._point_column
            else:
                point_column = None

            self._data_tables[name] = MeshLinkedAnnotation(name,
                                                           table.data_original,
                                                           self._link_mesh,
                                                           point_column=point_column,
                                                           max_distance=table._max_distance,
                                                           index_column=index_column,
                                                           voxel_resolution=self.voxel_resolution)
            if self._filter_mesh is not None:
                self._data_tables[name]._filter_data(self._filter_mesh)


class MeshLinkedAnnotation(object):
    def __init__(self,
                 name,
                 data,
                 mesh=None,
                 link_to_mesh=True,
                 point_column=None,
                 max_distance=np.inf,
                 index_column=None,
                 voxel_resolution=None,
                 ):

        self._name = name
        self._data = data.reset_index()
        self._original_columns = data.columns
        self._max_distance = max_distance

        self._point_column = point_column
        if index_column is None:
            index_column = unique_column_name(
                None, 'mesh_index_base', data)
        self._index_column_base = index_column
        self._index_column_filt = unique_column_name(
            None, 'mesh_index', data)

        self._orig_col_plus_index = list(
            self._original_columns) + [self._index_column_filt]
        # Initalize to -1 so the column exists
        self._data[self._index_column_base] = -1
        self._data[self._index_column_filt] = -1

        valid_column = unique_column_name(index_column, 'valid', data)
        self._data[valid_column] = True
        self._valid_column = valid_column

        self._mask_column = unique_column_name(
            index_column, 'in_mask', data)
        # Initalize in_mask to True before any subsequent masking
        self._data[self._mask_column] = True

        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION
        if mesh.voxel_scaling is not None:
            voxel_resolution = voxel_resolution * mesh.voxel_scaling

        self._voxel_resolution = np.array(voxel_resolution).reshape((1, 3))

        self._linked = link_to_mesh
        if self._linked and mesh is not None:
            self._attach_points(mesh)

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    def __getitem__(self, key):
        return self.df.__getitem__(key)

    def __len__(self):
        return len(self.df)

    @property
    def name(self):
        return self._name

    @property
    def point_column(self):
        return self._point_column

    @property
    def index_column(self):
        return self._index_column_filt

    @property
    def _is_valid(self):
        return self._data[self._valid_column]

    @property
    def _in_mask(self):
        return self._data[self._mask_column]

    @property
    def _is_included(self):
        return np.logical_and(self._is_valid, self._in_mask)

    @property
    def df(self):
        if self._linked:
            return self._data[self._orig_col_plus_index][self._is_included]
        else:
            return self._data[self._original_columns][self._is_included]

    @property
    def voxel_resolution(self):
        return self._voxel_resolution

    @voxel_resolution.setter
    def voxel_resolution(self, new_res):
        self._voxel_resolution = np.array(new_res).reshape((1, 3))

    @property
    def voxels(self):
        if self.point_column is None:
            return np.zeros((0, 3))
        else:
            return np.vstack(self.df[self.point_column].values)

    @property
    def points(self):
        if self.point_column is None:
            return np.zeros((0, 3))
        else:
            return self.voxels * self.voxel_resolution

    @property
    def mesh_index(self):
        return self._data[self._index_column_filt][self._is_included].values

    @property
    def _mesh_index_base(self):
        return self._data[self._index_column_base].values

    @property
    def data_original(self):
        return self._data[self._original_columns]

    def _attach_points(self, mesh):
        dist, minds_filt = mesh.kdtree.query(self.points)
        self._data[self._index_column_filt] = minds_filt

        minds_base = mesh.map_indices_to_unmasked(minds_filt)
        self._data[self._index_column_base] = minds_base

        self._data[self._valid_column] = np.logical_and(
            dist < self._max_distance, self._is_valid)

    def _filter_data(self, new_mesh):
        """Get the subset of data points that are associated with the mesh
        """
        if self._linked:
            self._data[self._mask_column] = new_mesh.node_mask[self._mesh_index_base]
            self._data[self._index_column_filt] = new_mesh.filter_unmasked_indices_padded(
                self._mesh_index_base)

    def _reset_data(self):
        if self._linked:
            self._data[self._mask_column] = True
            self._data[self._index_column_filt] = self._mesh_index_base

    def _link_to_mesh(self, mesh):
        self._linked = True
        self._attach_points(mesh)

    @property
    def linked(self):
        return self._linked


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


class Meshwork(object):
    def __init__(self, mesh=None, seg_id=None, voxel_resolution=None):
        self._seg_id = seg_id
        self._mesh = mesh
        self._original_mesh_data = _compress_mesh_data(mesh)

        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION

        self._mesh_mask = np.full(self._mesh.n_vertices, True)

    @property
    def seg_id(self):
        return self._seg_id

    @property
    def mesh(self):
        return self._mesh

    @property
    def mesh_mask(self):
        return self._mesh_mask

    def apply_mask(self, mask):
        self._mesh = self._mesh.apply_mask(mask)
        self._mesh_mask = self.mesh.node_mask
