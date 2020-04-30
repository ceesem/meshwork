import blosc
from meshparty.trimesh_vtk import mesh_actor, point_cloud_actor
import pandas as pd
import numpy as np
from .utils import InputError, unique_column_name, DEFAULT_VOXEL_RESOLUTION, MaskedMeshMemory, _compress_mesh_data, _decompress_mesh_data


class AnchoredAnnotationManager(object):
    def __init__(self,
                 anchor_mesh=None,
                 filter_mesh=None,
                 voxel_resolution=None,
                 ):
        """Collection of dataframes anchored to a common mesh and filter.

        Parameters
        ----------
        mesh : trimesh_io.Mesh, optional
            Mesh to use to link points to vertex indices, by default None
        voxel_resolution : array-like, optional
            3-element array of the resolution of voxels in the point columns of the dataframes, by default None
        """
        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION
        self._voxel_resolution = np.array(voxel_resolution).reshape((1, 3))
        if anchor_mesh is None:
            self._anchor_mesh = None
        else:
            self._anchor_mesh = MaskedMeshMemory(anchor_mesh)
        self._filter_mesh = filter_mesh,
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
        "List of data table names"
        return list(self._data_tables.keys())

    @property
    def voxel_resolution(self):
        "Resolution in nm of the data in the point column"
        return self._voxel_resolution

    @voxel_resolution.setter
    def voxel_resolution(self, new_res):
        self._voxel_resolution = np.array(new_res).reshape((1, 3))
        for tn in self.table_names:
            self._data_tables[tn].voxel_resolution = self._voxel_resolution

    def update_anchor_mesh(self, new_mesh):
        "Change or add the mesh to use for proximity-linking"

        self._anchor_mesh = new_mesh

        for name in self.table_names:
            table = self._data_tables[name]
            if table.anchored is False:
                continue

            if table._index_column_base in table._original_columns:
                index_column = table._index_column_base
            else:
                index_column = None

            if table._point_column in table._original_columns:
                point_column = table._point_column
            else:
                point_column = None

            self._data_tables[name] = AnchoredAnnotation(name,
                                                         table.data_original,
                                                         self._anchor_mesh,
                                                         point_column=point_column,
                                                         max_distance=table._max_distance,
                                                         index_column=index_column,
                                                         voxel_resolution=self.voxel_resolution)
            if self._filter_mesh is not None:
                self._data_tables[name]._filter_data(self._filter_mesh)

    def add_annotations(self,
                        name,
                        data,
                        anchored=True,
                        point_column=None,
                        max_distance=np.inf,
                        index_column=None,
                        overwrite=False,
                        ):
        "Add a dataframe to the manager"

        if name in self.table_names and overwrite is False:
            raise ValueError(
                'Table name already taken. Overwrite or choose a different name.')

        self._data_tables[name] = AnchoredAnnotation(name,
                                                     data,
                                                     self._anchor_mesh,
                                                     point_column=point_column,
                                                     anchor_to_mesh=anchored,
                                                     max_distance=max_distance,
                                                     index_column=index_column,
                                                     voxel_resolution=self.voxel_resolution,
                                                     )

    def remove_annotations(self, name):
        "Remove a data table from the manager"
        if isinstance(name, str):
            name = [name]
        for n in name:
            if n in self._data_tables:
                del self._data_tables[n]

    def anchor_annotations(self, name):
        "If an annotation is not anchored, link it to the current anchor mesh and apply the current filters"
        if isinstance(name, str):
            name = [name]
        for n in name:
            self._data_tables[n]._anchor_to_mesh(self._anchor_mesh)
            if self._filter_mesh is not None:
                self._data_tables[n]._filter_data(self._filter_mesh)

    def filter_annotations(self, new_mesh):
        "Use a masked mesh to filter all anchored annotations"
        self._filter_mesh = MaskedMeshMemory(new_mesh)
        for tn in self.table_names:
            self._data_tables[tn]._filter_data(self._filter_mesh)

    def remove_filter(self):
        "Remove filters from the annotations"
        self._filter_mesh = None
        for tn in self.table_names:
            self._data_tables[tn]._reset_filter()


class AnchoredAnnotation(object):
    def __init__(self,
                 name,
                 data,
                 mesh=None,
                 anchor_to_mesh=True,
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

        self._anchored = anchor_to_mesh
        if self._anchored and mesh is not None:
            self._anchor_points(mesh)

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
        if self.anchored:
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
        if self.point_column is None or len(self.df) == 0:
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
        if self.anchored:
            return self._data[self._index_column_filt][self._is_included].values
        else:
            return None

    @property
    def _mesh_index_base(self):
        return self._data[self._index_column_base].values

    @property
    def data_original(self):
        return self._data[self._original_columns]

    def _anchor_points(self, mesh):
        dist, minds_filt = mesh.kdtree.query(self.points)
        self._data[self._index_column_filt] = minds_filt

        minds_base = mesh.map_indices_to_unmasked(minds_filt)
        self._data[self._index_column_base] = minds_base

        self._data[self._valid_column] = dist < self._max_distance

    def _filter_data(self, filter_mesh):
        """Get the subset of data points that are associated with the mesh
        """
        if self._anchored:
            self._data[self._mask_column] = filter_mesh.node_mask[self._mesh_index_base]
            self._data[self._index_column_filt] = filter_mesh.filter_unmasked_indices_padded(
                self._mesh_index_base)

    def _filter_query(self, node_mask):
        """Returns the data contained with a given filter without changing any indexing.
        """
        if self._anchored:
            keep_rows = node_mask[self._data[self._mesh_index_base]]
            return keep_rows[self._in_mask]
        else:
            return np.full(len(self.df), True)

    def _reset_filter(self):
        if self._anchored:
            self._data[self._mask_column] = True
            self._data[self._index_column_filt] = self._mesh_index_base

    def _anchor_to_mesh(self, anchor_mesh):
        self._anchored = True
        self._reset_filter()
        self._data[self._valid_column] = True
        self._anchor_points(anchor_mesh)

    @property
    def anchored(self):
        return self._anchored


class Meshwork(object):
    def __init__(self, mesh, seg_id=None, voxel_resolution=None):
        self._seg_id = seg_id
        self._mesh = mesh

        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION
        self._anno = AnchoredAnnotationManager(
            self._mesh, voxel_resolution=voxel_resolution)

        self._mesh_mask = mesh.node_mask
        self._original_mesh_data = _compress_mesh_data(mesh)

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
        self._anno.filter_annotations(self._mesh)

    def reset_mesh(self):
        self._mesh = _decompress_mesh_data(*self._original_mesh_data)
        self._mesh_mask = np.full(self._mesh.n_vertices, True)
        self._anno.remove_filter()

    @property
    def anno(self):
        return self._anno

    def add_annotations(self,
                        name,
                        data,
                        anchored=True,
                        point_column=None,
                        max_distance=np.inf,
                        index_column=None,
                        overwrite=False,
                        ):
        self._anno.add_annotations(
            name, data, anchored, point_column, max_distance, index_column, overwrite)

    def remove_annotations(self, name):
        self._anno.remove_annotations(name)

    def update_anchors(self):
        self._anno.update_anchor_mesh(self.mesh)

    def anchor_annotations(self, name):
        self._anno.anchor_annotations(name)

    def mesh_actor(self, **kwargs):
        return mesh_actor(self.mesh, **kwargs)

    def anno_point_actor(self, anno_name, **kwargs):
        return point_cloud_actor(self.anno[anno_name].points, **kwargs)
