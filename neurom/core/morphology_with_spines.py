import pandas
import h5py
import morphio
import trimesh

from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

# Columns of edge table dataframes
_C_SPINE_MESH = "spine_morphology"
_C_SPINE_ID = "spine_id"
_C_ROTATION = ["spine_rotation_x", "spine_rotation_y",	"spine_rotation_z",	"spine_rotation_w"]
_C_TRANSLATION = ["afferent_surface_x", "afferent_surface_y", "afferent_surface_z"]


# Names of groups in the morphology-w-spines hdf5 file
GRP_EDGES = "edges"
GRP_MORPH = "morphology"
GRP_SPINES = "spines"
GRP_MESHES = "meshes"
GRP_SKELETONS = "skeletons"
GRP_SOMA = "soma"
GRP_VERTICES = "vertices"
GRP_TRIANGLES = "triangles"
GRP_OFFSETS = "offsets"


class MorphologyWithSpines(object):
    """
    A helper class to access the advanced information contained in the MorphologyWithSpines format.
    """
    def __init__(self, meshes_filename, morphology_name,
                 morphology, spine_table, centered_spine_skeletons, 
                 spines_are_centered=True):
        self._fn = meshes_filename
        self.name = morphology_name
        self._spines_are_centered = spines_are_centered
        self._centered_spine_skeletons = centered_spine_skeletons
        self.spine_table = spine_table
        self.smooth_morphology = morphology

        if self._spines_are_centered:
            self._spine_skeletons = self._transform_spine_skeletons()
        else:
            self._spine_skeletons = self._centered_spine_skeletons
    
    @property
    def spine_count(self):
        return self.spine_table.shape[0]
    
    def spine_transformations(self, i):
        rot = Rotation.from_quat(self.spine_table.loc[i, _C_ROTATION].to_numpy(dtype=float))
        tf = self.spine_table.loc[i, _C_TRANSLATION].to_numpy(dtype=float)
        return rot, tf
    
    def transform_for_spine(self, i, pts):
        rot, tf = self.spine_transformations(i)
        return rot.apply(pts) + tf.reshape((1, -1))
    
    def _transform_spine_skeletons(self):
        from neurom import load_morphology
        spines = self._centered_spine_skeletons.to_morphio().as_mutable()
        assert len(spines.root_sections) == self.spine_table.shape[0]
        for i in range(len(spines.root_sections)):
            lst_in = [spines.root_sections[i]]
            while len(lst_in) > 0:
                lst_out = []
                for sec in lst_in:
                    pts = self.transform_for_spine(i, sec.points)
                    sec.points = pts
                    lst_out.extend(sec.children)
                lst_in = lst_out
        return load_morphology(spines.as_immutable())

    @property
    def spine_skeletons(self):
        return self._spine_skeletons.neurites
    
    @property
    def centered_spine_skeletons(self):
        return self._centered_spine_skeletons.neurites

    def _spine_mesh_points(self, i, transform=True):
        _spine_mesh_grp = self.spine_table.loc[i, _C_SPINE_MESH]
        _spine_id = int(self.spine_table.loc[i, _C_SPINE_ID])
        with h5py.File(self._fn, "r") as h5:
            grp = h5[GRP_SPINES][GRP_MESHES][_spine_mesh_grp] #[_spine_id_grp]
            fr_v = grp[GRP_OFFSETS][_spine_id, 0]
            to_v = grp[GRP_OFFSETS][_spine_id + 1, 0]
            pts = grp[GRP_VERTICES][fr_v:to_v].astype(float)
        
        if not transform:
            return pts
        return self.transform_for_spine(i, pts)
    
    def spine_mesh_triangles(self, i):
        _spine_mesh_grp = self.spine_table.loc[i, _C_SPINE_MESH]
        _spine_id = int(self.spine_table.loc[i, _C_SPINE_ID])
        with h5py.File(self._fn, "r") as h5:
            grp = h5[GRP_SPINES][GRP_MESHES][_spine_mesh_grp] #[_spine_id_grp]
            fr_v = grp[GRP_OFFSETS][_spine_id, 1]
            to_v = grp[GRP_OFFSETS][_spine_id + 1, 1]
            triangles = grp[GRP_TRIANGLES][fr_v:to_v].astype(int)
        return triangles
    
    def spine_mesh_points(self, i):
        return self._spine_mesh_points(i, transform=self._spines_are_centered)
    
    def centered_mesh_points(self, i):
        return self._spine_mesh_points(i, transform=False)
    
    def spine_mesh(self, i):
        tm = trimesh.Trimesh(vertices=self.spine_mesh_points(i),
                             faces=self.spine_mesh_triangles(i))
        return tm
    
    def centered_spine_mesh(self, i):
        tm = trimesh.Trimesh(vertices=self.centered_mesh_points(i),
                             faces=self.spine_mesh_triangles(i))
        return tm
    
    @property
    def soma_mesh_points(self):
        with h5py.File(self._fn, "r") as h5:
            return h5[GRP_SOMA][GRP_MESHES][self.name][GRP_VERTICES][:].astype(float)
    
    @property
    def soma_mesh_triangles(self):
        with h5py.File(self._fn, "r") as h5:
            return h5[GRP_SOMA][GRP_MESHES][self.name][GRP_TRIANGLES][:].astype(int)
    
    @property
    def soma_mesh(self):
        tm = trimesh.Trimesh(vertices=self.soma_mesh_points,
                             faces=self.soma_mesh_triangles)
        return tm
