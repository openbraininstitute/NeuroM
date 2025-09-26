"""Represents a neron morphology with spines.

Provides utility and data acces to a representation of a
neuron morphology with individual spines.
"""

import h5py
import trimesh

from scipy.spatial.transform import Rotation

from neurom.core.morphology import Morphology

# Columns of edge table dataframes
_C_SPINE_MESH = "spine_morphology"
_C_SPINE_ID = "spine_id"
_C_ROTATION = ["spine_rotation_x", "spine_rotation_y", "spine_rotation_z", "spine_rotation_w"]
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


class MorphologyWithSpines(Morphology):
    """Represents spiny neuron morphology.

    A helper class to access the advanced information contained
    in the MorphologyWithSpines format.
    """

    def __init__(
        self,
        meshes_filename,
        morphology_name,
        morphio_morphology,
        spine_table,
        centered_spine_skeletons,
        spines_are_centered=True,
        process_subtrees=False,
    ):
        """Default constructor.

        morphio.io.utils.load_morphology_with_spines() intended for users.
        """
        super().__init__(
            morphio_morphology, name=morphology_name, process_subtrees=process_subtrees
        )
        self._fn = meshes_filename
        self.name = morphology_name
        self._spines_are_centered = spines_are_centered
        self._centered_spine_skeletons = centered_spine_skeletons
        self.spine_table = spine_table

        if self._spines_are_centered:
            self._spine_skeletons = self._transform_spine_skeletons()
        else:
            self._spine_skeletons = self._centered_spine_skeletons

    @property
    def spine_count(self):
        """Number of spines on morphology."""
        return self.spine_table.shape[0]

    def spine_transformations(self, i):
        """Spine coordinate system transformations.

        Transformations from the local coordinate system of a spine
        (origin near its root, y-axis pointing towards its tip) to the
        global coordinate system of the neuron.
        """
        rot = Rotation.from_quat(self.spine_table.loc[i, _C_ROTATION].to_numpy(dtype=float))
        tf = self.spine_table.loc[i, _C_TRANSLATION].to_numpy(dtype=float)
        return rot, tf

    def transform_for_spine(self, i, pts):
        """Apply spine coordinate system transformations.

        Apply the transformation from the local spine coordinate system
        to the global neuron coordinate system to a set of points.
        """
        rot, tf = self.spine_transformations(i)
        return rot.apply(pts) + tf.reshape((1, -1))

    def _transform_spine_skeletons(self):
        """Apply transformations to spine skeletons.

        A helper that transforms all centered (in local coordinate system)
        spine skeletons of this class to the global neuron coordinate system.
        """

        spines = self._centered_spine_skeletons.to_morphio().as_mutable()
        assert len(spines.root_sections) == self.spine_table.shape[0]
        for i, root_spine in enumerate(spines.root_sections):
            lst_in = [root_spine]
            while len(lst_in) > 0:
                lst_out = []
                for sec in lst_in:
                    pts = self.transform_for_spine(i, sec.points)
                    sec.points = pts
                    lst_out.extend(sec.children)
                lst_in = lst_out
        return Morphology(spines.as_immutable(), name=self.name + "_spines")

    @property
    def spine_skeletons(self):
        """The spine skeletons in global coordinates."""
        return self._spine_skeletons.neurites

    @property
    def centered_spine_skeletons(self):
        """The spine skeletons in local coordinates."""
        return self._centered_spine_skeletons.neurites

    def _spine_mesh_points(self, i, transform=True):
        """Points of spine mesh.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines.
        """
        _spine_mesh_grp = self.spine_table.loc[i, _C_SPINE_MESH]
        _spine_id = int(self.spine_table.loc[i, _C_SPINE_ID])
        with h5py.File(self._fn, "r") as h5:
            grp = h5[GRP_SPINES][GRP_MESHES][_spine_mesh_grp]  # [_spine_id_grp]
            fr_v = grp[GRP_OFFSETS][_spine_id, 0]
            to_v = grp[GRP_OFFSETS][_spine_id + 1, 0]
            pts = grp[GRP_VERTICES][fr_v:to_v].astype(float)

        if not transform:
            return pts
        return self.transform_for_spine(i, pts)

    def spine_mesh_triangles(self, i):
        """Triangles of spine mesh.

        The triangles (i.e., faces) of the meshes describing the shape of
        individual spines.
        """
        _spine_mesh_grp = self.spine_table.loc[i, _C_SPINE_MESH]
        _spine_id = int(self.spine_table.loc[i, _C_SPINE_ID])
        with h5py.File(self._fn, "r") as h5:
            grp = h5[GRP_SPINES][GRP_MESHES][_spine_mesh_grp]  # [_spine_id_grp]
            fr_v = grp[GRP_OFFSETS][_spine_id, 1]
            to_v = grp[GRP_OFFSETS][_spine_id + 1, 1]
            triangles = grp[GRP_TRIANGLES][fr_v:to_v].astype(int)
        return triangles

    def spine_mesh_points(self, i):
        """Points of spine mesh - global.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines. In global coordinates.
        """
        return self._spine_mesh_points(i, transform=self._spines_are_centered)

    def centered_mesh_points(self, i):
        """Points of spine mesh - local.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines. In local spine coordinates.
        """
        return self._spine_mesh_points(i, transform=False)

    def spine_mesh(self, i):
        """Spine mesh representation - global.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In global neuron coordinates.
        """
        tm = trimesh.Trimesh(vertices=self.spine_mesh_points(i), faces=self.spine_mesh_triangles(i))
        return tm

    def centered_spine_mesh(self, i):
        """Spine mesh representation - local.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In local spine coordinates.
        """
        tm = trimesh.Trimesh(
            vertices=self.centered_mesh_points(i), faces=self.spine_mesh_triangles(i)
        )
        return tm

    @property
    def soma_mesh_points(self):
        """Points of the soma mesh.

        The points (i.e., vertices) of the mesh describing the shape of
        the neuron soma.
        """
        with h5py.File(self._fn, "r") as h5:
            return h5[GRP_SOMA][GRP_MESHES][self.name][GRP_VERTICES][:].astype(float)

    @property
    def soma_mesh_triangles(self):
        """Triangles of the soma mesh.

        The triangles (i.e., faces) of the mesh describing the shape of
        the neuron soma.
        """
        with h5py.File(self._fn, "r") as h5:
            return h5[GRP_SOMA][GRP_MESHES][self.name][GRP_TRIANGLES][:].astype(int)

    @property
    def soma_mesh(self):
        """Returns the mesh (as a trimesh.Trimesh) of the neuron soma."""
        tm = trimesh.Trimesh(vertices=self.soma_mesh_points, faces=self.soma_mesh_triangles)
        return tm
