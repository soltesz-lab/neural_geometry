
import numpy as np
from neural_geometry.geometry import transform_volume
from neural_geometry.rbf_surface import RBFSurface
from neural_geometry.rbf_volume import RBFVolume
import rbf

max_u = 4000.
max_v = 1250.

def CA1_volume_transform(u, v, l):
    return u, v, l

def CA1_volume(u, v, l, rotate=None):
    """Parametric equations of the CA1 volume."""

    return transform_volume(CA1_volume_transform, u, v, l, rotate=rotate)


def CA1_meshgrid(extent_u, extent_v, extent_l, resolution=[10, 10, 10], rotate=None, return_uvl=False):

    ures, vres, lres = resolution

    obs_u = np.linspace(extent_u[0], extent_u[1], num=ures)
    obs_v = np.linspace(extent_v[0], extent_v[1], num=vres)
    obs_l = np.linspace(extent_l[0], extent_l[1], num=lres)

    u, v, l = np.meshgrid(obs_u, obs_v, obs_l, indexing='ij')
    xyz = CA1_volume(u, v, l, rotate=rotate)

    if return_uvl:
        return xyz, obs_u, obs_v, obs_l
    else:
        return xyz


def make_CA1_volume(extent_u, extent_v, extent_l, rotate=None, basis=rbf.basis.phs3, order=2, resolution=[10, 10, 10],
                return_xyz=False):
    """Creates an RBF volume based on the parametric equations of the CA1 volume."""

    xyz, obs_u, obs_v, obs_l = CA1_meshgrid(extent_u, extent_v, extent_l, \
                                            rotate=rotate, resolution=resolution, return_uvl=True)
    vol = RBFVolume(obs_u, obs_v, obs_l, xyz, basis=basis, order=order)

    if return_xyz:
        return vol, xyz
    else:
        return vol


def make_CA1_surface(extent_u, extent_v, obs_l, rotate=None, basis=rbf.basis.phs2, order=1, resolution=[10, 10]):
    """Creates an RBF surface based on the parametric equations of the CA1 volume.
    """
    ures = resolution[0]
    vres = resolution[1]

    obs_u = np.linspace(extent_u[0], extent_u[1], num=ures)
    obs_v = np.linspace(extent_v[0], extent_v[1], num=vres)

    u, v = np.meshgrid(obs_u, obs_v, indexing='ij')
    xyz = CA1_volume(u, v, obs_l, rotate=rotate)

    srf = RBFSurface(obs_u, obs_v, xyz, basis=basis, order=order)

    return srf
