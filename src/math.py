"""
Contains math functions -these should probably be conbined with the geometry
module.

These include finding the rotation angle, creating the orbital data, finding
localisation etc...
"""


import numpy as np
import os
import time
import sys
if sys.version_info[0] > 2:
    xrange = range

# The P orbital creators, just for x
def SH_px(sizeX, sizeY, sizeZ, resolution, Origin = [0,0,0] ):
    x,y,z = np.mgrid[0:sizeX, 0:sizeY, 0:sizeZ].astype(float)
    resolution = float(resolution)
    sub_x = -(sizeX/2. + Origin[0]/resolution)
    sub_y = -(sizeY/2. + Origin[1]/resolution)
    sub_z = -(sizeZ/2. + Origin[2]/resolution)
    x += sub_x
    y += sub_y
    z += sub_z
    r = np.sqrt(x**2 + y**2 + z**2)
    return x*np.exp(-r)

# Calculates the locality using the IPR metric
def IPR(mol_coeffs):
   return 1/np.sum([np.absolute(i)**4 for i in mol_coeffs])

# Real P spherical harmonic for y
def SH_py(sizeX, sizeY, sizeZ, resolution, Origin = [0,0,0]  ):
    x,y,z = np.mgrid[0:sizeX, 0:sizeY, 0:sizeZ].astype(float)
    resolution = float(resolution)
    sub_x = -(sizeX/2. + Origin[0]/resolution)
    sub_y = -(sizeY/2. + Origin[1]/resolution)
    sub_z = -(sizeZ/2. + Origin[2]/resolution)
    x += sub_x
    y += sub_y
    z += sub_z
    r = np.sqrt(x**2 + y**2 + z**2)
    return y*np.exp(-r)

# Real P spherical harmonic for z
def SH_pz(sizeX, sizeY, sizeZ, resolution, Origin = [0,0,0]  ):
    x,y,z = np.mgrid[0:sizeX, 0:sizeY, 0:sizeZ].astype(float)
    resolution = float(resolution)
    sub_x = -(sizeX/2. + Origin[0]/resolution)
    sub_y = -(sizeY/2. + Origin[1]/resolution)
    sub_z = -(sizeZ/2. + Origin[2]/resolution)
    x += sub_x
    y += sub_y
    z += sub_z
    r = np.sqrt(x**2 + y**2 + z**2)
    return z*np.exp(-r)

# Real Spherical Harmonic, Optimised for 3 dimensions.
def SH_p(sizeX, sizeY, sizeZ, resolution, Origin = [0,0,0]):
    """
    Will create a p orbital centered at Origin. Been optimised
    for 3 dimensions. Psi_i = i * exp{-r} (r = |(x,y,z)|)

    Inputs:
        * sizeX        =>   size of X dim
        * sizeY        =>   size of Y dim
        * sizeZ        =>   size of Z dim
        * resolution   =>   resultion of the data (lower is better)
        * Origin       =>   center of the p orbital
    """
    # Creates an empty container for the orbital data.
    x,y,z = np.mgrid[0:sizeX, 0:sizeY, 0:sizeZ].astype(np.float64)
    resolution = float(resolution)

    # Careful with 2. here! Needs to be float for python2 back compatibility!
    #  I hate python2
    x -= (sizeX/2. + Origin[0]/resolution)
    y -= (sizeY/2. + Origin[1]/resolution)
    z -= (sizeZ/2. + Origin[2]/resolution)

    exp_neg_r = np.exp(
                       -np.sqrt(x**2 + y**2 + z**2) # r = sqrt(x^2 + y^2 + z^2)
                      )

    x *= exp_neg_r
    y *= exp_neg_r
    z *= exp_neg_r
    return np.array([x,y,z])

# Returns the dot product of 2 3D vectors.
def dot_3D(veca,vecb):
   return (veca[0]*vecb[0])+(veca[1]*vecb[1])+(veca[2]*vecb[2])

# Finds the angle required to put the long axis of the shape along a single axis
def find_angle(coords, lens, inds):
    i,j = inds
    if lens[i] > lens[j]:
        fit = np.polyfit(coords[:,i], coords[:,j],1)
        angle = np.arctan(fit[0])
    else:
        fit = np.polyfit(coords[:,j], coords[:,i],1)
        angle = np.pi/2. - np.arctan(fit[0])
    return angle

# Creates a 3D rotation matrix
def rot_mat_3D(angle, dim, unit='d'):
    if unit == 'd':
        angle *= np.pi/180.
    print(angle)
    if dim =='x':
        return np.matrix([
                [1, 0,          0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]])
    if dim =='y':
        return np.matrix([
                [np.cos(angle), 0,  np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]])
    if dim =='z':
        return np.matrix([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle),  0],
                [0,          0,           1]])

# Finds the dimensions of the system, including the center, min and max and lengths.
def find_sys_dims(coords):
    dims = {
    'max':np.array([np.max(coords[:,i]) for i in range(len(coords[0]))]),
    'min':np.array([np.min(coords[:,i]) for i in range(len(coords[0]))]),
            }
    dims['lens'] = dims['max']-dims['min']
    dims['center'] = dims['min']+dims['lens']/2.
    dims['largest_dims'] = np.argsort(dims['lens'])[::-1]
    return dims

# Finds a suitable rotation angle for the system
def find_auto_rotation(all_settings):
    coords = all_settings['coords'][0,all_settings['atoms_to_plot'],:]

    sys_dims = find_sys_dims(coords)
    coords -= sys_dims['center']

    # Aligns along the x axis
    zangle = find_angle(coords, sys_dims['lens'], [0,1])

    first_mol_indices = [i for i in  all_settings['mol_info'] if all_settings['mol_info'][i] == 0]
    coords2 = all_settings['coords'][0:,first_mol_indices,:][0]
    # Finds the rotation angle to align the first molecule to face the user (perp to z)
    new_sys_dims = find_sys_dims(coords2)
    xangle = find_angle(coords2, new_sys_dims['lens'], [1,2])
    return [xangle*(180./np.pi), 0, zangle*(180./np.pi)]

# # Converts a rotation matrix to Euler angles (that VMD can read)
# #This consists of a z rotation, followed by a y rotation, followed by an x rotation
# def rot_mat_to_euler_zyx(M):
#     # if M[0][2] == 1:
#     #     EXC.ERROR("GIMBAL LOCK Error. Trying to convert rotation matrix to Euler angles and getting infinite solutions")
#     roty = np.arcsin(M[0][2])
#     rotx = -np.arcsin(M[1][2]/np.cos(roty))
#     rotz = np.arccos(M[0][0]/np.cos(roty))
#     rotations = np.array([rotx, roty, rotz])*180/np.pi
#     for i,rotation in enumerate(rotations):
#         if rotation == np.nan:
#             EXC.WARN("Sorry the recorded rotation has resulted in Gimbal Lock. You could try again, with a slightly different rotation, or manually enter the required rotation in the settings file.\nFor now I will set the %s rotation angle to 0."%['x','y','z'][i])
#             rotations[i] = 0
#     return rotations
