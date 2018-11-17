"""
Created on Thu Jan 18 21:24:15 2018

@author: oem
"""
import numpy as np


# Returns the Euclidean distance between 2 points
def Euclid_dist(xyz1, xyz2):
    diff = np.array(xyz1)-np.array(xyz2)
    return np.linalg.norm(diff)

# Finds the K minimum points in a list
def K_min(L, K, cutoff=np.inf):
    mins = []
    count = 0
    Lorig = L[:]
    L = np.array(L)
    L = list(L[L<cutoff])
    while (count<K):
        amin = np.argmin(L)
        mins.append(L[amin])
        L = L[:amin]+L[amin+1:]
        count += 1
    return [Lorig.index(i) for i in mins]

# Finds the K nearest neighbours. I have created my own implementation here instead of using a library such as scikitlearn for portability.
def KNN(coords, num_nbours, a_ind, cutoff=np.inf):
   dist = [Euclid_dist(coords[a_ind], coord) for i,coord in enumerate(coords)]
   #dist = dist[dist > cutoff]
   return K_min(dist, num_nbours)

# Calculates the pvec for 1 atom
def calc_pvec(all_active_coords, atom, step_info):
    nearest_neighbours = KNN(all_active_coords, 4, atom)[1:]
    #nearest_neighbours = [atom+1, atom+2, atom+3]
    nearest_coords = all_active_coords[nearest_neighbours]

    v1 = nearest_coords[0] - nearest_coords[2]
    v2 = nearest_coords[1] - nearest_coords[2]

    pvec = np.zeros(3)
    if nearest_neighbours[0] == nearest_neighbours[1]+1:
        pvec = np.cross(v1,v2)
    else:
        pvec = np.cross(v2,v1)
    #pvec[0] = veca[1]*vecb[2]-veca[2]*vecb[1]
    #pvec[1] = veca[2]*vecb[0]-veca[0]*vecb[2]
    #pvec[2] = veca[0]*vecb[1]-veca[1]*vecb[0]

    pvec /= np.linalg.norm(pvec)
    return pvec

#Calculates the pvecs for a single step
# Will be fairly easy to optimise,
#     * Can iterate over fewer atoms (atoms_to_plot).
#     * Only find nearest neighbours within same molecule (step_info['active_atoms_index'])
#     * Ignore the zero value (will make it less general :( )).
#     * Use a cutoff for the KNN
#     * Only create the active data (don't create the zeros -will require change of main code too)
#     * Can optimise the actual implementation if required using numpy
def calc_pvecs_for_1_step(all_active_coords, step, step_info):
    pvecs = np.zeros((len(all_active_coords[step]),3))
    for i in step_info['active_mols']:
       atom = i[1]
       if i[0] in step_info['AOM_D'].keys():
           pvecs[atom] = calc_pvec(all_active_coords[step],atom, step_info)
    return pvecs


# Returns the center and the size of a ND minimum bounding box
def min_bounding_box(coord, scaling=[2,2,2]):
   center = []
   spans  = []
   for i,x in enumerate(coord):
      maxx, minx = max(x), min(x)
      spans.append((maxx-minx)+scaling[i])
      center.append((maxx+minx)/2)

   return center, spans
