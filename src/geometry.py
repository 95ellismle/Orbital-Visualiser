"""
Similar to math -contains functions that calculate geometric quantities.

E.g. Euclidian distance between 2 points, K minimum points in a list, K nearest
neighbours, pvec calculator, minimum bounding box
"""
import numpy as np


# Returns the Euclidean distance between 2 points
def Euclid_dist(xyz1, xyz2):
	diff = np.array(xyz1) - np.array(xyz2)
	return np.linalg.norm(diff)

# Finds the K minimum points in a list
def K_min(L, K, cutoff=None):
    '''Find K minimum point in a list

    1:
     Put all values in a dict with structure:
        {element: [index of first appearance, index of second appearance ...]}
    2:
     For i in K:
        add each new min to list and remove that value from the dict above
        if this results in an empty list then remove the value

    Args:
        L: The list
        K: Num of points
        cutoff: any values over a certain value to exclude

    Returns:
        the index of the points within the list.
        If K > len(L) then add len(L) - K None values to the end of the list
    '''
    mins = [None] * K
    if cutoff:
        L = [i for i in L if i <= cutoff]
    if K > len(L):
        K = len(L)

    # 1
    vals = {}
    for index, elm in enumerate(L):
        vals.setdefault(elm, []).append(index)

    # 2
    for count in range(K):
        min_ = min(vals.keys())
        mins[count] = vals[min_][0]

        vals[min_] = vals[min_][1:]

        if len(vals[min_]) == 0:
            vals.pop(min_)

    return mins


# Finds the K nearest neighbours. There are probably better implementations in scikitlearn
def KNN(coords, num_nbours, a_ind, cutoff=np.inf):
   dist = [Euclid_dist(coords[a_ind], coord) for i, coord in enumerate(coords)]
   return K_min(dist, num_nbours)


def calc_pvecs_1mol(mol_crds, act_ats):
    """
    Will calculate the pvecs for 1 molecule.
    Inputs:
        * mol_crds <array> => The coordinates of each atom in the molecule
        * act_ats <array> => Which atom to calculate the pvecs for
    Outputs:
        <array> The pvecs
    """
    nearest_neighbours = np.zeros((len(act_ats), 3, 3))
    at_inds = np.arange(len(mol_crds))
    at_map = {}  # map at num to active at num

    # Loop over active atoms and calc nearest neighbours
    for count, iat in enumerate(act_ats):
        at_crd = mol_crds[iat]
        dists = np.linalg.norm(mol_crds - at_crd, axis=1)

        dist_mask = dists < 3.5
        nn_ats = at_inds[dist_mask][:3]
        if len(nn_ats) != 3:
            # Set the map at to the next closest one
            closest_at = K_min(list(dists), 2)
            at_map[count] = closest_at[1]
            continue
        else:
            # Make sure iat is the first atom
            nn_ats = nn_ats[nn_ats != iat][:2]
            nn_ats = [iat, *nn_ats]
        assert len(nn_ats) == 3

        nearest_neighbours[count] = mol_crds[nn_ats]

    # Set pvecs the same as the closest atom if we can't calculate them
    for at in at_map:
        nearest_neighbours[at] = nearest_neighbours[at_map[at]]

    pvecs = []
    for a1, a2, a3 in nearest_neighbours:
        v1 = a2 - a1
        v2 = a3 - a1
        pvec = np.cross(v1, v2)
        pvec /= np.linalg.norm(pvec)
        pvecs.append(pvec)

    return np.array(pvecs)


# Returns the center and the size of a ND minimum bounding box
def min_bounding_box(coord):
    min_ = np.min(coord, axis=1)
    max_ = np.max(coord, axis=1)
    spans = max_ - min_
    center = (max_ + min_) / 2.
    return center, spans
