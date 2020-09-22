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

# Finds the K nearest neighbours. There are probably better implementations in scikitlearn
def KNN(coords, num_nbours, a_ind, cutoff=np.inf):
   dist = [Euclid_dist(coords[a_ind], coord) for i, coord in enumerate(coords)]
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



def calc_pvecs(all_settings, step):
	"""
	Will calc the pvec (perp vector to the atom and it's neighbouring 3 ats)

	Inputs:
		* all_settings => The big dict that stores all the data
	"""
	print("Calculating Pvecs")
	# These things can be pre-calculated in initialisation
	aps = all_settings['atoms_per_site']
	ats_to_calc_for = np.array(list(all_settings['mol_info'].keys()))
	active_mol_nums = ats_to_calc_for // aps
	unique_mol_nums = np.unique(active_mol_nums)

	# Get the crds of each molecule to calculate pvecs on
		# Get the full mol crds
		# Convert the active atoms to a mol and at_ind on that mol
	mol_to_at = {i: np.arange(i*aps, (i+1)*aps) for i in unique_mol_nums}
	all_mol_crds = {mol_num: all_settings['coords'][step][mol_to_at[mol_num]]
					for mol_num in unique_mol_nums}

	# Loop over the active atoms and find 2 nearest neighbours
	# Can optimise this with np.apply_along_axis()
	nearest_neighbours = []
	for iat, imol in zip(ats_to_calc_for, active_mol_nums):
		mol_crds = all_mol_crds[imol]
		at_crd = all_settings['coords'][step][iat]
		imolat = iat - (aps * imol)

		dists = np.linalg.norm(mol_crds - at_crd, axis=1)
		min_3 = [i[1] for i in sorted(zip(dists, mol_crds))[:3]]
		nearest_neighbours.append(min_3)

	nearest_neighbours = np.array(nearest_neighbours)

	# Take cross product of 2 nearest neighbours to get pvecs
	pvecs = {}
	for iat, ats in zip(ats_to_calc_for, nearest_neighbours):
		a1, a2, a3 = ats
		v1 = a2 - a1
		v2 = a3 - a1
		pvec = np.cross(v1, v2)
		pvec /= np.linalg.norm(pvec)
		key = all_settings['AOM_D'][iat][1]
		pvecs[key] = pvec

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
