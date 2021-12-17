'''Will test the geometry module in src.'''
import numpy as np

from src import geometry


def test_pvecs_1_mol():
    '''Test the calculation of the pvecs (direction of orbital) for 1 molecule'''
    np.random.seed(1209382)
    # Test pvecs with some reference values
    mol_crds = np.random.random(size=(10, 3))
    act_ats = np.array([0, 1, 2, 7])

    pvecs = geometry.calc_pvecs_1mol(mol_crds, act_ats)
    ref_pvecs = np.array([[-0.76291883,  0.44414717,  0.46977457],
                          [ 0.76291883, -0.44414717, -0.46977457],
                          [-0.76291883,  0.44414717,  0.46977457],
                          [-0.60108197, -0.69880286, -0.38778218]])
    assert np.all(np.isclose(pvecs, ref_pvecs))

    # Increase to large distances
    mol_crds *= 10
    pvecs = geometry.calc_pvecs_1mol(mol_crds, act_ats)


#def test_K_min():
#    '''Will test finding the K minimum points in a list'''
#    l = [1, 7, 23987, 23, 1 ,5]
#
#    # Test reasonable real case
#    k_min = geometry.K_min(l, 3)
#    assert all([i == j for i, j in zip(k_min, [0, 4, 5])])
#
#    # test cutoff and adding None
#    l = [1, 2, 3, 4, 5]
#    k_min = geometry.K_min(l, 3, cutoff=2)
#    assert k_min[0] == 0
#    assert k_min[1] == 1
#    assert k_min[2] is None


