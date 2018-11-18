"""
Contains functions that read the pvecs, the coordinates and the coefficients.

N.B. Maybe the AOM_COEFF file reader should be here...
"""

from src import IO as io
from src import type as typ
from src import consts

import numpy as np

# Will read the coordinates and combine them in the specified way
def read_coords(all_settings):
    all_coord_data = [io.read_xyz_file(filename,
                                       num_data_cols=3,
                                       min_step=all_settings['start_step'],
                                       max_step=all_settings['max_step'],
                                       stride=all_settings['stride'],
                                       ignore_steps=all_settings['n_steps_to_ignore'],
                                       metadata=all_settings['pos_metadata']) for filename in all_settings['CP2K_output_files']['pos']]
# Mean the reps
    if all_settings['mean_rep']:
        # Reading the nuclear positions
        all_settings['coords']      = np.mean([i[0] for i in all_coord_data], axis=0)
        all_settings['at_num']      = all_coord_data[0][1]
        all_settings['Ntime-steps'] = np.mean([i[2] for i in all_coord_data], axis=0)

    all_settings['coords'] = all_settings['coords']*consts.ang2bohr

    all_settings['at_num'] = [i.flatten() for i in all_settings['at_num']]
    # We only read the atom numbers for the first step and assume they're always the same
    all_settings['at_num'] = np.array([typ.atomic_num_convert(i) for i in all_settings['at_num'][0]])

    if all_settings['global_steps_to_ignore']: print("Finished reading coords, %i steps (already completed steps %s and %i)"%(len(all_settings['coords']),", ".join([str(i) for i in all_settings['global_steps_to_ignore']][:-1]), all_settings['steps_to_ignore'][-1]))
    else: print("Finished reading coords (%i steps)"%len(all_settings['coords']))

# Will read the coefficient files
def read_coeffs(all_settings):
    # Reading the mol coeffs
    all_mol_data = [io.read_xyz_file(f,
                                     num_data_cols=2,
                                     min_step=all_settings['start_step'],
                                     max_step=all_settings['max_step'],
                                     stride=all_settings['stride'],
                                     ignore_steps=all_settings['c_steps_to_ignore'],
                                     metadata=all_settings['coeff_metadata']) for f in all_settings['CP2K_output_files']['coeff']]
    if all_settings['mean_rep']:
        all_settings['mol'] = np.mean([i[0] for i in all_mol_data], axis=0)
        all_settings['Mtime-steps'] = np.mean([i[2] for i in all_mol_data], axis=0)
        all_settings['pops'] = np.linalg.norm(all_settings['mol'], axis=len(np.shape(all_settings['mol']))-1)
        all_settings['mol'] = np.array([[complex(*i) for i in step] for step in all_settings['mol']])

# Will read the pvecs file
def read_pvecs(all_settings):
    # Read all pvecs
    all_pvecs = [io.read_xyz_file(f,
                              num_data_cols=3,
                              min_step=all_settings['start_step'],
                              max_step=all_settings['max_step'],
                              stride=all_settings['stride'],
                              ignore_steps=all_settings['p_steps_to_ignore'],
                              metadata=all_settings['pvecs_metadata'])[0]
             for f in all_settings['CP2K_output_files']['pvecs']]

    # Check the pvecs (sometimes the initial step in the pvecs file gives all zeros)
    if all_settings['calibrate'] and np.sum(all_pvecs[0][0]) == 0:
        all_pvecs = [io.read_xyz_file(f,num_data_cols=3,
                                      min_step=1, max_step=2, stride=1)[0]
                     for f in all_settings['CP2K_output_files']['pvecs']]

    # Average all replicas
    if all_settings['mean_rep']:
        all_settings['pvecs'] = np.mean(all_pvecs, axis=0)
        if len(all_settings['pvecs']) == 0: raise SystemExit("Haven't read any pvecs!")

    # Same check as before but this time we'll use the combined pvecs
    if np.sum(all_settings['pvecs'][0]) == 0 and not all_settings['calibrate']:
        all_settings['pvecs'][0] = all_settings['pvecs'][1]

    if not (len(all_settings['pvecs']) == len(all_settings['mol']) and len(all_settings['pvecs']) == len(all_settings['coords'])):
        raise SystemExit("""Sorry something has gone wrong with the reading of the data.

The number of timesteps parsed in the pvecs array isn't equal to the coefficients or the coords.
You can let Matt know and he will make the code work for different length arrays (or you could make sure that
you print the pvecs and coefficients for every timestep you print the coordinates.)

    * len(pvecs) = %i
    * len(coeffs) = %i
    * len(positions) = %i
"""%(len(all_settings['pvecs']), len(all_settings['mol']), len(all_settings['coords'])))
