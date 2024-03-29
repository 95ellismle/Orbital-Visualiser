from  __future__ import division
"""
Utility functions for the initialisation. These include things like setting all
the filepaths, initialising the colours, find which steps to ignore etc...

This file does not contain functions to read the coords, pvecs or the coeffs.
These can be found in the init/init_IO.py file
"""

from src import type as typ
from src import text as txt_lib
from src import IO as io
from src import math as MT
from src import consts
from src import EXCEPT as EXC
from src import load_xyz as xyz

from Templates import defaults as dft
try:
    from Templates import permanent_settings as ps
except:
    EXC.replace_perm_settings()
    from Templates import permanent_settings as ps

from init import settings_file
from init import init_IO as i_IO

import numpy as np
import datetime
import difflib
import os
import re
import sys
import subprocess
from pathlib import Path
Path.__repr__ = Path.__str__  # makes things easier
from collections import OrderedDict



def reverseDict(Dict):
    """
    Will create a dictionary that has all unique values as keys with
    """
    inv = {}
    for key, val in Dict.items():
        inv[val] = inv.get(val, []) + [key]
    return inv


# Will declare all the paths that are required in the code
def init_output_files_and_folders(all_settings):
    # Make all paths pathlib.Path objects
    for var_name, fp in (('img_fold', consts.img_folderpath),
                         ('data_fold', consts.data_folderpath),
                         ('tmplte_fold', consts.template_folderpath),
                         ('vmd_script_fold', './src/TCL'),
                         ('bin_fold', './bin'),
                         ('vmd_log_file', './visualisation.log'),
                         ('vmd_exe', all_settings['vmd_exe'])):
        all_settings[var_name] = Path(fp).absolute()

    for var_name, fp in (('vmd_temp', all_settings['tmplte_fold'] / "VMD_TEMP.vmd"),
                         ('ffmpeg_bin', all_settings['bin_fold'] / 'ffmpeg'),
                         ('ps_filepath',
                             all_settings['tmplte_fold'] / 'permanent_settings.py')):
        all_settings[var_name] = fp

    # Set up paths to data
    all_settings['path'] = Path(all_settings['path']).absolute()
    for var_name in ('aom_file', 'lumo_file', 'homo_file', 'decomp_file',
                     'cp2k_inp_file', 'coeff_file', 'pos_file'):
        all_settings[var_name] = all_settings['path'] / all_settings[var_name]

    for var_name, var in (('graph_files', []),
               ('vmd_junk', {}),
               ('vmd_script', {}),
               ('vmd_err', {}),
               ('delete_these', []),
               ('tcl', {})):
        all_settings[var_name] = var
    all_settings['tcl']['vmd_source_file'] = all_settings['tmplte_fold'] / 'include.vmd'


def check_filepaths(all_settings):
    """Will check we have all the files we need"""
    missing_files = []

    # Check AOM coeff filepaths
    if all_settings['do_transition_state']:
        all_settings.pop('aom_file')
        if not all_settings['lumo_file'].is_file():
            missing_files.append('lumo_file')
        if not all_settings['homo_file'].is_file():
            missing_files.append('homo_file')
    else:
        if not all_settings['aom_file'].is_file():
            missing_files.append('aom_file')

    # Check main data files
    required_files = ('pos_file', 'coeff_file', 'cp2k_inp_file')
    for f in required_files:
        if not all_settings[f].is_file():
            missing_files.append(f)

    if missing_files:
        print("The following input files were missing from your system:\n\t* "
              "\n\t* ".join(missing_files))
        print("Please check and correct these in the 'settings.inp' file.")
        raise IOError("Missing input files.")

    # Check internal files
    all_settings['vmd_script_fold'].mkdir(exist_ok=True)
    if not all_settings['vmd_temp'].is_file():
        raise IOError("VMD tcl template script can't be found. Please check it out from git.")
    if not all_settings['ps_filepath'].is_file():
        raise IOError("Permanent settings file can't be found. Please check it out from git.")
    if not all_settings['tmplte_fold'].is_dir():
        raise IOError("Templates folder can't be found. Please check it out from git.")



# I should definitely break up the bits that aren't actually used within the step here!
#output folders

# Will initialise settings that aren't file/folder paths
def init_all_settings_other(all_settings):
    all_settings['any_extra_raw_tcl_code'] = ''
    all_settings['clean_settings_dict'] = settings_file.final_orig_settings
    all_settings['settings_file']   = settings_file.settings_file
    all_settings['orig_settings']   = io.open_read(all_settings['settings_file'])
    all_settings['defaults']         = dft.defaults
    # Misc required step data
    all_settings['mols_plotted'] = ''
    all_settings['verbose_output']  = typ.translate_to_bool(all_settings['verbose_output'], 'verbose_output')
    all_settings['restart_vis']  = typ.translate_to_bool(all_settings['restart_vis'], 'restart_vis')
    all_settings['do_transition_state'] = typ.translate_to_bool(all_settings['do_transition_state'], 'do_transition_state')
    init_missing_pos_step_vars(all_settings)


# Will initialise the permanent settings (check tachyon path, read and write permanent settings, set prev_calibrate etc...)
def init_permanent_settings(all_settings):
    # Create the docs if they haven't already been created
    if not ps.created_docs:
       os.system("python3 Create_docs.py")
       io.read_write_perm_settings(all_settings['ps_filepath'], "created_docs", True)
    # Save the previous runtime
    io.read_write_perm_settings(all_settings['ps_filepath'], "previous_runtime",
                datetime.datetime.strftime(datetime.datetime.now(), ps.time_format))

#    # Checking Tachyon Renderer Path
#    new_tachyon_path = io.find_tachyon(ps.tachyon_path)
#    if new_tachyon_path != ps.tachyon_path:
#        io.read_write_perm_settings(all_settings['ps_filepath'], "tachyon_path", new_tachyon_path)
#        tachyon_path = new_tachyon_path
#    else:
#        tachyon_path = ps.tachyon_path
#    all_settings['tcl']['tachyon_path'] = tachyon_path

    # Did we calibrate last time?
    io.read_write_perm_settings(all_settings['ps_filepath'], "previous_path", all_settings['path'])
    if all_settings['calibrate']:
        io.read_write_perm_settings(all_settings['ps_filepath'], "previous_calibrate", True)
    else:
        io.read_write_perm_settings(all_settings['ps_filepath'], "previous_calibrate", False)


# Choose whether to show the bounding box or not in visualisation
def init_show_box(all_settings):
    all_settings['show_box'] = typ.translate_to_bool(all_settings['show_box'] ,'show_box')
    all_settings['dyn_bound_box'] = typ.translate_to_bool(all_settings['dynamic_bounding_box'], 'dynamic_bounding_box')


# Choose how many replicas to use
def init_rep_files(all_settings):
    all_settings['all_reps'] = txt_lib.fuzzy_variable_translate(all_settings['num_reps'], ["all","A list containing absolute positions i.e. [Pos_x, Pos_y, Pos_z]"], all_settings['verbose_output'], False)
    all_settings['mean_rep'] = txt_lib.fuzzy_variable_translate(all_settings['rep_comb_type'], ['mean'], all_settings['verbose_output'])
    if not all_settings['all_reps'] and type(all_settings['num_reps']) == int:
        all_settings['num_reps'] = range(all_settings['num_reps'])


# Will find which steps have already been completed so we can ignore them and
#   only do the steps that need doing.
def find_ignoring_steps(all_settings):
    """
    Find the global steps to ignore. These come from steps already completed in
    a previous run. This function will loop through the img folder and find the
    timesteps already completed and decide to ignore these steps.
    """

# Determines the correct all_settings['rotation'] vector to use
def init_rotation(all_settings):
    rot = all_settings['rotation']
    all_settings['rotation'] = np.array([0, 0, 0])
    return

    if type(rot) == str:
        auto_rot, _, _ = txt_lib.fuzzy_variable_translate(all_settings['rotation'],
                                                         ['auto', 'A list containing [Rotx, Roty, Rotz]', 'no'],
                                                         all_settings['verbose_output'], False)
        if auto_rot:
            all_settings['rotation'] = MT.find_auto_rotation(all_settings=all_settings)

        elif 'n' in all_settings['rotation'].lower() or bool(all_settings['rotation']) == False:
            all_settings['rotation'] = [0,0,0]


    elif all([isinstance(i, (float, int)) for i in rot]):
        all_settings = [i % 360 for i in rot]


# Sets the default tcl values
def init_tcl_dict(all_settings):
    all_settings['tcl']['any_extra_raw_tcl_code'] = ""
    all_settings['tcl']['isoval']   = all_settings['isosurface_to_plot']
    all_settings['tcl']['Ccol']     = all_settings['carbon_color']
    all_settings['tcl']['Hcol']     = all_settings['hydrogen_color']
    all_settings['tcl']['zoom_val'] = all_settings['zoom_value']
    #all_settings['tcl']['mol_id']   =  0
    all_settings['tcl']['Necol']         = all_settings['neon_color']
    all_settings['tcl']['atom_style']   = all_settings['mol_style']
    all_settings['tcl']['mol_material'] = all_settings['mol_material']
    all_settings['tcl']['iso_material'] = all_settings['iso_material']
    all_settings['tcl']['time_step']    = '" "'
    all_settings['tcl']['iso_type']     = 0
    all_settings['tcl']['density_color']     = str(all_settings['density_iso_col']).replace("(",'"').replace(")",'"').replace(",",'')
    all_settings['tcl']['imag_pos_col'] = str(all_settings['pos_imag_iso_col']).replace("(",'"').replace(")",'"').replace(",",'')
    all_settings['tcl']['imag_neg_col'] = str(all_settings['neg_imag_iso_col']).replace("(",'"').replace(")",'"').replace(",",'')
    all_settings['tcl']['real_pos_col'] = str(all_settings['pos_real_iso_col']).replace("(",'"').replace(")",'"').replace(",",'')
    all_settings['tcl']['real_neg_col'] = str(all_settings['neg_real_iso_col']).replace("(",'"').replace(")",'"').replace(",",'')
    all_settings['tcl']['maxX'] = all_settings['xdims'][1]
    all_settings['tcl']['minX'] = all_settings['xdims'][0]
    all_settings['tcl']['maxY'] = all_settings['ydims'][1]
    all_settings['tcl']['minY'] = all_settings['ydims'][0]
    all_settings['tcl']['maxZ'] = all_settings['zdims'][1]
    all_settings['tcl']['minZ'] = all_settings['zdims'][0]
    all_settings['tcl']['tachyon_path'] = io.get_tachyon_path(all_settings['vmd_exe'])

    imgSize = all_settings['img_size']
    if type(imgSize) == str:
        if imgSize.lower() == 'auto':
            if all_settings['calibrate']:
                imgSize = [1000, 1000]
            else:
                imgSize = [650, 650]
        else:
            raise SystemExit("Unknown setting %s for the `img_size'" % imgSize)
    all_settings['tcl']['pic_sizex'] = imgSize[0]
    all_settings['tcl']['pic_sizey'] = imgSize[1]

    all_settings['tcl']['backgrnd_mols'] = ""
    if all_settings['background_mols']:
        all_settings['tcl']['bckg_mols_on_off'] = ''
    else:
        all_settings['tcl']['bckg_mols_on_off'] = '#'
    if all_settings['show_box']:
       all_settings['tcl']['iso_type']  = 2
    all_settings['tcl'] = txt_lib.tcl_3D_input(all_settings['background_color'],
                                               ['R','G','B'],
                                               all_settings['tcl'],
                                               "backgrnd_")
    all_settings['tcl'] = txt_lib.tcl_3D_input(all_settings['translate_by'],
                                               ['x','y','z'],
                                               all_settings['tcl'],
                                               "trans_")
    all_settings['tcl'] = txt_lib.tcl_3D_input([0,0,0],
                                               ['x','y','z'],
                                               all_settings['tcl'],
                                               "time_lab_")
    all_settings['tcl'] = txt_lib.tcl_3D_input(all_settings['rotation'],
                                               ['x','y','z'],
                                               all_settings['tcl'],
                                               "rot")
    all_settings['tcl']['pic_filename'] = {}
    # set this in settings
    all_settings['tcl']['vmd_log_file'] = all_settings['vmd_log_file']
    # if not io.use_prev_scaling(all_settings['path']) and os.path.isfile(all_settings['tcl']['vmd_source_file']):
    #     bool(all_settings['draw_time'], "draw_time")

# Choose how to display the data after calibration
def init_cal_display_img(all_settings):
    all_settings['show_img_after_vmd']  = typ.translate_to_bool(all_settings['show_img_after_vmd'], 'show_img_after_vmd')
    all_settings['load_in_vmd'] = typ.translate_to_bool(all_settings['load_in_vmd'], 'load_in_vmd')
    if not all_settings['calibrate']:
        all_settings['load_in_vmd'] = False


# Choose which files to keep
def init_files_to_keep(all_settings):
    # Translate the input commands to a boolean (logical)
    all_settings['keep_cube_files'] = typ.translate_to_bool(all_settings['keep_cube_files'],'keep_cube_files')
    all_settings['keep_img_files']  = typ.translate_to_bool(all_settings['keep_img_files'],'keep_img_files')
    all_settings['keep_tga_files']  = typ.translate_to_bool(all_settings['keep_tga_files'],'keep_tga_files')

    all_settings['files_to_keep'] = []
    if all_settings['keep_cube_files']:
        all_settings['files_to_keep'].append("cube")
    if all_settings['keep_img_files']:
        all_settings['files_to_keep'].append("img")
    if all_settings['keep_tga_files']:
        all_settings['files_to_keep'].append("tga")


def init_AOM_D(all_settings):
    """Read the AOM file"""
    # Standard Movie Maker
    if not all_settings['do_transition_state']:
        all_settings['AOM'] = i_IO.AOMFile(all_settings['aom_file'],
                                           all_settings['atoms_per_site'],
                                           all_settings['nmol'])

    else:
        # Transition State
        # Parse AOM file
        all_settings['LUMO'] = i_IO.AOMFile(all_settings['lumo_file'],
                                            all_settings['atoms_per_site'],
                                            all_settings['nmol'])
        all_settings['HOMO'] = i_IO.AOMFile(all_settings['homo_file'],
                                            all_settings['atoms_per_site'],
                                            all_settings['nmol'])

        # Check the atoms are the same in each file
        for lat, hat in zip(all_settings['LUMO'],
                            all_settings['HOMO']):
            if lat != hat:
                raise SystemExit("Atom indices don't match up in LUMO and HOMO files.")


def get_all_files_metadata(all_settings):
    """Will retrieve the metadata for the coefficient, position and pvecs file.
    The metadata is needed to decide which steps to complete when running the simulation
    """
    do_cal, calTStep = all_settings['calibrate'], all_settings['timestep_to_render']
    if do_cal:
        max_time = all_settings['timestep_to_render']
    else:
        max_time = all_settings['end_time']
        if isinstance(max_time, str):
            max_time = None

    all_settings['pos_metadata']   = xyz.get_xyz_metadata(all_settings['pos_file'], max_time)
    all_settings['coeff_metadata'] = xyz.get_xyz_metadata(all_settings['coeff_file'], max_time)


# Finds what format the animation file should take
def init_animation_type(all_settings):
    use_mp4 = txt_lib.fuzzy_variable_translate(all_settings['movie_format'], ["MP4"], all_settings['verbose_output'])
    if use_mp4:
            all_settings['movie_format'] = 'mp4'
    # if use_gif:
    #     all_settings['movie_format'] = 'gif'


def init_colors(all_settings):
    """
    Initialises the colors of the wavefunction e.g. whether to use density,
    a purely real phase (neg and pos) or full complex phase (pos, neg, imag,
    real).
    """
    density, real_phase, full_phase = txt_lib.fuzzy_variable_translate(all_settings['type_of_wavefunction'], ["density", "real-phase", "phase"], all_settings['verbose_output'])
    if density:
        all_settings['color_type'] = 'density'
    elif real_phase:
        all_settings['color_type'] = 'real-phase'
    elif full_phase:
        all_settings['color_type'] = 'phase'
    else:
        EXC.WARN("Sorry I'm not sure what type of color I should use, defaulting to %s"%dft.defaults['type_of_wavefunction'])
        all_settings['color_type'] = dft.defaults['type_of_wavefunction']


# Will initialise the bounding box (make it a list of 3 nums 1 for each dim)
def init_bounding_box(all_settings):
    if (type(all_settings['bounding_box_scale']) == int) or (type(all_settings['bounding_box_scale']) == float):
       all_settings['bounding_box_scale'] = [all_settings['bounding_box_scale']]*3
    if type(all_settings['bounding_box_scale']) != list and all_settings['verbose_output']:
       EXC.WARN("The 'bounding_box_scale' variable doesn't seem to be set correctly! \nCorrect options are:\n\t* integer or float\n\tlist of ints or floats (for x,y,z dimensions)")


# Reading the inp file and finding useful data from it
def read_cp2k_inp_file(all_settings):
    with open(all_settings['cp2k_inp_file'], 'r') as f:
        run_inp = f.read()
    all_settings['atoms_per_site'] = int(txt_lib.inp_keyword_finder(run_inp, "ATOMS_PER_SITE")[0])
    ndiab_states = int(txt_lib.inp_keyword_finder(run_inp, "NUMBER_DIABATIC_STATES")[0])
    norbits = int(txt_lib.inp_keyword_finder(run_inp, "NUMBER_ORBITALS")[0])
    all_settings['nmol'] = int(ndiab_states/norbits)


def init_ignore_steps_for_restart(all_settings):
    """
    Find which steps to ignore for restarting visualisations.

    We loop over any files in the folder that we're writing to and if
    the images corresponding to the timesteps that need to be done are there
    then don't re-plot them.
    """
    if all_settings['restart_vis'] and not all_settings['calibrate']:
       # Check if we have the image folder containing prior data
       img_fold = all_settings['img_fold'] + "/" + all_settings['title']
       if os.path.isdir(img_fold):
           all_tga_files = [re.findall("[0-9,]+", i) for i in os.listdir(img_fold) if '.tga' == i[-4:]]
           completed_timesteps = set([float(i[0].replace(",", ".")) for i in all_tga_files if i])

           # If we don't use a missing position correction then use strict rules for pos and coeff
           if all_settings['missing_pos_steps'] == 'skip':
               timesteps_names = ('nucl_tsteps_to_read', 'coeff_tsteps_to_read')
           else:
               timesteps_names = ('coeff_tsteps_to_read',)

           # Remove any steps to do that have been done
           for name in timesteps_names:
              tmp = []
              for i in all_settings[name]:
                  if i not in completed_timesteps:
                      tmp.append(i)
              all_settings[name] = tmp

              if len(tmp) == 0:
                  raise SystemExit("No more steps for me to carry out. If you would like to stitch the images then use a script from the UsefulScripts folder.")


# Will find which atoms should be plotted
def init_atoms_to_plot(all_settings):
    """Find which atoms should be plotted"""
    # Translate the input file to something a computer can understand
    all_settings['background_mols']       = typ.translate_to_bool(all_settings['background_mols'],'background_mols')
    if type(all_settings['atoms_to_plot']) == str:
        options = ["all", "have_population", "A list containing atom indices (or range or xrange)", "An integer" ]
        tmp = txt_lib.fuzzy_variable_translate(all_settings['atoms_to_plot'],
                                               options,
                                               all_settings['verbose_output'],
                                               False)
        plot_all_atoms, plot_pop_mols, _ ,_ = tmp

    # Need atoms_to_plot to be a list
    if isinstance(all_settings['atoms_to_plot'], int):
        all_settings['atoms_to_plot'] = [all_settings['atoms_to_plot']]

    # Deciding which atoms to plot
    if all_settings['do_transition_state']:
        aom_file = all_settings['LUMO']
    else:
        aom_file = all_settings['AOM']

    if plot_all_atoms:
        all_settings['atoms_to_plot'] = None #np.array(list(aom_file.get_active_atoms()))
    if plot_pop_mols:
        all_settings['atoms_to_plot'] = 'have_population'


# Finds list of keys attached to a value in a dictionary
def find_value_dict(D, value):
    """Find every key that has a corresponding value of `value`"""
    return [key for key in D if D[key] == value]


# Will check if the charge is spread too much and report back to the user
def check_charge_spread(all_settings):
    print(all_settings['active_atoms_index'])
    raise SystemExit
    all_settings['num_mols_active'] = len([i for i in  all_settings['active_atoms_index'] if i])
    if all_settings['max_act_mol'] > all_settings['num_mols_active']:
        cont = raw_input("The charge will eventually leave the number of plotted molecules in this simulation.\n\nCharge reaches mol %i (this is out of bounds for the mols plotted).\n\nThis is often due to the AOM file not having a coefficient for each atom. You may want to check this.\n\nAre you sure you want to continue now [y/n]:\t"%all_settings['max_act_mol'])
        if not typ.translate_to_bool(cont, 'cont'):
            raise SystemExit("\n\nOk, exiting so you can change the settings file.\nI suggest using the keyword:\n`atoms_to_plot = 'auto'` in the settings file.")
        else:
            print("Okie doke, you will get empty molecules though!")
    all_active_coords = all_settings['coords'][:,[i[0] for i in all_settings['active_mols']]]


def init_steps_to_do(all_settings):
   """
   Will simply create arrays with all steps that are possible for the nuclear and coefficient steps.

   Will add 2 arrays to the all_settings dict: 'nucl_tsteps_to_read' and 'coeff_tsteps_to_read'.
   """
   all_settings['nucl_tsteps_to_read'] = all_settings['pos_metadata']['time_steps']
   all_settings['coeff_tsteps_to_read'] = all_settings['coeff_metadata']['time_steps']


# Will create the storage containers to store the timings
def init_times_dict(all_settings):
    """
    Will initialise the dictionary that stores all the timing data. This
    involves creating arrays that are just big enough to store all the timings
    from each step for the different categories.
    """
    all_settings['init_times'] = {}
    num_steps = len(all_settings['pos_step_inds'])
    all_settings['times'] = OrderedDict()
    all_settings['times']['Create Wavefunction'] = np.zeros(num_steps)
    all_settings['times']['Create Cube Data'] = np.zeros(num_steps)
    all_settings['times']['Write Cube File'] = np.zeros(num_steps)
    all_settings['times']['Create Pvecs'] = np.zeros(num_steps)
    all_settings['times']['VMD Visualisation'] = np.zeros(num_steps)
    all_settings['times']['Plot and Save Img'] = np.zeros(num_steps)
    all_settings['times']['WF Post Processing'] = np.zeros(num_steps)
    all_settings['times']['Create All SOMOs'] = np.zeros(num_steps)
    all_settings['times_taken'] = []


def __translateEndStep(all_settings, numSteps, setting):
    """
    Will translate the val from a float or a string to a step. For example if
    `half' is given as the value then this function will return the half (pos)
    timestep.

    Inputs:
        * all_settings => all setting dict
        * numSteps => max number of steps allowed
        * setting => The setting to translate
    """
    val = all_settings[setting]
    if type(val) == str:
        allowedOptions = np.array(['all', 'half', 'last',
                                   "An integer or float (see docs)"])
        choices = {'all': numSteps,
                   'half': int(numSteps * 0.5),
                   'last': numSteps}

        settMask = txt_lib.fuzzy_variable_translate(
                                            all_settings['end_time'],
                                            list(allowedOptions),
                                            all_settings['verbose_output'],
                                            False)
        choice = allowedOptions[settMask]
        if len(choice) > 1:
            raise SystemExit("Error in numpy mask")
        return choices[choice[0]]


    elif type(val) == float:
        if val > 1:
            return int(val)
        else:
            return int(numSteps * val)

    elif type(val) == int:
        return val
    else:
        msg = "Don't have any rules to process %s in the input" % (str(val))
        msg += ". Bad setting = %s" % setting
        raise SystemExit(msg)


# Will initialise the start_time, end_time and stride variables
def find_step_numbers(all_settings):
    """
    This function will set the start step, end step and stride for the simulation.

    N.B. Step nums are inclusive i.e. start <= i <= end.
    """
    # Choose either nuclear of coefficient timesteps based on the missing pos step setting.
    if all_settings['missing_pos_steps'] == 'skip':
        availSteps = all_settings['pos_metadata']['time_steps']
    else:
        availSteps = all_settings['coeff_metadata']['time_steps']

    do_cal, calTStep = all_settings['calibrate'], all_settings['timestep_to_render']
    startT, endT, stride = all_settings['start_time'], all_settings['end_time'], all_settings['stride']

    if type(calTStep) == str:
        if calTStep.lower() == "last":
            calTStep = max(availSteps)
            all_settings['timestep_to_render'] = calTStep
        else:
            raise SystemExit("Bad value for 'timestep_to_render' please see choose int, float or 'last' (not '%s'). See docs for more info." % calTStep)

    # Render 1 pic
    if do_cal:
        all_settings['stride'] = 1
        if calTStep > max(availSteps):
            if all_settings['missing_pos_steps'] != 'skip':
               EXC.WARN("Calibration step (%.2f) chosen to be out of bounds of the simulation data. Using last step instead." % calTStep)
               calTStep = availSteps[-1]
            else:
               EXC.ERROR("Calibration step (%.2f) chosen to be out of bounds of the simulation data." % calTStep)

        if calTStep not in availSteps:
           if all_settings['missing_pos_steps'] != 'skip':
              diffs = availSteps - calTStep
              cal_step_ind = np.argmin(np.abs(diffs))
              new_calTStep = availSteps[cal_step_ind]
              EXC.WARN("Calibration step %.2f isn't in the available timesteps. Using timestep %.2f instead." % (calTStep, new_calTStep))
              calTStep = new_calTStep
           else:
              EXC.ERROR("Calibration step %.2f isn't in the available timesteps." % (calTStep))

        all_settings['start_time'] = calTStep
        all_settings['end_time'] = calTStep

    # Making a movie
    else:
        # Set the end timestep if a string is given (e.g. 'all')
        if type(endT) == str:
            possVals = ['all', 'A float or integer representing the timestep to end the visualisation (see docs)']
            var = txt_lib.fuzzy_variable_helper(endT.lower(), possVals)
            if var == 'all':  all_settings['end_time'] = availSteps[-1]
        else:
            all_settings['end_time'] = __convert_to_float(endT, "end_time")

        all_settings['start_time'] = __convert_to_float(startT, "start_time")
        all_settings['stride'] = __convert_to_int(stride, "stride")

    correct_steps_to_read_startMaxStride(all_settings)

def __convert_to_int(string, var_name):
    """
     Will convert a number to a int and if it can't be done raise an error.
    """
    try:
       val=int(string)
    except:
       EXC.ERROR("'%s' must be a int" % var_name)
    return val


def __convert_to_float(string, var_name):
    """
    Will convert a number to a float and if it can't be done raise an error.
    """
    try:
       val=float(string)
    except:
       EXC.ERROR("'%s' must be a float" % var_name)
    return val


def correct_steps_to_read_startMaxStride(all_settings):
   """
   Will adjust the steps to read so that no steps outside of the
   start_time, end_time and stride are included (subject to caveats).

   Caveats:
      * If 'missing_pos_steps' is used then we don't adjust the pos steps
        and allow another function to fix those later. This doesn't apply if
        the 'skip' setting is used in missing pos steps.

   Will change everything in place in the all_settings dictionary.
   """
   start_time = all_settings['start_time']
   end_time = all_settings['end_time']
   stride = all_settings['stride']
   var = all_settings['missing_pos_steps']

   # Using no correction for missing position steps
   if var == "skip":
       names = ('nucl_tsteps_to_read', 'coeff_tsteps_to_read')
   else:
       names = ('coeff_tsteps_to_read',)

   # Only allow steps allowed by min_step, max_step and stride
   for name in names:
       corr_stride_steps = set(all_settings[name][::stride])
       tmp = []
       for i in all_settings[name]:
           if i in corr_stride_steps and i <= end_time and i >= start_time:
               tmp.append(i)

       if len(tmp) == 0 and all_settings['missing_pos_steps'] == 'skip':
         common_timesteps = set(all_settings['nucl_tsteps_to_read']).intersection(set(all_settings['coeff_tsteps_to_read']))
         EXC.ERROR("Can't find any nucl and coeff timesteps to read.\n\nPlease adjust your settings.inp file." +
                                   "Available steps are: %s" % ', '.join(map(str, common_timesteps)))

       elif len(tmp) == 0 and all_settings['missing_pos_steps'] != 'skip':
           EXC.ERROR("Can't find any coeff timesteps to read.\n\nPlease adjust your settings.inp file" +
                     ".\nAvailable steps are: %s" % ', '.join(map(str, all_settings['coeff_tsteps_to_read'])))

       all_settings[name] = tmp


def init_missing_pos_step_vars(all_settings):
    """
    Will initialise the missing_pos_steps variable.

    There are 3 options:
         'skip' -> Will simply ignore the steps that don't have positions.
         'closest' -> Will use the closest known position to the coeff timestep.
         'use N' -> Will use a specified position timestep.

    This will change the setting in the all_settings dictionary.
    """
    var = all_settings['missing_pos_steps'].lower()
    varSplit = var.strip().split()

    if len(varSplit) == 0:
        EXC.ERROR("Please set the variable 'missing_pos_steps'.\n\n"
                 +"It is currently: '%s'" % var)

    poss_vars = ('skip', 'closest', 'use')
    varFixed = txt_lib.fuzzy_variable_helper(varSplit[0], poss_vars)
    if varFixed == 'use':

        if len(varSplit) != 2:
            EXC.ERROR("The correct syntax for using the 'use' keyword in the 'missing_pos_steps'"
                    + " is `missing_pos_steps = 'use N' where N represents the step you wish to"
                    + " use as the atomic coords for the full visualisation.")
        else:
            try:
               all_settings['use_missing_pos_step'] = int(varSplit[1])
            except:
               EXC.ERROR("Can't find position step '%s'. Please choose an integer." % varSplit[1])

    all_settings['missing_pos_steps'] = varFixed


def get_closest_inds(arr1, arr2):
    """
    Will return an array of size len(arr2) with integers that tell the user
    which index in arr1 is closest to the value in arr2.

    N.B.
    The inputs arr1 and arr2 will be sorted at the beginning. This will give
    incorrect results if the data isn't sorted.
    """
    arr1_list = sorted(list(arr1))
    arr2_list = sorted(list(arr2))
    arr1_hm_inds = {val: i for i, val in enumerate(arr1)}

    closest_inds = []
    i, j = 0, 0
    while (i < len(arr2_list)):
        coeff_dt = arr2_list[i]
        pos_dt = arr1_list[j]

        # If the value is exactly there just add the index of
        #   where it appears in the coeff list
        if coeff_dt in arr1_hm_inds:
            j = arr1_hm_inds[coeff_dt]
            closest_inds.append(j)

        # Find the closest index
        else:
            curr_val = abs(arr1_list[j] - coeff_dt)
            for k in range(j+1, len(arr1_list)):
                if abs(arr1_list[k]-coeff_dt) < curr_val:
                    j = k
                else: break
            closest_inds.append(j)
        i += 1

    return closest_inds



def fix_missing_pos_steps(all_settings):
    """
    Will initialise which timesteps to carry out based on which steps are
    available.

    This is dependant on the method chosen to correct for missing position steps.
    If 'skip' is chosen then the following applies:
             if nuclear timesteps = [1,2,3,    6]
             and coeff timesteps =  [1,2,3,4,5,6]
             we would carry out     [1,2,3,    6]...

    If 'closest' or 'use ' is chosen we correct for missing position steps as
    outlined in the documentation or the docstr on the function `missing_pos_steps`

    We do this by finding which steps aren't common to all 3 lists as that is
    the way the xyz reader works.
    """
    if all_settings['missing_pos_steps'] == 'skip':
       common_timesteps = np.intersect1d(all_settings['nucl_tsteps_to_read'],
                                         all_settings['coeff_tsteps_to_read'])
       all_settings['nucl_tsteps_to_read'] = common_timesteps
       all_settings['coeff_tsteps_to_read'] = common_timesteps
       all_settings['pos_step_inds'] = np.arange(len(common_timesteps))

    elif all_settings['missing_pos_steps'] == 'use':
       use_step = all_settings['use_missing_pos_step']
       pos_steps = all_settings['nucl_tsteps_to_read']
       mol_steps = all_settings['coeff_tsteps_to_read']
       if 0 > use_step > len(pos_steps):
          EXC.ERROR("Step %i is out of bounds to use as a the correction to missing position steps." % use_step
                  + "\n\nPlease choose a step 0 <= i <= %i as `missing_pos_steps = 'use N'`" % len(pos_steps))
       all_settings['nucl_tsteps_to_read'] = [pos_steps[use_step]]
       all_settings['pos_step_inds'] = [0] * len(mol_steps)

    elif all_settings['missing_pos_steps'] == 'closest':
        pos_steps = all_settings['nucl_tsteps_to_read']
        mol_steps = all_settings['coeff_tsteps_to_read']
        all_settings['pos_step_inds'] = get_closest_inds(pos_steps, mol_steps)
        pos_inds = sorted(np.unique(all_settings['pos_step_inds']))
        all_settings['nucl_tsteps_to_read'] = [pos_steps[i] for i in pos_inds]
        all_settings['pos_step_inds'] = np.array(sorted(all_settings['pos_step_inds']))
        all_settings['pos_step_inds'] -= all_settings['pos_step_inds'][0]
        if all_settings['calibrate'] and pos_inds:
            all_settings['pos_step_inds'] = [0]


# Will check the last lines of the file TemplatesVMD_TEMP.vmd are known
def check_VMD_TEMP(all_settings):
    ltxt = io.open_read(all_settings['vmd_temp']).strip('\n').split('\n')
    for i, line in enumerate(ltxt):
        if 'render' in line.lower() and 'tachyon' in line.lower():
            break
    ltxt = ltxt[:i+1]
    ltxt.append(consts.end_of_vmd_file)
    io.open_write(all_settings['vmd_temp'], '\n'.join(ltxt))

# Will find the replica number in the filename
def find_rep_num_in_name(filename):
#    labs = ['pos','ham','ener','coeff','vel']
    regex_matches = {'n-pos':"-\d*-\d*\.xyz",
                     'ham':"-\d*-\d*\.xyz",
                     'n-ener_':"_\d*-\d*\.dat",
                     'coeff-':"f-\d*-\d*\.xyz",
                     'n-vel':"-\d*-\d*\.xyz",
                     'n-frc':"_\d*-\d*\.xyz",
                     't_frc':"_\d*\.xyz",
                     'd_frc':"_\d*\.xyz",
                     "coeff_a":"_\d*-\d*\.xyz",
                     "QM":"-\d*-\d*\.xyz",
                     "d_ener":"-\d*-\d*\.csv",}
    regex_matches2 = {'n-pos':"\d*-\d*\.",
                     'ham':"\d*-\d*\.",
                     'n-ener_':"\d*-\d*\.",
                     'coeff-':"\d*-\d*\.",
                     'n-vel':"\d*-\d*\.",
                     'n-frc':"\d*-\d*\.",
                     't_frc':"\d*\.",
                     'd_frc':"\d*\.",
                     "coeff_a":"\d*-\d*\.",
                     "QM":"\d*-\d*\.",
                     "d_ener":"\d*-\d*\.",}
    final_delim = {'n-pos':"-",
                   'ham':"-",
                   'n-ener_':"-",
                   'coeff-':"-",
                   'n-vel':"-",
                   'n-frc':"-",
                   't_frc':".",
                   'd_frc':".",
                   "coeff_a":"-",
                   "QM":"-",
                   "d_ener":"-",}

    for lab in regex_matches:
        if lab in filename:
            reduced_fname = re.findall(regex_matches[lab], filename)
            if len(reduced_fname) == 1:
                rep = re.findall(regex_matches2[lab], reduced_fname[0])
                if len(rep) == 1:
                    for ichar, char in enumerate(rep[0]):
                        if char == final_delim[lab]:
                            try:
                                rep = int(rep[0][:ichar])
                                return rep
                            except TypeError:
                                print("Sorry something went wrong extracting the replica number from a file")
                                print("Cannot convert %s to an integer"%str(rep[0][:ichar]))
                                print("\nFilename = %s"%filename)
                                print("Please make sure the replica number is followed by a hyphen i.e. 'run_pos-(1-)1.xyz'")
                                raise SystemExit("Type Conversion Error")
                            break
                    else:
                        print("Cannot find a hyphen in the filename.")
                        print("Please make sure the replica number is followed by a hyphen i.e. 'run_pos-(1-)1.xyz'")
                        print("\nFilename = %s"%filename)
                        raise SystemExit("Missing Final Delimeter")
                    break
                else:
                    raise SystemExit("""Sorry I couldn't find the replica number with the regex.

Filename = '%s'            regex = '\d*-\d*\.'

Filename after regex = '%s' """%(reduced_fname, str(rep)))
            else:
                raise SystemExit("""Sorry I the pre-programmed regex doesn't work for this file.

Filename = '%s'            regex = '%s'

Filename after regex = '%s' """%(filename, regex_matches[lab], str(reduced_fname)))
    else:
        raise SystemExit("Sorry I couldn't find the file type (pos, vel, frc, coeff etc...), something went wrong!\n\nFilename = %s"%(filename))


def get_mol_groupings(all_settings):
    """Will calculate which atoms belong to which molecule and which are active."""
    # Get the set of active atoms
    if all_settings['do_transition_state']:
        aom = all_settings['LUMO']
    else:
        aom = all_settings['AOM']

    act_ats = aom.get_active_atoms()
    act_mol = aom.get_active_mols()

    # Set which atoms are active in the sim
    all_settings['active_atoms'] = np.array(list(range(all_settings['coords'].shape[1])))
    all_settings['active_mols'] = np.array(list(range(all_settings['nmol'])))

    # Group atoms by molecule
    # mol_info = at_num -> mol_num
    all_settings['mol_info'] = {i: int(i // all_settings['atoms_per_site'])
                                for i in all_settings['active_atoms']}
    # reversed_mol_info = mol_num -> at_nums
    all_settings['reversed_mol_info'] = {}
    for at_i, mol_i in all_settings['mol_info'].items():
        all_settings['reversed_mol_info'].setdefault(mol_i, []).append(at_i)
    # active_atom_index = mol_num -> active at nums (AOM)
    all_settings['active_atom_index'] = {mol_num: [j for j in at_nums if abs(aom[j].coeff) > 1e-5]
                                         for mol_num, at_nums in all_settings['reversed_mol_info'].items()}


def handle_decomp_file(all_settings):
    """Will read the DECOMP.inp file if it exists. This will also set the active mols and atoms."""
    # Get active mols
    if all_settings['decomp_file'].is_file():
        DF = i_IO.DecompFile(all_settings['decomp_file'])
        assert DF.num_active_ats == (all_settings['nmol'] * all_settings['atoms_per_site'])
        all_settings['active_mol'] = DF.active_mols

    else:
        all_settings['active_mol'] = set(range(all_settings['nmol']))

    # Set active mols
    if all_settings['do_transition_state']:
        all_settings['LUMO'].set_active_mol(all_settings['active_mol'])
        all_settings['HOMO'].set_active_mol(all_settings['active_mol'])
    else:
        all_settings['AOM'].set_active_mol(all_settings['active_mol'])

