from src import type as typ
from src import text as txt_lib
from src import IO as io
from src import math as MT
from src import consts
from src import EXCEPT as EXC

from Templates import defaults as dft
try:
    from Templates import permanent_settings as ps
except:
    EXC.replace_perm_settings()
    from Templates import permanent_settings as ps

from init import settings_file

import numpy as np
import datetime
import difflib
import os

## TODO: Need to finish using all the consts.py folderpaths instead of declaring them here.


# Will declare all the paths that are required in the code
def init_output_files_and_folders(all_settings):
    all_settings['img_fold']      = io.folder_correct(consts.img_folderpath)
    all_settings['data_fold']     = io.folder_correct(consts.data_folderpath)
    all_settings['tmplte_fold'] = io.folder_correct(consts.template_folderpath)
    all_settings['f.txt'] = io.folder_correct('./f.txt', True)
    all_settings['graph_files'] = []
    all_settings['vmd_script_folder'] = io.folder_correct('./src/TCL/', True)
    all_settings['vmd_junk'] = {}
    all_settings['vmd_script'] = {}
    all_settings['vmd_err'] = {}
    all_settings['vmd_temp'] = io.folder_correct(all_settings['tmplte_fold']+"VMD_TEMP.vmd")
    all_settings['bin_fold'] = io.folder_correct('./bin/')
    all_settings['ffmpeg_bin'] = io.folder_correct(all_settings['bin_fold']+'ffmpeg')
    all_settings['delete_these'] = []
    all_settings['vmd_log_file'] = io.folder_correct("./visualisation.log")
    all_settings['tcl'] = {}
    all_settings['tcl']['vmd_source_file'] = io.folder_correct("%sinclude.vmd"%all_settings['tmplte_fold'])
    all_settings['ps_filepath'] = "%spermanent_settings.py"%all_settings['tmplte_fold']

    init_rep_files(all_settings)
    use_fuzzy_files = typ.translate_to_bool(all_settings['find_fuzzy_files'], 'fuzzy_find_files')
    if use_fuzzy_files:
        all_settings['CP2K_output_files'] = io.fuzzy_file_find(all_settings['path'])
    if all_settings['all_reps'] == True:
        all_settings['CP2K_output_files'] = {ftyp:[io.folder_correct(all_settings['path'] + f) for f in all_settings['CP2K_output_files'][ftyp]] for ftyp in all_settings['CP2K_output_files']}
    else:
        all_settings['CP2K_output_files'] = {ftyp:[io.folder_correct(all_settings['path'] + f) for i,f in enumerate(all_settings['CP2K_output_files'][ftyp]) if i in all_settings['num_reps']]
                                                   for ftyp in all_settings['CP2K_output_files']}

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

# Will initialise the permanent settings (check tachyon path, read and write permanent settings, set prev_calibrate etc...)
def init_permanent_settings(all_settings):
    # Create the docs if they haven't already been created
    if not ps.created_docs:
       os.system("python3 Create_docs.py")
       io.read_write_perm_settings(all_settings['ps_filepath'], "created_docs", True)
    # Save the previous runtime
    io.read_write_perm_settings(all_settings['ps_filepath'], "previous_runtime",
                datetime.datetime.strftime(datetime.datetime.now(), ps.time_format))
    # Checking Tachyon Renderer Path
    new_tachyon_path = io.find_tachyon(ps.tachyon_path)
    if new_tachyon_path != ps.tachyon_path:
        io.read_write_perm_settings(all_settings['ps_filepath'], "tachyon_path", new_tachyon_path)
        tachyon_path = new_tachyon_path
    else:
        tachyon_path = ps.tachyon_path
    all_settings['tcl']['tachyon_path'] = tachyon_path
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
    if all_settings['title'] in os.listdir(all_settings['img_fold']):
        all_steps = range(all_settings['start_step'], all_settings['end_step'], all_settings['stride'])
        img_folderpath = io.folder_correct(all_settings['img_fold']+all_settings['title'])
        tga_files = [i.replace(",",".") for i in os.listdir(img_folderpath)]
        completed_timesteps = ['%.2f'%float(i[:i.find('_')]) for i in tga_files if typ.is_num(i[:i.find('_')])]
        ltxt = io.open_read(all_settings['CP2K_output_files']['coeff'][0]).split('\n')
        max_step = int(len(ltxt)/all_settings['coeff_metadata']['lines_in_step'])
        timesteps = ["%.2f"%float(txt_lib.string_between(ltxt[all_settings['coeff_metadata']['lines_in_step']*i + all_settings['coeff_metadata']['time_ind']], "time = ", all_settings['coeff_metadata']['time_delim'])) for i in range(max_step)]
        return [i for i in all_steps if all_steps[i] in completed_timesteps]
    else:
        return []

# Determines the correct all_settings['rotation'] vector to use
def init_rotation(all_settings):
    if type(all_settings['rotation']) != list and all(type(i) != int or type(i) != float for i in all_settings['rotation']):
        auto_rot, _, _ = txt_lib.fuzzy_variable_translate(all_settings['rotation'], ['auto','A list containing [Rotx, Roty, Rotz]', 'no'], all_settings['verbose_output'], False)
        if auto_rot:
            #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
            all_settings['rotation'] = MT.find_auto_rotation(all_settings=all_settings)
        if type(all_settings['rotation']) == str and ('n' in all_settings['rotation'].lower() or bool(all_settings['rotation']) == False):
            all_settings['rotation'] = [0,0,0]

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
    all_settings['tcl']['backgrnd_mols'] = ""
    if all_settings['background_mols']:
        all_settings['tcl']['bckg_mols_on_off'] = ''
    else:
        all_settings['tcl']['bckg_mols_on_off'] = '#'
    if all_settings['show_box']:
       all_settings['tcl']['iso_type']  = 2
    all_settings['tcl'] = txt_lib.tcl_3D_input(all_settings['background_color'], ['R','G','B'], all_settings['tcl'], "backgrnd_")
    all_settings['tcl'] = txt_lib.tcl_3D_input(all_settings['translate_by'], ['x','y','z'], all_settings['tcl'], "trans_")
    all_settings['tcl'] = txt_lib.tcl_3D_input([0,0,0], ['x','y','z'], all_settings['tcl'], "time_lab_")
    all_settings['tcl'] = txt_lib.tcl_3D_input(all_settings['rotation'], ['x','y','z'], all_settings['tcl'], "rot")
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

# Finds the aom dictionary
def init_AOM_D(all_settings):
    _, all_settings['AOM_D'], tmp = io.AOM_coeffs(all_settings['CP2K_output_files']['AOM'][0], all_settings['atoms_per_site'])
    all_settings['mol_info'] = tmp
    if not tmp:
        all_settings['mol_info'] = {i:int(i/all_settings['atoms_per_site']) for i in all_settings['AOM_D']} #converts atom number to molecule number
    # All the active molecules (according to the AOM_COEFFICIENT.include file)
    all_settings['active_mols'] = [(i,all_settings['AOM_D'][i][1]) for i in all_settings['AOM_D']]
    all_settings['AOM_D'] = {i:all_settings['AOM_D'][i] for i in all_settings['AOM_D'] if np.abs(all_settings['AOM_D'][i][0]) > 0} # Removing inactive atoms from all_settings['AOM_D']

# Will initialise the molecules that are needed to be highlighted
def init_mols_to_plot(all_settings):
    if type(all_settings['mols_to_highlight']) == int:
       all_settings['mols_to_highlight'] = [all_settings['mols_to_highlight']]
    if type(all_settings['mols_to_highlight']) == 'str':
       if np.any(i in all_settings['mols_to_highlight'].lower() for i in ['max','min']) :
          all_settings['mols_to_highlight'] = all_settings['mols_to_highlight'].lower()
       else:
          EXC.WARN("The variable 'mols_to_highlight' was declared incorrectly. Valid Options are:\n\t'max'\n\tmin\n\tinteger\n\trange.")

# Will retrieve the metadata for the coefficient, position and pvecs file.
# The metadata is needed to decide which steps to complete when running the simulation
def get_all_files_metadata(all_settings):
    all_settings['pos_metadata']   = io.get_xyz_step_metadata(all_settings['CP2K_output_files']['pos'][0])
    all_settings['pvecs_metadata'] = io.get_xyz_step_metadata(all_settings['CP2K_output_files']['pvecs'][0])
    all_settings['coeff_metadata'] = io.get_xyz_step_metadata(all_settings['CP2K_output_files']['coeff'][0])

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
    density, real_phase, full_phase = txt_lib.fuzzy_variable_translate(all_settings['type_of_wavefunction'], ["Density", "Real phase", "Phase"], all_settings['verbose_output'])
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
    run_inp = open(all_settings['CP2K_output_files']['inp'][0]).read()
    all_settings['atoms_per_site'] = int(txt_lib.inp_keyword_finder(run_inp, "ATOMS_PER_SITE")[0])
    ndiab_states = int(txt_lib.inp_keyword_finder(run_inp, "NUMBER_DIABATIC_STATES")[0])
    norbits = int(txt_lib.inp_keyword_finder(run_inp, "NUMBER_ORBITALS")[0])
    all_settings['nmol'] = int(ndiab_states/norbits)

# Will find which steps to ignore
def init_global_steps_to_ignore(all_settings):
    all_settings['global_steps_to_ignore'] = []
    if all_settings['restart_vis'] and not all_settings['calibrate']:
        all_settings['global_steps_to_ignore'] = find_ignoring_steps(all_settings)

# Will find which atoms should be plotted
def init_atoms_to_plot(all_settings):
    #Translate the input file
    all_settings['ignore_inactive_atoms'] = typ.translate_to_bool(all_settings['ignore_inactive_atoms'],'ignore_inactive_atoms')
    all_settings['background_mols']       = typ.translate_to_bool(all_settings['background_mols'],'background_mols')
    all_settings['ignore_inactive_mols']  = typ.translate_to_bool(all_settings['ignore_inactive_mols'], 'ignore_inactive_mols')
    plot_all_atoms, plot_min_active, plot_auto_atoms = [False]*3
    if type(all_settings['atoms_to_plot']) == str:
        plot_all_atoms, plot_min_active, plot_auto_atoms,_ ,_ = txt_lib.fuzzy_variable_translate(all_settings['atoms_to_plot'], ["all","min_active",'auto',"A list containing atom indices (or range or xrange)","An integer" ], all_settings['verbose_output'], False)
    # Deciding which atoms to plot
    if type(all_settings['atoms_to_plot']) == int:
        all_settings['atoms_to_plot'] = [all_settings['atoms_to_plot']]

    pop_indices = np.array([np.arange(len(all_settings['pops'][0])) for i in range(len(all_settings['pops']))])
    plottable_pop_indices = pop_indices[all_settings['pops'] > all_settings['min_abs_mol_coeff']]
    all_settings['max_act_mol'] = np.max(plottable_pop_indices) + 1
    if plot_all_atoms:
       all_settings['atoms_to_plot'] = range(len(all_settings['coords'][0]))
    elif plot_min_active:
        min_plotted_coeff = np.min([min(np.arange(0,all_settings['nmol'])[np.abs(i)**2 > all_settings['min_abs_mol_coeff']]) for i in all_settings['mol']])
        all_settings['atoms_to_plot'] = range(min_plotted_coeff*all_settings['atoms_per_site'], all_settings['max_act_mol']*all_settings['atoms_per_site'])
    elif plot_auto_atoms:
        all_settings['atoms_to_plot'] = range(0, all_settings['max_act_mol']*all_settings['atoms_per_site'])
    else: # Can add a test here for all_settings['atoms_to_plot'] type... (should be list -could convert int but not float etc..)
       if not (type(all_settings['atoms_to_plot']) == list or type(all_settings['atoms_to_plot']) == type(xrange(1))):
          EXC.ERROR("Sorry the variable 'atoms_to_plot' seems to be in an unfamiliar format (please use a list, an xrange or an integer).\n\nCurrent all_settings['atoms_to_plot'] type is:\t%s"%str(type(all_settings['atoms_to_plot'])))
       all_settings['atoms_to_plot'] = [i for i in all_settings['atoms_to_plot'] if i < len(all_settings['mol_info'])]
       if len(all_settings['atoms_to_plot']) == 0:
          EXC.ERROR('NO DATA PLOTTED, THERE ARE NO ATOMS CURRENTLY PLOTTED. PLEASE CHECK THE VARIABLE "atoms_to_plot"')
       all_settings['AOM_D'] = {i:all_settings['AOM_D'][i] for i in all_settings['AOM_D'] if all_settings['AOM_D'][i][1] in all_settings['atoms_to_plot']}

    poss_atoms = [i for i, elm in enumerate(all_settings['at_num']) if elm not in all_settings['atoms_to_ignore'] ]

    if all_settings['ignore_inactive_mols']:
      all_settings['atoms_to_plot'] = [i[0] for i in all_settings['active_mols'] if i[1] in all_settings['atoms_to_plot']]
    all_settings['mol_info'] = {i:all_settings['mol_info'][i] for i in all_settings['mol_info'] if i in all_settings['AOM_D'].keys()}
    active_atoms = [all_settings['AOM_D'][i][1] for i in all_settings['AOM_D']]
    if all_settings['ignore_inactive_atoms']:
        all_settings['atoms_to_plot'] = [i for i in all_settings['AOM_D'] if all_settings['AOM_D'][i][1] in all_settings['atoms_to_plot']]
    all_settings['atoms_to_plot'] = [i for i in all_settings['atoms_to_plot'] if i in poss_atoms]
    all_settings['active_atoms_index'] = [find_value_dict(all_settings['mol_info'],i) for i in range(all_settings['nmol'])]

# Finds list of keys attached to a value in a dictionary
def find_value_dict(D, value):
     return [i for i in D if D[i] == value]

# Will check if the charge is spread too much and report back to the user
def check_charge_spread(all_settings):
    all_settings['num_mols_active'] = len([i for i in  all_settings['active_atoms_index'] if i])
    if all_settings['max_act_mol'] > all_settings['num_mols_active']:
        cont = raw_input("The charge will eventually leave the number of plotted molecules in this simulation.\n\nMax molecule the charge reaches = %i.\n\nAre you sure you want to continue [y/n]:\t"%all_settings['max_act_mol'])
        if not typ.translate_to_bool(cont, 'cont'):
            raise SystemExit("\n\nOk, exiting so you can change the settings file.\nI suggest using the keyword:\n\nall_settings['atoms_to_plot'] = 'auto'")
        else:
            print("Okie doke, you will get empty molecules though!")
    all_active_coords = all_settings['coords'][:,[i[0] for i in all_settings['active_mols']]]

# Will create the storage containers to store the timings
def init_times_dict(all_settings):
    all_settings['init_times'] = {}
    num_steps = len(all_settings['coords'])
    all_settings['times'] = {'Create Wavefunction':np.zeros(num_steps),
             'Create Cube Data': np.zeros(num_steps),
             'Write Cube File':np.zeros(num_steps),
             'VMD Visualisation':np.zeros(num_steps),
             'Plot and Save Img':np.zeros(num_steps)}
    all_settings['times_taken'] = []

# Will initialise the start_step, end_step and stride variables
def find_step_numbers(all_settings):
    if not all_settings['calibrate']:
        all_steps = txt_lib.fuzzy_variable_translate(all_settings['end_step'],
                                                                ["all","An Integer end step"],
                                                                all_settings['verbose_output'],
                                                                False)
        if all_steps:
            all_settings['end_step'] = all_settings['pos_metadata']['nsteps']
    else:
        all_settings['start_step'] = all_settings['calibration_step']
        all_settings['end_step']   = all_settings['calibration_step'] + 1
        all_settings['stride']     = 1
    all_settings['max_step'] = np.max([all_settings['pos_metadata']['nsteps'],
                                       all_settings['coeff_metadata']['nsteps'],
                                       all_settings['pvecs_metadata']['nsteps']])

# Will initialise which timesteps to carry out based on which steps are available
# E.g. if nuclear timesteps = [1,2,3,6] and coeff timesteps = [1,2,3,4,5,6] we would carry out [1,2,3,6]...
def init_local_steps_to_ignore(all_settings):
    all_nucl_steps = np.arange(all_settings['start_step'], all_settings['end_step'], all_settings['stride'])

    # Read in which timesteps are available
    n_avail_dt = all_settings['pos_metadata']['tsteps'][all_nucl_steps]
    c_avail_dt = all_settings['coeff_metadata']['tsteps']
    p_avail_dt = all_settings['pvecs_metadata']['tsteps']

    # Create some arrays to store the steps we are going to want to ignore
    nucl_to_ignore = all_nucl_steps[:]
    all_coef_steps, coef_to_ignore = np.arange(len(c_avail_dt)), np.arange(len(c_avail_dt))
    all_pvec_steps, pvec_to_ignore = np.arange(len(p_avail_dt)), np.arange(len(p_avail_dt))

    # First remove any nuclear timesteps not found in coeffs or pvecs
    nc_mask = [i in c_avail_dt for i in n_avail_dt]
    all_nucl_steps = all_nucl_steps[nc_mask]
    np_mask = [i in p_avail_dt for i in n_avail_dt]
    all_nucl_steps = all_nucl_steps[nc_mask]
    if len(all_nucl_steps) == 0: raise SystemExit("Sorry I can't seem to find any timesteps that overlap between the nuclear, coefficients and pvecs!")

    # We can now remove any steps in the coefficients array that aren't found elsewhere
    cp_mask = [i in p_avail_dt for i in c_avail_dt]
    all_coef_steps = all_coef_steps[cp_mask]
    cn_mask = [i in n_avail_dt for i in c_avail_dt]
    all_coef_steps = all_coef_steps[cn_mask]
    if len(all_coef_steps) == 0: raise SystemExit("Sorry I can't seem to find any timesteps that overlap between the nuclear, coefficients and pvecs!")

    #Finally do the same for pvecs
    pc_mask = [i in c_avail_dt for i in p_avail_dt]
    all_pvec_steps = all_pvec_steps[pc_mask]
    pn_mask = [i in n_avail_dt for i in p_avail_dt]
    all_pvec_steps = all_pvec_steps[pn_mask]
    if len(all_pvec_steps) == 0: raise SystemExit("Sorry I can't seem to find any timesteps that overlap between the nuclear, coefficients and pvecs!")

    # Finally save any steps to ignore in arrays to be read later
    all_settings['c_steps_to_ignore'] = np.array([i for i in coef_to_ignore if i not in all_coef_steps ]+list(all_settings['global_steps_to_ignore']))
    all_settings['p_steps_to_ignore'] = np.array([i for i in pvec_to_ignore if i not in all_pvec_steps ]+list(all_settings['global_steps_to_ignore']))
    if not all_settings['calibrate']:
        all_settings['n_steps_to_ignore'] = np.array([i for i in nucl_to_ignore if i not in all_nucl_steps ]+list(all_settings['global_steps_to_ignore']))
    else:
        all_settings['n_steps_to_ignore'] = [i for i in range(all_settings['pos_metadata']['nsteps']) if i != all_nucl_steps[0]]


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
