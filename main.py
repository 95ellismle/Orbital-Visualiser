"""Will import the python3 print function."""
from __future__ import print_function
from src import EXCEPT as EXC


"""
 The program that calls all the necessary functions and actually runs the code.

 This will initialise the code by reading everything, write the spherical
 harmonic data as a cube file, visualise the cube file with VMD and save them,
 then stitch them all together.
"""

"""A function to import modules and check if they exist."""
def import_and_check(str_lib, error=True):
  try:
    i = __import__(str_lib)
    return i
  except ImportError:
    if error:
        EXC.ERROR("You need the library named '%s' to run this program please install it!\n\n\t* If you are using conda use the command conda install <lib>\n\n\t* If you are using pip use the command sudo pip install <lib>\n\netc..."%str_lib)
        return None
    else:
        return False

sys  = import_and_check("sys")
datetime = import_and_check("datetime")
time = import_and_check('time')
os   = import_and_check('os')
np   = import_and_check('numpy')
subprocess = import_and_check('subprocess')
insp = import_and_check("inspect", False)
dfl  = import_and_check("difflib")
mp   = import_and_check("multiprocessing")
coll = import_and_check("collections")



if insp != False:
    # Returns the current line number
    def lineno():
        return insp.currentframe().f_back.f_lineno

from src import IO as io
from src import text as txt_lib
from src import type as typ
from src import math as MT
from src import Warnings as WRN
#from src import Colour as col
from src import geometry as geom
from Templates import defaults as dft
import Templates.permanent_settings as ps

if sys.version_info[0] > 2:
    xrange = range
    raw_input = input

##################################################################################################
# Reading the settings file
settings_file = io.folder_correct("./settings.inp")
orig_settings_ltxt = io.open_read(settings_file).split('\n')
settings_ltxt = io.settings_read(settings_file)

clean_settings_dict = {i.split('=')[0].replace(' ',''):'='.join(i.split('=')[1:]) for i in settings_ltxt}
clean_settings_dict = {i:eval(clean_settings_dict[i]) for i in clean_settings_dict}
clean_settings_dict = {i: np.array(clean_settings_dict[i]) if type(clean_settings_dict[i]) == list else clean_settings_dict[i] for i in clean_settings_dict}
orig_settings = coll.OrderedDict()
setting_names = []
count = 2
for i in orig_settings_ltxt:
    if i == '':
       orig_settings['x'*count] = ''
       count += 1
    if txt_lib.comment_remove(i) == '' and '#' in i:
       orig_settings[i] = 'cmt'
    else:
       line_split = i.split('=')
       setting_name = line_split[0].replace(' ','')
       setting_names.append(setting_name)
       orig_settings[setting_name] = '='.join(line_split[1:])


#Reading the settings file. If the settings aren't there check for typos. If that isn't ok use defaults.
setting_file_settings = []
replacer_settings = []
for line in settings_ltxt:
    line = txt_lib.setting_typo_check(line, np.array(list(dft.defaults.keys())), setting_file_settings, replacer_settings)
    try:
        exec(line)
    except:
        print("Dodgy line in settings.inp:  %s"%line)

dodgy_vars = []
for default_var in dft.defaults:
    if default_var not in globals():
        if type(dft.defaults[default_var]) == str:
            dft.defaults[default_var] = '"%s"'%dft.defaults[default_var]
        dodgy_vars.append((default_var,dft.defaults[default_var]))
        exec('%s=%s'%(str(default_var), str(dft.defaults[default_var])) )


#################################################################################################

step_info = {}

calibrate = typ.translate_to_bool(calibrate, "calibrate")
if calibrate:
   step_info['Title'] = "Calibration"
   print("In Calibration Mode...")
else:
   if Title == None:
      step_info['Title'] = raw_input("What should I call this visualisation?\n")             #can be anything will be ignored by VMD.
      print("\n\n")
   else:
      step_info['Title'] = str(Title)
START_TIME = time.time()


#TO DO:
#
# * Improve Parallelism by using subprocess.Popen() and threading when opening VMD, so it opens a new instance of VMD on a new thread.
#
# * Default the step_info['mol'] to (1,0,0,0,0,...,0)
#
# * Make it work for files where the number of pvecs steps isn't equal to the number of position steps. (Will be fixed with fixed pvecs)
#    Remove bad steps from the simulation data
#    ** Calculate own pvecs. (nearly)
#
# * Add crash report data so other users can tell me exactly why it crashed and I will be able to fix it.
#      Add more checks and debugging info (better error messages etc).
#
# * Investigate the maximum radius for a bunch of atoms with varying step_info['mol']. The max radii
#     should be proportional (hopefully linearly) to the mol_coeff. If so this relationship can be
#     used to find an optimised abs_min_mol_coeff at the first time-step. That is at the first
#     time-step the abs_min_mol_coeff can be set to the max mol_coeff to produce an isosurface with
#     a radius less than or equal to that of the atom (or just a certain value).


# I should definitely break up the bits that aren't actually used within the step here!
#output folders
step_info['any_extra_raw_tcl_code'] = ''
step_info['img_fold']      = io.folder_correct('./img/')
step_info['data_fold']     = io.folder_correct('./data/')
step_info['tmplte_fold'] = io.folder_correct('./Templates')
step_info['clean_settings_dict'] = clean_settings_dict
step_info['settings_file']   = settings_file
step_info['orig_settings']   = orig_settings
step_info['defaults']         = dft.defaults
# Misc required step data
step_info['min_molC'] = min_abs_mol_coeff
step_info['length_of_animation'] = length_of_animation
step_info['verbose'] = verbose_output
step_info['atoms_to_plot']  = atoms_to_plot
step_info['background_mols_end_extend'] = background_mols_end_extend
step_info['bounding_box'] = bounding_box_scale
step_info['res'] = resolution
step_info['to_stitch'] = ''
step_info['time_lab_txt'] = time_label_text
step_info['f.txt'] = io.folder_correct('./f.txt', True)
step_info['graph_files'] = []
step_info['vmd_timeout'] = vmd_timeout
step_info['vmd_script_folder'] = io.folder_correct('./src/TCL/', True)
step_info['vmd_junk'] = {}
step_info['vmd_script'] = {}
step_info['vmd_err'] = {}
step_info['vmd_temp'] = io.folder_correct(step_info['tmplte_fold']+"VMD_TEMP.vmd")
step_info['bin_fold'] = io.folder_correct('./bin/')
step_info['ffmpeg_bin'] = io.folder_correct(step_info['bin_fold']+'ffmpeg')
step_info['delete_these'] = []
step_info['yfont'] = yfontsize
step_info['img_format'] = img_format
step_info['xfont'] = xfontsize
step_info['ylab'] = ylabel
step_info['xlab'] = xlabel
step_info['graph_title'] = graph_title
step_info['tfont'] = title_fontsize
step_info['max_graph_data'] = max_data_in_graph
step_info['graph_img_ratio'] = graph_to_vis_ratio
step_info['highlighted_mols'] = mols_to_highlight
step_info['img_prefix'] = 'img'
step_info['mols_plotted'] = ''
step_info['vmd_log_file'] = io.folder_correct("./visualisation.log")
step_info['tcl'] = {}
step_info['tcl']['vmd_source_file'] = io.folder_correct("%sinclude.vmd"%step_info['tmplte_fold'])
permanent_settings_filepath = "%spermanent_settings.py"%step_info['tmplte_fold']
# Create the Docs on the computer this has been downloaded on.
if not ps.created_docs:
   os.system("python Create_docs.py")
   io.read_write_perm_settings(permanent_settings_filepath, "created_docs", True)

if 'path' not in globals().keys():
    if insp != False:
        line_number = lineno()
        EXC.ERROR("Can't find the path variable! Please set it in settings.inp", line_number)
    else:
        EXC.ERROR("Can't find the path variable! Please set it in settings.inp")
path = io.folder_correct(path)
step_info['path'] = path

if not io.path_leads_somewhere(path):
   sys.exit("\nThe specified path doesn't lead anywhere:\n%s\n\n\t Where is my data?! "%path)

pi = np.pi
ang2bohr = 1.88971616463207
bohr2ang = 0.52918

# Translate the input commands to a boolean (logical)
keep_cube_files = typ.translate_to_bool(keep_cube_files,'keep_cube_files')
ignore_inactive_atoms = typ.translate_to_bool(ignore_inactive_atoms,'ignore_inactive_atoms')
step_info['background_mols'] = typ.translate_to_bool(background_mols,'background_mols')
ignore_inactive_mols =  typ.translate_to_bool(ignore_inactive_mols, 'ignore_inactive_mols')
keep_img_files  = typ.translate_to_bool(keep_img_files,'keep_img_files')
keep_tga_files  = typ.translate_to_bool(keep_tga_files,'keep_tga_files')
step_info['verbose']  = typ.translate_to_bool(step_info['verbose'], 'verbose')
draw_time      = typ.translate_to_bool(draw_time, "draw_time")
show_img_after_vmd  = typ.translate_to_bool(show_img_after_vmd, 'show_img_after_vmd')
step_info['load_in_vmd'] = typ.translate_to_bool(load_in_vmd, 'load_in_vmd')
step_info['side_by_side_graph'] = typ.translate_to_bool(side_by_side_graph, 'side_by_side_graph')
show_box = typ.translate_to_bool(show_box ,'show_box')
if step_info['side_by_side_graph']:
   plt = import_and_check("matplotlib.pyplot", error=False)
   plt = plt.pyplot
   if not plt:
      EXC.WARN("You don't seem to have the python library matplotlib.pyplot installed...\n\nI can't plot the graph next to the visualisation.")
      step_info['side_by_side_graph'] = False
if not calibrate:
    step_info['load_in_vmd'] = False

if not io.use_prev_scaling(path) and os.path.isfile(step_info['tcl']['vmd_source_file']):
    os.remove(step_info['tcl']['vmd_source_file'])

io.read_write_perm_settings(permanent_settings_filepath, "previous_runtime",
            datetime.datetime.strftime(datetime.datetime.now(), ps.time_format))
io.read_write_perm_settings(permanent_settings_filepath, "previous_path", path)
if calibrate:
    io.read_write_perm_settings(permanent_settings_filepath, "previous_calibrate", True)
else:
    io.read_write_perm_settings(permanent_settings_filepath, "previous_calibrate", False)

step_info['files_to_keep'] = []
if keep_cube_files:
    step_info['files_to_keep'].append("cube")
if keep_img_files:
    step_info['files_to_keep'].append("img")
if keep_tga_files:
    step_info['files_to_keep'].append("tga")

WRN.default_variable_warn(dodgy_vars, step_info)
WRN.typo_variable_warn(replacer_settings, setting_file_settings, step_info)


if type(mols_to_highlight) == int:
   mols_to_highlight = [mols_to_highlight]
if type(mols_to_highlight) == 'str':
   if np.any(i in mols_to_highlight.lower() for i in ['max','min']) :
      mols_to_highlight = mols_to_highlight.lower()
   else:
      EXC.WARN("The variable mols_to_highlight was declared incorrectly. Valid Options are:\n\t'max'\n\tmin\n\tinteger\n\trange.")

density, real_phase, full_phase = txt_lib.fuzzy_variable_translate(type_of_wavefunction, ["Density", "Real phase", "Phase"], step_info['verbose'])
use_gif, use_mp4 = txt_lib.fuzzy_variable_translate(movie_format, ["GIF", "MP4"], step_info['verbose'])
auto_time_label = txt_lib.fuzzy_variable_translate(pos_time_label, ["auto","A list containing absolute positions i.e. [Pos_x, Pos_y, Pos_z]"], step_info['verbose'], False)
plot_all_atoms, plot_min_active, plot_auto_atoms = [False]*3
if type(step_info['atoms_to_plot']) == str:
    plot_all_atoms, plot_min_active, plot_auto_atoms,_ ,_ = txt_lib.fuzzy_variable_translate(step_info['atoms_to_plot'], ["all","min_active",'auto',"A list containing atom indices (or range or xrange)","An integer" ], step_info['verbose'], False)
CP2K_output_files = {i:io.folder_correct(path + CP2K_output_files[i]) for i in CP2K_output_files}

if use_mp4:
    step_info['movie_format'] = 'mp4'
if use_gif:
    step_info['movie_format'] = 'gif'

if density:
    step_info['colour_type'] = 'density'
elif real_phase:
    step_info['colour_type'] = 'real-phase'
elif full_phase:
    step_info['colour_type'] = 'phase'
else:
    EXC.WARN("Sorry I'm not sure what type of colour I should use, defaulting to %s"%dft.defaults['type_of_wavefunction'])
    step_info['colour_type'] = dft.defaults['type_of_wavefunction']

if (type(step_info['bounding_box']) == int) or (type(step_info['bounding_box']) == float):
   step_info['bounding_box'] = [step_info['bounding_box']]*3
if type(step_info['bounding_box']) != list and step_info['verbose']:
   EXC.WARN("The step_info['bounding_box'] variable doesn't seem to be set correctly! \nCorrect options are:\n\t* integer or float\n\tlist of ints or floats (for x,y,z dimensions)")
pvecs_on = io.path_leads_somewhere(CP2K_output_files['pvecs'])
if not pvecs_on:
    EXC.WARN("Can't find the Pvecs file, this contains information about the orientation of the P orbitals.")

if type(end_step) != int:
    if type(end_step) == str:
        if 'al' not in end_step.lower():
            EXC.WARN("Sorry the end_step variable needs to be an integer not a %s!\n\nConverting from %.2g to %i"%(type(end_step),end_step,int(round(end_step))))
            end_step = int(round(end_step))
    else:
        EXC.WARN("Sorry the end_step variable needs to be an integer not a %s!\n\nConverting from %.2g to %i"%(type(end_step),end_step,int(round(end_step))))
        end_step = int(round(end_step))

io.create_data_img_folders(step_info)
keep_tga_files = WRN.redundant_img(keep_img_files, keep_tga_files)


# Reading the inp file and finding useful data from it
run_inp = open(CP2K_output_files['inp']).read()
atoms_per_site = int(txt_lib.inp_keyword_finder(run_inp, "ATOMS_PER_SITE")[0])
ndiab_states = int(txt_lib.inp_keyword_finder(run_inp, "NUMBER_DIABATIC_STATES")[0])
norbits = int(txt_lib.inp_keyword_finder(run_inp, "NUMBER_ORBITALS")[0])
nmol = int(ndiab_states/norbits)
#natoms = int(txt_lib.inp_keyword_finder(open(CP2K_output_files['basis'],'r').read(), 'NUMBER_OF_ATOMS')[0])

_, step_info['AOM_D'], tmp = io.AOM_coeffs(CP2K_output_files['AOM'], atoms_per_site)
step_info['mol_info'] = tmp
if not tmp:
    step_info['mol_info'] = {i:int(i/atoms_per_site) for i in step_info['AOM_D']} #converts atom number to molecule number

# All the active molecules (according to the AOM_COEFFICIENT.include file)
step_info['active_mols'] = [(i,step_info['AOM_D'][i][1]) for i in step_info['AOM_D']]

if calibrate:
    start_step = calibration_step
    end_step = calibration_step+1
    stride = 1

# Reading the nuclear positions
step_info['coords'], step_info['at_num'], step_info['Ntime-steps']  = io.read_xyz_file(CP2K_output_files['xyz'], 3, start_step, end_step, stride)
step_info['at_num'] = [i.flatten() for i in step_info['at_num']]
num_atoms = len(step_info['coords'][0])
step_info['coords'] = step_info['coords']*ang2bohr
print("Finished reading coords -%i steps"%len(step_info['coords']))

# Reading the mol coeffs
Fmol_coeffs, _, step_info['Mtime-steps'] = io.read_xyz_file(CP2K_output_files['mol_coeff'],2, start_step, 'all', stride)
if calibrate:
   mol_cal_index = np.arange(len(step_info['Mtime-steps']))[step_info['Mtime-steps'] == step_info['Ntime-steps'][0]]
   Fmol_coeffs = Fmol_coeffs[mol_cal_index]
   step_info['Mtime-steps'] = step_info['Mtime-steps'][mol_cal_index]
if time_step == 'n':
    matching_indices = [i for i in xrange(len(step_info['Mtime-steps'])) if step_info['Mtime-steps'][i] in step_info['Ntime-steps']]
    step_info['mol'] = np.array([Fmol_coeffs[i] for i in matching_indices])[:,:,1:]
    step_info['mol'] = np.array([[complex(*j) for j in i] for i in Fmol_coeffs])
    step_info['mci'] = np.arange(len(step_info['Mtime-steps']))
    mask      = [i in step_info['Ntime-steps'] for i in step_info['Mtime-steps']]
    step_info['mci'] = step_info['mci'][mask]
    step_info['aci'] = np.arange(len(step_info['Ntime-steps']))

if time_step == 'e':
    step_info['mol'] = np.array(Fmol_coeffs)[:,:,1:]
    step_info['mol'] = np.array([[complex(*j) for j in i] for i in Fmol_coeffs])
    step_info['aci'] = np.zeros(len(step_info['Mtime-steps']))
    step_info['mci'] = range(len(step_info['Mtime-steps']))
    index = 0
    for i in xrange(len(step_info['Mtime-steps'])):
        if index < len(step_info['Ntime-steps'])-1:
            if step_info['Mtime-steps'][i] == step_info['Ntime-steps'][index]:
                step_info['aci'][i] = index
            elif step_info['Mtime-steps'][i] > step_info['Ntime-steps'][index] and step_info['Mtime-steps'][i] < step_info['Ntime-steps'][index+1]:
                step_info['aci'][i] = index
            elif (step_info['Mtime-steps'][i] > step_info['Ntime-steps'][index+1] and step_info['Mtime-steps'][i] < step_info['Ntime-steps'][index+2]) or step_info['Mtime-steps'][i] == step_info['Ntime-steps'][index+1]:
                index += 1
                step_info['aci'][i] = index
        elif index >= len(step_info['Ntime-steps']) - 1:
            if step_info['Mtime-steps'][i] == step_info['Ntime-steps'][index]:
                step_info['aci'][i] = index
            elif step_info['Mtime-steps'][i] > step_info['Ntime-steps'][index]:
                    step_info['aci'][i] = index
step_info['aci'] = [int(i) for i in step_info['aci']]

step_info['AOM_D'] = {i:step_info['AOM_D'][i] for i in step_info['AOM_D'] if np.abs(step_info['AOM_D'][i][0]) > 0} # Removing inactive atoms from step_info['AOM_D']
# Deciding which atoms to plot
if type(step_info['atoms_to_plot']) == int:
    step_info['atoms_to_plot'] = [step_info['atoms_to_plot']]
max_active_mol = np.max([max(np.arange(0,nmol)[np.abs(i)**2 > step_info['min_molC']]) for i in step_info['mol']])+1
if plot_all_atoms:
   step_info['atoms_to_plot'] = range(len(step_info['coords'][0]))
elif plot_min_active:
    min_plotted_coeff = np.min([min(np.arange(0,nmol)[np.abs(i)**2 > step_info['min_molC']]) for i in step_info['mol']])
    step_info['atoms_to_plot'] = range(min_plotted_coeff*atoms_per_site, max_active_mol*atoms_per_site)
elif plot_auto_atoms:
    step_info['atoms_to_plot'] = range(0, max_active_mol*atoms_per_site)
else: # Can add a test here for step_info['atoms_to_plot'] type... (should be list -could convert int but not float etc..)
   if not (type(step_info['atoms_to_plot']) == list or type(step_info['atoms_to_plot']) == type(xrange(1))):
      EXC.ERROR("Sorry the variable 'atoms_to_plot' seems to be in an unfamiliar format (please use a list, an xrange or an integer).\n\nCurrent step_info['atoms_to_plot'] type is:\t%s"%str(type(step_info['atoms_to_plot'])))
   step_info['atoms_to_plot'] = [i for i in step_info['atoms_to_plot'] if i < len(step_info['mol_info'])]
   if len(step_info['atoms_to_plot']) == 0:
      EXC.ERROR('NO DATA PLOTTED, THERE ARE NO ATOMS CURRENTLY PLOTTED. PLEASE CHECK THE VARIABLE "atoms_to_plot"')
   step_info['AOM_D'] = {i:step_info['AOM_D'][i] for i in step_info['AOM_D'] if step_info['AOM_D'][i][1] in step_info['atoms_to_plot']}

poss_atoms = [i for i, elm in enumerate(step_info['at_num'][0]) if elm not in atoms_to_ignore]
step_info['at_num'][0] = np.array([typ.atomic_num_convert(i) for i in step_info['at_num'][0]])

if ignore_inactive_mols:
  step_info['atoms_to_plot'] = [i[0] for i in step_info['active_mols'] if i[1] in step_info['atoms_to_plot']]
step_info['mol_info'] = {i:step_info['mol_info'][i] for i in step_info['mol_info'] if i in step_info['AOM_D'].keys()}
active_atoms = [step_info['AOM_D'][i][1] for i in step_info['AOM_D']]
if ignore_inactive_atoms:
    step_info['atoms_to_plot'] = [i for i in step_info['AOM_D'] if step_info['AOM_D'][i][1] in step_info['atoms_to_plot']]
step_info['atoms_to_plot'] = [i for i in step_info['atoms_to_plot'] if i in poss_atoms]

# Finds list of keys attached to a value in a dictionary
def find_value_dict(D, value):
     return [i for i in D if D[i] == value]
step_info['active_atoms_index'] = [find_value_dict(step_info['mol_info'],i) for i in range(nmol)]
step_info['num_mols_active'] = len([i for i in  step_info['active_atoms_index'] if i])
if max_active_mol > step_info['num_mols_active']:
    cont = raw_input("The charge will eventually leave the number of plotted molecules in this simulation.\n\nMax molecule the charge reaches = %i.\n\nAre you sure you want to continue [y/n]:\t"%max_active_mol)
    if not typ.translate_to_bool(cont, 'cont'):
        SystemExit("Ok, exiting so you can change the settings file.\nI suggest using the keyword:\n\natoms_to_plot = 'auto'")
    else:
        print("Okie doke, you will get empty molecules though!")

# Position the time label.
if type(pos_time_label) == list:
   if len(pos_time_label) == 3:
      avgx,avgy,avgz = pos_time_label
else:
   if not auto_time_label and step_info['verbose']:
      EXC.WARN("Assuming I should position the time label automatically, as the position hasn't been set and type of positioning hasn't been stated.")
   if background_mols:
       largest_dim = np.argmax([np.max(step_info['coords'][0][:,i]) for i in range(3)])
       dims = [Xdims, Ydims, Zdims][largest_dim]
       max_coord = np.max(step_info['coords'][0][step_info['atoms_to_plot']][:,largest_dim])+step_info['background_mols_end_extend']
       mask = step_info['coords'][0][:,largest_dim]<max_coord
       all_time_label_coords = step_info['coords'][0][mask]
   else:
       all_time_label_coords = np.array([[np.max(step_info['coords'][:,atom,dim]) for dim in range(3)] for atom in step_info['atoms_to_plot'] ])
   all_time_label_coords = [all_time_label_coords[:,i] for i in range(3)]
   max_center, max_span = geom.min_bounding_box(all_time_label_coords,[1,1,1])
   #max_span[int(np.argmin(max_span))] *= 4
   max_span = [max_span[0], max_span[1], 0]
   Tcoords = np.array(max_center) + np.array(max_span)/np.array([8,-4,1])
   avgx, avgy, avgz = Tcoords

all_active_coords = step_info['coords'][:,[i[0] for i in step_info['active_mols']]]

if pvecs_on:
    step_info['pvecs'], _, _ = io.read_xyz_file(CP2K_output_files['pvecs'],3, start_step, end_step, stride)
    if np.sum(step_info['pvecs'][0]) == 0 and not calibrate:
       step_info['pvecs'][0] = step_info['pvecs'][1]
else:
    print("Calculating the pvecs...")
    step_info['pvecs'] = [geom.calc_pvecs_for_1_step(all_active_coords, i, step_info) for i in range(len(step_info['Ntime-steps']))]



step_info['tcl']['any_extra_raw_tcl_code'] = ""
step_info['tcl']['isoval']   = isosurface_to_plot
step_info['tcl']['Ccol']     = Carbon_colour
step_info['tcl']['Hcol']     = Hydrogen_colour
step_info['tcl']['zoom_val'] = zoom_value
#step_info['tcl']['mol_id']   =  0
step_info['tcl']['Ncol'] = Neon_color
step_info['tcl']['atom_style'] = mol_style
step_info['tcl']['mol_material'] = mol_material
step_info['tcl']['iso_material'] = iso_material
step_info['tcl']['time_step']    = '" "'
step_info['tcl']['iso_type']     = 0
step_info['tcl']['density_colour']     = str(density_iso_col).replace("(",'"').replace(")",'"').replace(",",'')
step_info['tcl']['imag_pos_col'] = str(pos_imag_iso_col).replace("(",'"').replace(")",'"').replace(",",'')
step_info['tcl']['imag_neg_col'] = str(neg_imag_iso_col).replace("(",'"').replace(")",'"').replace(",",'')
step_info['tcl']['real_pos_col'] = str(pos_real_iso_col).replace("(",'"').replace(")",'"').replace(",",'')
step_info['tcl']['real_neg_col'] = str(neg_real_iso_col).replace("(",'"').replace(")",'"').replace(",",'')
step_info['tcl']['maxX'] = Xdims[1]
step_info['tcl']['minX'] = Xdims[0]
step_info['tcl']['maxY'] = Ydims[1]
step_info['tcl']['minY'] = Ydims[0]
step_info['tcl']['maxZ'] = Zdims[1]
step_info['tcl']['minZ'] = Zdims[0]
step_info['tcl']['backgrnd_mols'] = ""
if step_info['background_mols']:
    step_info['tcl']['bckg_mols_on_off'] = ''
else:
    step_info['tcl']['bckg_mols_on_off'] = '#'
if show_box:
   step_info['tcl']['iso_type']  = 2
step_info['tcl'] = txt_lib.tcl_3D_input(background_colour, ['R','G','B'], step_info['tcl'], "backgrnd_")
step_info['tcl'] = txt_lib.tcl_3D_input(translate_by, ['x','y','z'], step_info['tcl'], "trans_")
step_info['tcl'] = txt_lib.tcl_3D_input([avgx,avgy,avgz], ['x','y','z'], step_info['tcl'], "time_lab_")

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
crds = step_info['coords'][0,step_info['atoms_to_plot'],:]

sys_dims = MT.find_sys_dims(crds)
crds -= sys_dims['center']

# Aligns along the x axis
zangle = MT.find_angle(crds, sys_dims['lens'], [0,1])

first_mol_indices = [i for i in  step_info['mol_info'] if step_info['mol_info'][i] == 0]
crds2 = step_info['coords'][0:,first_mol_indices,:][0]
# Finds the rotation angle to align the first molecule to face the user (perp to z)
new_sys_dims = MT.find_sys_dims(crds2)
xangle = MT.find_angle(crds2, new_sys_dims['lens'], [1,2])

rotation = [xangle*180./np.pi, 0, zangle*180./np.pi]
rotation = [0,0,0]
step_info['tcl'] = txt_lib.tcl_3D_input(rotation, ['x','y','z'], step_info['tcl'], "rot")
step_info['tcl']['pic_filename'] = {}
# set this in settings
step_info['tcl']['vmd_log_file'] = step_info['vmd_log_file']

# Checking Tachyon Renderer Path
new_tachyon_path = io.find_tachyon(ps.tachyon_path)
if new_tachyon_path != ps.tachyon_path:
    print("Finding Tachyon Ray Tracer")
    io.read_write_perm_settings(permanent_settings_filepath, "tachyon_path", new_tachyon_path)
    tachyon_path = new_tachyon_path
else:
    tachyon_path = ps.tachyon_path
step_info['tcl']['tachyon_path'] = tachyon_path

localities = []
TimEsTeps  = []
num_steps = len(step_info['aci'])
all_steps = xrange(num_steps)
if calibrate:
   all_steps = [0]
num_leading_zeros = int(np.ceil(np.log10(num_steps)))

step_info['times'] = {'Time to create Wavefunction':np.zeros(num_steps),
         'Create Cube Data': np.zeros(num_steps),
         'Write Cube File':np.zeros(num_steps),
         'VMD Visualisation':np.zeros(num_steps),
         'Plot and Save Img':np.zeros(num_steps)}
step_info['times_taken'] = []



# Completes 1 full step.
class MainLoop(object):

    """ This will carry out all the main functions and actually run the code.
         It will create the data, save it as a cube file, and make vmd render
         it      """

    def __init__(self, step_info, all_steps, calibrate, errors):
        self.tcl_dict_ind = 0
        self.errors = errors
        self.step_info = step_info
        self.neg_iso_cols = {}
        self.pos_iso_cols = {}
        self.PID = "MainProcess"
        for step in all_steps:
            self.aind = self.step_info['aci'][step]
            self.mind = self.step_info['mci'][step]
            self.theta1 = np.angle(self.step_info['mol'][self.mind][0])
            start_time = time.time()
            self.do_step(step)
            self._print_timings(step, len(all_steps), start_time)
        self._finallise(calibrate, num_steps)

    # Completes 1 step
    def do_step(self, step):
        self._find_active_molecules()
        self.data_files_to_visualise = []
        self._vmd_filename_handling()
        if self.step_info['background_mols']:
            self._write_background_mols()
        self.theta1 = np.angle(self.step_info['mol'][self.mind][0])
        for mol_i,mol_id in enumerate(self.active_step_mols):
            self._find_active_atoms(mol_id)
            self._create_wf_data(mol_id, step)
            self._set_wf_colours()
            self._save_wf_colours()
            self._create_cube_file_txt(step)
            self._write_cube_file(step, mol_id)
        if not self.step_info['load_in_vmd']:
            self._vmd_visualise(step, calibrate)
        if self.step_info['side_by_side_graph']:
            self._plot(step)

    # Finds a dynamic bounding box scale. Makes the bounding box smaller when the mol_coeff is smaller
    def _dynamic_bounding_box_scale(self, mol_ind, BBS): # Can calculate the mol_C_abs here instead of in the create_wf_data function
        """
        Calculates the new bounding box scale according to how much population
        there is on the molecule. Uses a tanh function to vary scale.

        Inputs:
            BBS      =>  Original bounding box scale
            mol_ind  =>  The index of the molecule
        """
        w = 0.4 # When do we first start getting there
        c = 4  # minimum bounding box scale
        pop = np.abs(self.step_info['mol'][self.mind][mol_ind])**2
        new_bb_scale = np.tanh(pop/w)
        new_bb_scale *= (BBS - c)
        new_bb_scale += c
        new_bb_scale = np.ceil(new_bb_scale)
        return int(new_bb_scale)

    # Will handle where to save the various vmd files created
    def _vmd_filename_handling(self):
        self.step_info['vmd_script'][self.PID] = self.step_info['vmd_script_folder']+ self.PID+".tcl"
        self.step_info['vmd_junk'][self.PID] = self.step_info['vmd_script_folder'] + self.PID + '.out'
        self.step_info['vmd_err'][self.PID] = self.step_info['vmd_script_folder'] + self.PID + '.error'
        self.step_info['delete_these'].append(self.step_info['vmd_junk'][self.PID])
        self.step_info['delete_these'].append(self.step_info['vmd_err'][self.PID])

    # Will save the background molecules in an xyz file to be loaded by vmd
    def _write_background_mols(self):
        # Dealing with the background molecules
        largest_dim = np.argmax([np.max(self.step_info['coords'][self.mind][:,i]) for i in range(3)])
        #dims = [Xdims, Ydims, Zdims][largest_dim]
        max_coord = np.max(self.step_info['coords'][self.aind][self.step_info['atoms_to_plot']][:,largest_dim])+self.step_info['background_mols_end_extend']
        mask = self.step_info['coords'][self.aind][:,largest_dim]<max_coord
        background_mols_pos = self.step_info['coords'][self.aind][mask]
        background_mols_at_num = self.step_info['at_num'][0][mask]
        backgrnd_mols_filepath = self.step_info['data_fold']+ "bckgrnd_mols-%s.xyz"%self.PID
        io.xyz_step_writer(background_mols_pos, background_mols_at_num['Mtime-steps'][self.mind], self.aind, backgrnd_mols_filepath, bohr2ang)
        tcl_load_xyz_cmd = 'mol new {%s} type {xyz} first 0 last -1 step 1 waitfor 1'%backgrnd_mols_filepath
        self.step_info['tcl']['backgrnd_mols'] = tcl_load_xyz_cmd

    # Finds how many molecules have a significant charge to visualise
    def _localisation(self): # Probably isn't actually that useful! The min tolerance thing is actually more useful.
        localisation = MT.IPR(self.step_info['mol'][self.mind])
        if localisation > 1.00001:
            localisation *= 1+np.exp(-localisation)
        localisation = int(np.ceil(localisation))
        print("Localisation = ", localisation, len(self.active_step_mols))

    # Will find the active molecules which need looping over
    def _find_active_molecules(self):
        self.active_step_mols = np.arange(0,nmol)[np.abs(self.step_info['mol'][self.mind])**2 > self.step_info['min_molC'] ]
        self.step_info['mols_plotted'] = len(self.active_step_mols)
        if self.step_info['mols_plotted'] == 0:
            self.active_step_mols = np.array([0])
            if self.step_info['verbose']:
               EXC.WARN("No molecules found that have a high enough molecular coefficient to be plotted for trajectory %i"%step)
        return self.active_step_mols

    # Will find the active atoms to loop over
    def _find_active_atoms(self, mol_id):
        self.active_coords = np.array([self.step_info['coords'][self.aind][i] for i in self.step_info['active_atoms_index'][mol_id]])
        if len(self.active_coords) <= 0:
            if any(act_mol_id > self.step_info['num_mols_active'] for act_mol_id in self.active_step_mols):
                msg = "The charge is no longer contained by the molecules shown."
                msg += "\nPlease extend the range to allow for this!"
                msg += "\n\nMax charged molecule = %i\t Max molecule plotted = %i"%(max(self.active_step_mols)['num_mols_active'])
                EXC.WARN(msg, True)
            else:
                SystemExit("Something went wrong and I don't know what sorry!\nThe length of the active_coords array is %i. It should be >0"%len(self.active_coords))
                return False

    # Will create the wavefunction data
    def _create_wf_data(self, mol_id, step):
       start_data_create_time = time.time()
       # Drawing a bounding box around the active atoms to prevent creating unecessary data
       BBS_dyn = [self._dynamic_bounding_box_scale(mol_id, i) for i in self.step_info['bounding_box']]
       trans, active_size  = geom.min_bounding_box([self.active_coords[:,k] for k in range(3)],
                                                        BBS_dyn)
       self.sizes  = typ.int_res_marry(active_size, step_info['res'], [1,1,1])     #How many grid points
       scale_factors = np.array([i*self.step_info['res'] for j, i in enumerate(self.sizes)])
       # Actually create the data
       self.data = np.zeros(self.sizes, dtype=complex)
       self.origin = scale_factors/-2 + trans
       self.mol_C = self.step_info['mol'][self.mind][mol_id]
       mol_C_abs = np.absolute(self.mol_C)**2
       print("MOL C = ", mol_C_abs)
       for j in self.step_info['AOM_D']:
           if self.step_info['mol_info'][j] == mol_id:
               ac = self.step_info['coords'][self.aind][j] - trans
               self.atom_I = self.step_info['AOM_D'][j][1]
               self.data += MT.dot_3D(MT.SH_p(self.sizes[0], self.sizes[1], self.sizes[2], self.step_info['res'],ac),
                           self.step_info['pvecs'][self.aind][self.atom_I])*self.step_info['AOM_D'][j][0]
       if self.step_info['colour_type'] == 'density':
           self.data *= self.mol_C
           self.data *= np.conjugate(self.data)
       else:
           self.data *= mol_C_abs
       self.step_info['times']['Time to create Wavefunction'][step] += time.time()-start_data_create_time

    # Creates the cube file to save
    def _create_cube_file_txt(self, step):
        start_cube_create_time = time.time()
        xyz_basis_vectors    = [[self.step_info['res'],0,0], # Probably not too bad creating this tiny list here at every step.
                                [0,self.step_info['res'], 0],
                                [0,0,self.step_info['res']]]   #X, Y and Z vector directions.
        self.cube_txt = txt_lib.cube_file_text(self.data.real,
                                          vdim=self.sizes,
                                          mol_info=self.step_info['mol_info'],
                                          orig=self.origin,
                                          Ac=self.step_info['coords'][self.aind], An=self.step_info['at_num'][0],
                                          tit=self.step_info['Title'],
                                          atoms_to_plot=self.step_info['atoms_to_plot'],
                                          basis_vec=np.array([np.array(i) for i in xyz_basis_vectors]))
        self.step_info['times']['Create Cube Data'][step] = time.time()-start_cube_create_time

    # Handles the saving the wf colours in a dictionary of the wavefunction.
    def _set_wf_colours(self):
         thetai = np.angle(self.mol_C*self.atom_I) - self.theta1
         # Could optimise (and tidy) this, the code doesn't need to do all this at every step
         if self.step_info['colour_type'] == 'density':
              self.neg_iso_cols[self.tcl_dict_ind] = 22
              self.pos_iso_cols[self.tcl_dict_ind] = 22
         elif self.step_info['colour_type'] == 'real-phase':
              self.neg_iso_cols[self.tcl_dict_ind] = 21
              self.pos_iso_cols[self.tcl_dict_ind] = 20
         elif self.step_info['colour_type'] == 'phase':
           if -np.pi/4<thetai<=np.pi/4: # Pos Real Quadrant
              self.neg_iso_cols[self.tcl_dict_ind] = 21
              self.pos_iso_cols[self.tcl_dict_ind] = 20
           elif np.pi/4<thetai<=3*np.pi/4: # Pos Imag Quadrant
              self.neg_iso_cols[self.tcl_dict_ind] = 19
              self.pos_iso_cols[self.tcl_dict_ind] = 18
           elif 3*np.pi/4<thetai<=5*np.pi/4: # Neg Real Quadrant
              self.neg_iso_cols[self.tcl_dict_ind] = 20
              self.pos_iso_cols[self.tcl_dict_ind] = 21
           else:                         # Neg imag Quadrant
              self.neg_iso_cols[self.tcl_dict_ind] = 18
              self.pos_iso_cols[self.tcl_dict_ind] = 19
         self.tcl_dict_ind += 1

    # Saves the wavefunction colouring in the tcl dictionary
    def _save_wf_colours(self):
         neg_col_dict_str = "set Negcols " + str(self.neg_iso_cols).replace(',','').replace('[','{').replace(']',' }').replace(':','').replace("'","")
         pos_col_dict_str = "set Poscols " + str(self.pos_iso_cols).replace(',','').replace('[','{').replace(']',' }').replace(':','').replace("'","")
         self.step_info['tcl']['neg_cols'] = neg_col_dict_str
         self.step_info['tcl']['pos_cols'] = pos_col_dict_str

    # Visualises the vmd data and adds timings to the dictionary
    def _vmd_visualise(self, step, calibrate):
        start_vmd_time = time.time()
        self.step_info['tcl']['pic_filename'][self.PID] = self.tga_filepath
        if 'tga' not in self.step_info['files_to_keep'] and not calibrate:
            self.step_info['delete_these'].append(self.tga_filepath)
        io.vmd_variable_writer(self.step_info, self.PID)
        io.VMD_visualise(self.step_info, self.PID)
        self.step_info['times']['VMD Visualisation'][step] =  time.time() -start_vmd_time

    # Handles the writing of the necessary files
    def _write_cube_file(self, step, mol_id):
        start_data_write_time = time.time()
        if keep_cube_files:
           data_filename = "%i-%s.cube"%(step, mol_id)
        else:
           data_filename = "tmp%i-%s.cube"%(mol_id, self.PID)
        data_filepath = self.step_info['data_fold'] + data_filename
        step_info['delete_these'].append(data_filepath)
        self.data_files_to_visualise = [data_filepath] + self.data_files_to_visualise
        self.step_info['tcl']['cube_files'] = ' '.join(self.data_files_to_visualise)
        self.tga_folderpath, _, self.tga_filepath = io.file_handler(self.step_info['img_prefix']+txt_lib.add_leading_zeros(step,num_leading_zeros), 'tga', self.step_info)
        if draw_time:
           self.step_info['tcl']['time_step'] = '"%s"'%(self.step_info['time_lab_txt'].replace("*",str(self.step_info['Mtime-steps'][self.mind])))
        self.step_info['tcl']['cube_files'] = ' '.join(self.data_files_to_visualise)
        io.open_write(data_filepath, self.cube_txt)
        self.step_info['times']['Write Cube File'][step] = time.time() - start_data_write_time

    # Handles the plotting of the side graph.
    def _plot(self, step):
        # Plotting if required
       start_plot_time = time.time()
       files  = {'name':"G%i"%step, 'tga_fold':self.tga_filepath}
       self.step_info['delete_these'].append(io.plot(self.step_info, self.mind, files, plt))
       self.step_info['times']['Plot and Save Img'][step] = time.time() - start_plot_time

    # Runs the garbage collection and deals with stitching images etc...
    def _finallise(self, calibrate, num_steps):
        if not calibrate:
            self._stitch_movie(num_steps)
        else:
            self._display_img()
        self._store_imgs()
        self._garbage_collector()

    # Show the image in VMD or load the image in a default image viewer
    def _display_img(self):
        if self.step_info['mols_plotted'] > 0:
            if self.step_info['load_in_vmd']:
                self.step_info['tcl']['pic_filename'][self.PID] = self.tga_filepath
                if 'tga' not in self.step_info['files_to_keep'] and not calibrate:
                    self.step_info['delete_these'].append(self.tga_filepath)
                io.vmd_variable_writer(self.step_info, self.PID)
                os.system("vmd -nt -e %s"%(self.step_info['vmd_script'][self.PID]) )
                io.settings_update(self.step_info)
            if show_img_after_vmd:
                open_pic_cmd = "xdg-open %s"%(self.tga_filepath)
                subprocess.call(open_pic_cmd, shell=True)
        else:
            EXC.WARN("There were no wavefunctions plotted on the molecules!")

    # Handles Garbage Collection
    def _garbage_collector(self):
        #self.step_info['delete_these'].append(self.step_info['vmd_log_file'])
        self.step_info['delete_these'].append(io.folder_correct('./vmdscene.dat'))
        # Garbage collection
        self.step_info['delete_these'].append(self.step_info['f.txt'])
        for i in self.step_info['delete_these']:
           if io.path_leads_somewhere(i):
              os.remove(i)

    # Handles converting the image to another img format for storing
    def _store_imgs(self):
        # Convert all .tga to .img
        if 'img' in self.step_info['files_to_keep']:
            cnvt_command = "mogrify -format %s %s*.tga"%(self.step_info['img_format'], self.tga_folderpath)
            subprocess.call(cnvt_command, shell=True)

    # Stitches the movie together from other files
    def _stitch_movie(self, num_steps):
        num_leading_zeros = int(np.ceil(np.log10(num_steps)))
        files = "img%0"+str(num_leading_zeros)+"d.tga"
        # Creating the ffmpeg and convert commands for stitching
        if self.step_info['movie_format'] == 'mp4':
            Stitch_cmd, tmp, _ = io.stitch_mp4(files, self.tga_folderpath, self.tga_folderpath+self.step_info['Title'], self.step_info['length_of_animation'], self.step_info['ffmpeg_bin'])
            self.step_info['delete_these'].append(tmp)
            self.step_info['delete_these'].append(_)
            #io.settings_update(self.step_info)
        if self.step_info['movie_format'] == 'gif':
          io.open_write(self.step_info['f.txt'], self.step_info['to_stitch']) #Writing all the image filenames (maybe need to sort this after paralellisation)
          Stitch_cmd = 'convert -delay '+str(100*(self.step_info['length_of_animation']/(end_step-start_step)))+' @'+self.step_info['f.txt']+' -loop 0 "'+self.tga_folderpath+self.step_info['Title']+'.gif"'
        subprocess.call(Stitch_cmd, shell=True) # Actually stitch the movie

    # Prints the timing info
    def _print_timings(self, step, num_steps, start_step_time):
        traj_print = "\n"+txt_lib.align("Trajectory %i/%i    %s    Timestep %s"%(step+1, num_steps,
          typ.seconds_to_minutes_hours(time.time()-start_step_time, "CPU: "), self.step_info['Mtime-steps'][self.mind]),
                   69, "l") + "*"
        if self.step_info['verbose']:
            print("*"*70)
            print (traj_print)
            io.times_print(self.step_info['times'],step, 70, time.time()-start_step_time)
        else:
            io.print_same_line(traj_print, sys, print)
        if self.step_info['verbose']:
            print("*"*70, "\n")
        self.step_info['times_taken'].append(time.time()-start_step_time)

step_info['to_stitch'] = '\n'.join([io.file_handler(step_info['img_prefix']+txt_lib.add_leading_zeros(step,num_leading_zeros), 'tga', step_info)[2] for step in all_steps])

errors = {}
step_data = MainLoop(step_info, all_steps, calibrate, errors)


# Print timings for full code
print("\r                             ")
time_elapsed_str = typ.seconds_to_minutes_hours(time.time()-START_TIME,"\nTotal Time Elapsed: ")
print(time_elapsed_str)

if not calibrate or not step_info['verbose']:
   for i in step_info['times']:
        step_info['times'][i] = [np.sum(step_info['times'][i])]

   io.times_print(step_info['times'],0, time.time()-START_TIME)
