from __future__ import print_function
"""
Will call all the relevant functions to initialise the program. The import
settings_file statement will read and parse the settings file into the
all_settings dictionary.
"""

"""Will import the python3 print function."""

from src import EXCEPT as EXC
from src import consts
from src import IO as io
from src import text as txt_lib
from src import type as typ
from src import math as MT
from src import Warnings as WRN
#from src import Colour as col
from src import geometry as geom

from init import I_utils as IU
from init import init_IO as i_IO
from init import settings_file #Importing this will initialise the all_settings dict
all_settings = settings_file.all_settings


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


from Templates import defaults as dft
# Import the permanent settings file if not create it and init with defaults
try:
    from Templates import permanent_settings as ps
except:
    EXC.replace_perm_settings()
    from Templates import permanent_settings as ps

# Make the code backwards compatible with python 2.7
if sys.version_info[0] > 2:
    xrange = range
    raw_input = input

# Set the title of the visualisation
all_settings['calibrate'] = typ.translate_to_bool(all_settings['calibrate'], "calibrate")
if all_settings['calibrate']:
   all_settings['title'] = "Calibration"
   print("In Calibration Mode...")
else:
   if all_settings['title'] == None:
      all_settings['title'] = raw_input("What should I call this visualisation?\n")             #can be anything will be ignored by VMD.
      print("\n\n")
   else:
      all_settings['title'] = str(all_settings['title'])


START_TIME = time.time()

#TO DO:
#
# * Improve Parallelism by using subprocess.Popen() and threading when opening VMD, so it opens a new instance of VMD on a new thread.
#
# * Add an estimated time remaining by using the average time taken for each step
#
# * Default the all_settings['mol'] to (1,0,0,0,0,...,0)
#
# * Make it work for files where the number of pvecs steps isn't equal to the number of position steps. (Will be fixed with fixed pvecs)
#    Remove bad steps from the simulation data
#    ** Calculate own pvecs. (nearly)
#
# * Add crash report data so other users can tell me exactly why it crashed and I will be able to fix it.
#      Add more checks and debugging info (better error messages etc).


if type(all_settings['end_step']) != int:
    if type(all_settings['end_step']) == str:
        if 'al' not in all_settings['end_step'].lower():
            EXC.WARN("Sorry the 'end_step' variable needs to be an integer not a %s!\n\nConverting from %.2g to %i"%(type(all_settings['end_step']),all_settings['end_step'],int(round(all_settings['end_step']))))
            all_settings['end_step'] = int(round(all_settings['end_step']))
    else:
        EXC.WARN("Sorry the 'end_step' variable needs to be an integer not a %s!\n\nConverting from %.2g to %i"%(type(all_settings['end_step']),all_settings['end_step'],int(round(all_settings['end_step']))))
        all_settings['end_step'] = int(round(all_settings['end_step']))

# Sorting out filenames
IU.init_output_files_and_folders(all_settings) # Will declare all the paths that are required in the code
IU.init_all_settings_other(all_settings) # Will initialise settings that aren't file/folder paths
io.create_data_img_folders(all_settings)
keep_tga_files = WRN.redundant_img(all_settings['keep_img_files'], all_settings['keep_tga_files'])
IU.init_permanent_settings(all_settings)
IU.init_tcl_dict(all_settings)

# Functions that don't need coords, coeffs, pvecs etc...
IU.get_all_files_metadata(all_settings)
IU.find_step_numbers(all_settings)
IU.init_global_steps_to_ignore(all_settings)
IU.init_local_steps_to_ignore(all_settings)
IU.init_show_box(all_settings)
IU.init_cal_display_img(all_settings)
IU.init_files_to_keep(all_settings)
IU.init_animation_type(all_settings)
IU.init_colors(all_settings)    # Initialises the colors of the wavefunction
IU.check_VMD_TEMP(all_settings)

# Read input files
IU.read_cp2k_inp_file(all_settings)
IU.init_AOM_D(all_settings)
i_IO.read_coords(all_settings)
i_IO.read_coeffs(all_settings)
i_IO.read_pvecs(all_settings)

# Functions needing data from input files
IU.init_mols_to_plot(all_settings)
IU.init_atoms_to_plot(all_settings)
IU.init_rotation(all_settings)
IU.init_bounding_box(all_settings)
IU.check_charge_spread(all_settings)
IU.init_times_dict(all_settings)
all_steps = xrange(len(all_settings['coords']))



















# # Position the time label.
# if type(pos_time_label) == list:
#    if len(pos_time_label) == 3:
#       avgx,avgy,avgz = pos_time_label
# else:
#    auto_time_label = txt_lib.fuzzy_variable_translate(pos_time_label, ["auto","A list containing absolute positions i.e. [Pos_x, Pos_y, Pos_z]"], all_settings['verbose_output'], False)
#    if not auto_time_label and all_settings['verbose_output']:
#       EXC.WARN("Assuming I should position the time label automatically, as the position hasn't been set and type of positioning hasn't been stated.")
#    if background_mols:
#        largest_dim = np.argmax([np.max(all_settings['coords'][0][:,i]) for i in range(3)])
#        dims = [Xdims, Ydims, Zdims][largest_dim]
#        max_coord = np.max(all_settings['coords'][0][all_settings['atoms_to_plot']][:,largest_dim])+all_settings['all_settings['background_mols_end_extend']']
#        mask = all_settings['coords'][0][:,largest_dim]<max_coord
#        all_time_label_coords = all_settings['coords'][0][mask]
#    else:
#        all_time_label_coords = np.array([[np.max(all_settings['coords'][:,atom,dim]) for dim in range(3)] for atom in all_settings['atoms_to_plot'] ])
#    all_time_label_coords = [all_time_label_coords[:,i] for i in range(3)]
#    max_center, max_span = geom.min_bounding_box(all_time_label_coords,[1,1,1])
#    #max_span[int(np.argmin(max_span))] *= 4
#    max_span = [max_span[0], max_span[1], 0]
#    Tcoords = np.array(max_center) + np.array(max_span)/np.array([8,-4,1])
#    avgx, avgy, avgz = Tcoords
# if not use_fuzzy_files:
#    pvecs_on = io.path_leads_somewhere(all_settings['CP2K_output_files']['pvecs'])
# else:
#     pvecs_on = len(all_settings['CP2K_output_files']['pvecs']) > 0
# if not pvecs_on:
#     EXC.WARN("Can't find the Pvecs file, this contains information about the orientation of the P orbitals.")
