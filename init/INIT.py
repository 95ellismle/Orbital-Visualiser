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


# Probably no need for this, just use default python errors
def import_and_check(str_lib, error=True):
    """A function to import modules and check if they exist."""
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


# Python version should be 3.6 or above
if (sys.version_info.major != 3 or
        sys.version_info.minor < 6):
    raise SystemError('Please use Python3.6+ to run this code.')



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


# Sorting out filenames
IU.init_output_files_and_folders(all_settings) # Will declare all the paths that are required in the code
IU.init_all_settings_other(all_settings) # Will initialise settings that aren't file/folder paths
IU.init_permanent_settings(all_settings)
IU.init_tcl_dict(all_settings)
io.create_data_img_folders(all_settings)
all_settings['vmd_exe'] = io.find_vmd(all_settings['vmd_exe'])
IU.check_filepaths(all_settings)
consts.at_num_orb_map = all_settings['atomic_number_orbital_map']

# Functions that don't need coords, coeffs, pvecs etc...
print("Reading data files' metdata", end='\r')
IU.get_all_files_metadata(all_settings)
print("\rFinished metdata read        ")

IU.init_steps_to_do(all_settings)
IU.find_step_numbers(all_settings)
IU.init_ignore_steps_for_restart(all_settings)
IU.fix_missing_pos_steps(all_settings)
IU.init_show_box(all_settings)
IU.init_cal_display_img(all_settings)
IU.init_files_to_keep(all_settings)
IU.init_animation_type(all_settings)
IU.init_colors(all_settings)    # Initialises the colors of the wavefunction
IU.check_VMD_TEMP(all_settings)
all_settings['keep_tga_files'] = WRN.redundant_img(all_settings['keep_img_files'], all_settings['keep_tga_files'])

# Read input files
print("Reading Input Data    ", end='\r')
IU.read_cp2k_inp_file(all_settings)
IU.init_AOM_D(all_settings)
IU.handle_decomp_file(all_settings)
print("\rReading Coeff File     ", end='\r')
i_IO.read_coeffs(all_settings)
print("\rReading Pos File    ", end='\r')
i_IO.read_coords(all_settings)
print("\rCreating pvecs      ", end='\r')
i_IO.read_pvecs(all_settings)
print("\rFinished Reading Data     ")

# Functions needing data from input files
IU.get_mol_groupings(all_settings)
IU.init_rotation(all_settings)
IU.init_bounding_box(all_settings)
#IU.check_charge_spread(all_settings)
#all_settings['reversed_mol_info'] = IU.reverseDict(all_settings['mol_info'])
IU.init_times_dict(all_settings)

all_steps = range(len(all_settings['pos_step_inds']))

