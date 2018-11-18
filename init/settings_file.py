'''
Will initialise the all_settings dictionary. This will read the settings.inp
file and fill out any settings that are missing from the default settings. This
will also correct any typos in the settings file.

N.B. The all_settings dictionary is a big dictionary with every setting used in
the code. This is where the MainLoop class looks for any settings it needs.
This is especially important for filepaths as these really need to be consistent
throughout the code. 
'''

from src import IO as io
from src import consts
from src import text as txt_lib

from Templates import defaults as dft

import collections as coll
import numpy as np
import difflib as dfl


# Will remove bad lines from a settings file.
def remove_bad_lines(settings_ltxt):
    def is_bad_line(line):
        if '=' not in line:
            return True
        else:
            try:
                exec(line)
            except:
                return True
    return [i for i in settings_ltxt if not is_bad_line(i)]

# Creates a dictionary with only settings that make sense
def create_clean_settings(settings_ltxt):
    clean_settings_dict = coll.OrderedDict()
    for i in settings_ltxt:
        setting_name = i.split('=')[0].replace(' ','')
        success = False
        try:
            success = True
            setting_val  = eval(i.split('=')[-1])
        except:
            print("Warning the setting %s is a bit dodgy. Using the default value"%setting_name)
        if success:
            if type(setting_val) == list:
                clean_settings_dict[setting_name] = np.array(setting_val)
            else:
                clean_settings_dict[setting_name] = setting_val
    return clean_settings_dict

# Creates a dictionary with only the original settings that make sense including comments
def create_orig_settings(settings_ltxt):
    orig_settings_dict = coll.OrderedDict()
    count = 0
    for line in settings_ltxt:
        cmt = ''
        if '#' in line:
            cmt = '#'+'#'.join(line.split('#')[1:])
        if line == cmt:
            orig_settings_dict[count] = ['',cmt]
            count += 1
            continue
        setting_name = line.replace(cmt, "").split('=')[0].replace(' ','')
        setting_val = ''
        success = False
        try:
            setting_val  = eval(line.split('=')[-1])
            success = True
        except:
            pass
        if success:
            orig_settings_dict[setting_name] = [setting_val, cmt]
    return orig_settings_dict

# Prints out which variables have typos and have been changed
def print_dodgy_vars(dodgy_vars):
    if dodgy_vars:
        tab_width  = 22
        clear_row  = "|"+" "*(2*tab_width + 5)+"|"
        line_row   = "|"+"-"*(2*tab_width + 5)+"|"
        top_row    = "_"*(2*tab_width + 6)
        bottom_row = "|"+"_"*(2*tab_width + 5)+"|"
        print("Discovered some typos in the input file:\n")
        print(top_row)
        print(clear_row)
        print("| %s -> %s |"%(txt_lib.align("Previous Setting", tab_width-1, 'l'), txt_lib.align("New Setting", tab_width, 'r')))
        print(clear_row)
        print(line_row)
        print(clear_row)
        for bad_sett, good_set in dodgy_vars:
            s = "| %s -> %s |"%(txt_lib.align(bad_sett, tab_width-1, 'l'), txt_lib.align(good_set, tab_width, 'r'))
            print(s)
            print(clear_row)
        print(bottom_row)

def get_act_setting(setting, all_setting_names):
    """
    Will apply a fuzzy finder to the settings in the settings.inp file and
    choose the best one.

    Inputs:
        setting            => The setting to be corrected
        all_setting_names  => All possible setting names
    """
    poss_setts = txt_lib.fuzzy_variable_translate(setting.lower(), all_setting_names,False, False,0.6)
    if sum(poss_setts) > 1:
        raise SystemExit("There are too many possible settings for '%s'. These are:\n\t* %s.\n\nI do not want to assume which one it is, please correct it in the input file!"%(setting, '\n\t* '.join(all_setting_names[poss_setts])))
    elif sum(poss_setts) < 1:
        raise SystemExit("I don't know what you mean by '%s'. Please correct it in the input file!"%(setting))
    return all_setting_names[poss_setts][0]

# Creates the all_settings dictionary and fills it with the settings from the input file (if there) else the defaults
def grab_defaults(clean_settings_dict):
    dodgy_vars = []
    all_settings = {}
    all_setting_names = np.array([i.lower() for i in dft.defaults.keys()])
    for setting in dft.defaults:
        #Loop over variables and clean/store them
        actual_setting = get_act_setting(setting, all_setting_names) #use fuzzy logic to correct setting name
        if actual_setting.lower() != setting.lower():
            dodgy_vars.append((setting, actual_setting))

        # Take the variable from the input file, if not use the default
        if type(clean_settings_dict.get(setting)) != type(None):
            all_settings[actual_setting] = clean_settings_dict[setting]
        else:
            all_settings[actual_setting] = dft.defaults[setting]
    print_dodgy_vars(dodgy_vars)
    return all_settings

# Get the cleaned_original settings to write to a file
def find_final_orig_settings(orig_settings_dict, all_settings):
    final_orig_settings_dict = coll.OrderedDict()
    all_setting_names = np.array([i.lower() for i in dft.defaults.keys()])
    path_done = False
    # Get the path variable
    for setting in orig_settings_dict:
        # Handle comment lines
        if type(setting) == int:
            final_orig_settings_dict[setting] = orig_settings_dict[setting]
            continue

        # Don't handle the path variable as it hasn't got a default
        if not path_done and dfl.SequenceMatcher(None, setting, 'path').ratio() > 0.8:
            path_done = True
            all_settings['path'] = clean_settings_dict[setting]
            final_orig_settings_dict['path'] = orig_settings_dict[setting]
            continue

        actual_setting = get_act_setting(setting, all_setting_names)
        if type(orig_settings_dict[setting][0]) == str:
            final_orig_settings_dict[actual_setting] = ["'%s'"%orig_settings_dict[setting][0], orig_settings_dict[setting][1]]
        else:
            final_orig_settings_dict[actual_setting] = orig_settings_dict[setting]
    return final_orig_settings_dict

settings_file = io.folder_correct(consts.settings_filepath)
orig_settings_ltxt = io.open_read(settings_file).split('\n')
settings_ltxt = remove_bad_lines(orig_settings_ltxt)
clean_settings_dict = create_clean_settings(settings_ltxt)
orig_settings_dict = create_orig_settings(orig_settings_ltxt)
all_settings = grab_defaults(clean_settings_dict)
final_orig_settings = find_final_orig_settings(orig_settings_dict, all_settings)
io.write_cleaned_orig_settings(final_orig_settings, settings_file)
if 'path' not in all_settings:
    raise SystemExit("""Sorry I can't find the path variable.... Are you sure you set this?

Use path='...' in the settings.inp file.
""")

if not io.path_leads_somewhere(all_settings['path']):
   raise SystemExit("\nThe specified path doesn't lead anywhere:\n%s\n\n\t Where is my data?! "%all_settings['path'])

all_settings['path'] = io.folder_correct(all_settings['path'])
