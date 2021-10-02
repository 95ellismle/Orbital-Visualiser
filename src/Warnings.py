"""
Contains functions that provide warnings.

N.B. This should be merged with EXCEPT.py
"""

from src import text as txt_lib
from src import type as typ

import sys

if sys.version_info[0] > 2:
    xrange = range
    raw_input = input

# Warns the user if any variables have been taken from the defaults
def default_variable_warn(dodgy_vars, step_info):
    if dodgy_vars and False:
       print("The following variables have been taken from defaults:\n")
       print("\t"+txt_lib.align("Default",20,'l')+"|  "+txt_lib.align("Setting", 30,'l'))
       print("\t"+"-"*20+"|"+"-"*32)
       for i in dodgy_vars:
           print("\t"+txt_lib.align(str(i[0]), 20, 'l') + "|   " + txt_lib.align(str(i[1]), 20, 'l'))
       print("\n\n")

# Warns the user about any typos in the settings file
def typo_variable_warn(replacer_settings, setting_file_settings, step_info):
     if replacer_settings and step_info['verbose']:
         print("The following typos have been corrected from the settings file:")
         print("\t"+txt_lib.align("Typo", 20, 'l') + "|   " + txt_lib.align("Correction", 20, 'l'))
         print("\t"+"-"*20+"|" + "-"*23)
         for sett,new_sett in zip(setting_file_settings, replacer_settings):
             print( "\t" +txt_lib.align(sett, 20, 'l') + "|   " + txt_lib.align(new_sett, 20, 'l'))
         print("\n")

# Warns user about keeping redundant images
def redundant_img(keep_png_files, keep_tga_files):
    if keep_png_files and keep_tga_files:
        Q = typ.translate_to_bool(raw_input("Are you sure you want to keep both the tga and png files? Should I delete the tga files:\t"),'answer to keep both png,tga question')
        if Q:
            return False
        return True
