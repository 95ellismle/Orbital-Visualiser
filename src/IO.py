from  __future__ import division
from __future__ import print_function
'''
This large module contains many functions that are involved in the input/output
operations in the code. These are things like reading the xyz file, writing the
cube file, writing the xyz file, a generic open_read function and open_write.
'''

import os
import sys
import datetime as datetime
import numpy as np
import time
import subprocess as sb
import difflib as dfl
from collections import OrderedDict
#import multiprocessing as mp

from src import text as txt_lib
from src import type as typ
from src import EXCEPT as EXC
from src import consts

try:
    from Templates import permanent_settings as ps
except:
    EXC.replace_perm_settings()
    from Templates import permanent_settings as ps

from Templates import defaults as dft

if sys.version_info[0] > 2:
    xrange = range
    raw_input = input

# Checks whether the tachyon path specified is the correct one.
def check_tachyon(tachyon_path, times=0):
    """
    Check the tachyon path in the permanent settings file.

    This will attempt to run tachyon and check the output. If
    it doesn't work then the permissions will be changed and
    Tachyon will be checked again. If this still doesn't work
    return False

    Inputs:
        tachyon_path <str> => the path to the Tachyon binary
        times <int> => DON'T CHANGE. How many times the function
                       has been called (only for recursion).


    return bool
    """
    if not tachyon_path:
        return False
    tachyon_out =os.popen(tachyon_path, 'r').read()
    tachyon_spiel_ind = tachyon_out.lower().find("tachyon parallel/multiprocessor ray tracer")
    if tachyon_spiel_ind != -1:
        return True
    else:
       os.chmod(tachyon_path, int('755', base=8))
       if times < 1:
           result = check_tachyon(tachyon_path, times+1)
           return result
    return False

# Reads and writes the updated permanent settings file
def read_write_perm_settings(filepath, setting, value):
    txt = open_read(filepath)
    new_txt = txt_lib.change_perm_settings(txt, setting, value)
    open_write(filepath, new_txt,'w')

def check_dir_for_tachyon(directory):
    """
    Will check a directory for the tachyon_LINUXAMD64 file. This should be found
    in the vmd program directory as it is a plugin that come with it. The
    tachyon program is a ray tracer engine that renders the pictures.
    """
    for dpath, dnames, fnames in os.walk(directory):
        for dname, fname in zip(dnames, fnames):
            if 'tachyon' in dname or 'tachyon' in fname:
                return dpath+'/'+fname
    return False

# Searches recursively for the VMD tachyon renderer path.
def find_tachyon(current_tachyon_path=''):
    if check_tachyon(current_tachyon_path):
        return current_tachyon_path
    if check_tachyon("./bin/tachyon_LINUXAMD64"):
        return "./bin/tachyon_LINUXAMD64"
    print("""The tachyon binary in the bin folder doesn't seem to be working!""")
    print("Trying to find Tachyon Ray-Tracer elsewhere on computer")
    try:
      vmd_path = sb.check_output("which vmd", shell=True)
    except sb.CalledProcessError:
      print("\n\n\n*******   CAN'T FIND VMD ERROR    ************")
      print("Sorry you don't seem to have VMD installed on your computer.")
      print("\nAt least I can't find it with the command 'which vmd'")
      print("\nPlease make sure you have set a bash alias `vmd' to open VMD")
      print("*******   CAN'T FIND VMD ERROR    ************\n")
      raise SystemExit("Exiting... Can't find VMD")
    vmd_path = vmd_path.decode('utf-8')
    for N in range(1,vmd_path.count('/')-1):
        vmd_path = txt_lib.folderpath_back_N(vmd_path,N)
        tachyon_path  = check_dir_for_tachyon(vmd_path)
        if tachyon_path:
            break
    if not tachyon_path:
        return False
    if check_tachyon(tachyon_path):
         return tachyon_path
    else:
        raise SystemExit("""
           *******    CAN'T FIND TACHYON ERROR   ***********
Sorry I couldn't find the path to the tachyon ray tracer.

Please specify this as a string in the Templates/permanent_settings.py
file. This is required for rendering the images.

To find the tachyon ray tracer engine try:
   * Open VMD

   * Select File/Render... in the menu bar at the top of the VMD main
     window.

   * Select 'Tachyon' from the 'Render the current scene using:'
     drop-down menu (first box).

   * At the begining of the 'Render Command' (third box down) you
     should see a string with the path to the tachyon ray-tracer
     engine. Something such as "/usr/local/lib/vmd/tachyon_LINUXAMD64"

   * Copy this into the Templates/permanent_settings.py file. (In the
   Orb_Mov_Mak directory)

If this doesn't work you may not have the tachyon ray-tracer with your
VMD package. Try re-installing VMD.""")


# Will stitch together an mp4 video
def stitch_mp4(files, files_folder, output_name, length, ffmpeg_binary, Acodec='aac', Vcodec='libx264', extra_flags="-pix_fmt yuv420p -preset slow -qscale 14", log_file="a.log", err_file="a.err"):
    if all(i in files for i in ['%','d','.']):
        ext = files.split('.')[-1]
        num_of_nums = eval(files.split('%')[-1].split('d')[0].strip('0'))
        prefix = files.split('%')[0]
        all_files = os.listdir(files_folder)
        all_files = [i for i in all_files if ext in i.split('.')[-1]] #removing files that don't have the correct extension
        all_files = [i for i in all_files if len(i[len(prefix):i.find('.')]) == num_of_nums] # only files with the correct amount of nums
        num_files = len(all_files)
        framerate = int(np.round(num_files/length, 0))
        if framerate == 0:
            framerate = 1
        in_files = files_folder+files
        pre_flags = ""

    elif "*" in files and '.' in files:
        ext = files.split('.')[-1]
        all_files = os.listdir(files_folder)
        all_files = [i for i in all_files if ext in i.split('.')[-1]] #removing files that don't have the correct extension
        num_files = len(all_files)
        framerate = int(np.round(num_files/length, 0))
        if framerate == 0:
            framerate = 1
        pre_flags = '-pattern_type glob' # Glob type input files
        in_files = '"%s"'%(files_folder+files) #input files must be inside string

    else:
        EXC.ERROR("Input format for image files is incorrect.\nPlease input them in the format:\n\n\t'pre%0Xd.ext'\n\nwhere pre is the prefix (can be nothing), X is the number of numbers in the filename, and ext is the file extensions (e.g. png or tga).")
    if path_leads_somewhere(output_name+'.mp4'):  os.remove(output_name+'.mp4') #remove file before starting to prevent hanging on 'are you sure you want to overwrite ...'
    options = (ffmpeg_binary, pre_flags, framerate, in_files, Vcodec, Acodec, extra_flags, output_name, log_file, err_file)

    Stitch_cmd = "%s -f image2 %s -framerate %s -i %s -vcodec %s -acodec %s %s %s.mp4 > %s 2> %s"%options
    print(Stitch_cmd)
    return Stitch_cmd, log_file, err_file


# Decides whether to use the previous rotations and scaling from the calibration step
def use_prev_scaling(path):
    if ps.previous_calibrate:
        if datetime.datetime.now() - datetime.datetime.strptime(ps.previous_runtime, ps.time_format) < datetime.timedelta(days=1):
            if path == ps.previous_path:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


# Reads the VMD log file and parses the information then updates the settings file
def settings_update(all_settings):
    """
    Reads/parses the VMD log file. Then we decide to put the rotations in the
    include.vmd file and combine the zooms/scalings and translations into a
    single operation. These are then written into the settings file.
    """
    vmd_log_text = open_read(all_settings['vmd_log_file'], False)
    if bool(vmd_log_text) is not False:
        os.remove(all_settings['vmd_log_file'])
        new_transforms = vmd_log_text[vmd_log_text.find(consts.end_of_vmd_file)+len(consts.end_of_vmd_file):].split('\n')

        # First handle the scalings
        new_zoom = txt_lib.combine_vmd_scalings(new_transforms) * all_settings['zoom_value']
        inp_zoom = all_settings['clean_settings_dict'].get('zoom_value')
        if type(inp_zoom) != type(None): # If the settings file declare a zoom value use the comments from it
           inp_zoom[0] = new_zoom
        else: # else use a standard comment
           inp_zoom = [new_zoom, '# How much to zoom by']
        all_settings['clean_settings_dict']['zoom_value'] = inp_zoom

        # Now handle translations
        new_translations = np.array(txt_lib.combine_vmd_translations(new_transforms))+all_settings['translate_by']
        inp_translate = all_settings['clean_settings_dict'].get("translate_by")
        if type(inp_translate) != type(None): # If the settings file declare a zoom value use the comments from it
           inp_translate[0] = new_translations
        else: # else use a standard comment
           inp_translate = [new_translations, '# How much to translate in xyz directions']
        all_settings['clean_settings_dict']['translate_by'] = inp_translate

        # Now save only certain actions to the include.vmd file to be sourced later
        whitelist = ['rotate']
        new_transforms = [line for line in new_transforms if any(j in line for j in whitelist)]
        new_include = open_read(all_settings['tcl']['vmd_source_file'], False) +'\n'*2 + '\n'.join(new_transforms)
        open_write(all_settings['tcl']['vmd_source_file'], new_include)
        write_cleaned_orig_settings(all_settings['clean_settings_dict'], 'settings.inp')
    else:
        EXC.WARN("VMD hasn't created a logfile!", all_settings['verbose_output'])
        return all_settings


# Writes the settings file with updated info from the step_info dict
def write_settings_file(step_info):
    settings_to_write = ''
    for i in step_info['orig_settings']:
        print(i)
        if "xx" in i:
            settings_to_write += '\n'
        elif step_info['orig_settings'][i] == 'cmt':
            settings_to_write += "%s\n"%i
        elif i:
             settings_to_write += "%s\n"%(' = '.join([i.strip(),step_info['orig_settings'][i].strip()]))
    open_write(step_info['settings_file'], settings_to_write)


# Will create the visualisation in VMD
def VMD_visualise(step_info, PID):
    os.system("touch %s"%step_info['vmd_junk'][PID])
    os.system("touch %s"%step_info['vmd_err'][PID])
    
    vmd_exe = step_info['vmd_exe']
    vmd_script = step_info['vmd_script'][PID]
    vmd_junk = step_info['vmd_junk'][PID]
    vmd_err = step_info['vmd_err'][PID]
    vmd_commnd = "%s -nt -dispdev none -e %s > %s 2> %s &"%(vmd_exe, vmd_script, vmd_junk, vmd_err)
    #print(vmd_commnd)
    os.system(vmd_commnd)  #Maybe subprocess.call would be better as this would open VMD in a new thread?
    made_file = False
    race_start = time.time()
    while (not made_file): #Wait for VMD to have finished it's stuff to prevent race conditions
        made_file = bool(os.path.isfile(step_info['tcl']['pic_filename'][PID])*vmd_finished_check( step_info['vmd_junk'][PID])) # This checks if VMD has finished preventing race conditions
        race_time = time.time() - race_start
        if race_time > step_info['vmd_timeout']:
            if (not os.path.isfile(step_info['tcl']['pic_filename'][PID])) and vmd_finished_check( step_info['vmd_junk'][PID]):
                EXC.ERROR("\n\nVMD finished, but hasn't rendered a file! Check the VMD script at %s"%step_info['vmd_script'][PID])
                os._exit(0)
            else:
                EXC.ERROR("\n\nVMD is taking a long time! I think there may be a bug in VMD script. Try compiling the script manually with the command 'source ./src/TCL/script_4_vmd.vmd' within the tkconsole in VMD.\nIf everything works there then try increasing the 'vmd_step_info['vmd_timeout']' in python main.py settings.")


# Will clean up the settings file and make each line executable in order to execute each line individually
def settings_read(filepath, add_default=False, default=''):
    settings_txt = open_read(filepath)
    settings_ltxt = txt_lib.comment_remove(settings_txt).split('\n')
    settings_ltxt = txt_lib.ltxt_clean(settings_ltxt,',')
    if add_default:
       for i,line in enumerate(settings_ltxt):
           test_line = line.split('=')
           if len(test_line) == 1:
               settings_ltxt[i] = line+"=%s"%str(default)
           elif line.split('=')[1] == '':
               settings_ltxt[i] = line+str(default)

    return settings_ltxt


# Changes variable names in the vmd script
def vmd_variable_writer(step_info, PID):
    txt = txt_lib.comment_remove(open_read(step_info['vmd_temp']))
    for i in step_info['tcl']:
        val = step_info['tcl'][i]
        if type(val) == dict:
            val = val[PID]
        txt = txt.replace("$%s"%str(i),str(val) )
    open_write(step_info['vmd_script'][PID], str(txt) )


# Prints on a same line
def print_same_line(message, sys, print_func):
   if sys != False:
     sys.stdout.flush()
     print_func("\r%s"%(message), sep="", end="\r")
   else:
     print_func("\rTrajectory %i/%i"%(step+1, max_steps))


# Checks whether VMD has finished by trying to read the vmd output to prevent race conditions
def vmd_finished_check(vmd_junk_fname):
    txt = open_read(vmd_junk_fname).split('\n')
    if '' in txt:
        txt.remove('')
    test = sum([1 for i in txt[-5:] if 'exit' in i.lower() and 'normal' in i.lower()])
    return bool(test)


# Opens and write a string to a file
def open_write(filename, message, mkdir=False, TyPe="w+"):
    folder_correct(filename, mkdir)
    f = open(filename, TyPe)
    f.write(message)
    f.close()


# Reads a file and closes it
def open_read(filename, throw_error=True, max_size=1):
    filename = folder_correct(filename)
    if path_leads_somewhere(filename):
        check_size = True
        try:
            import psutil
        except ModuleNotFoundError:
            check_size = False
        if check_size:
            if os.path.getsize(filename) >= psutil.virtual_memory().available * max_size:
               raise IOError("\n\nFilesize too big.\n\t* "  
                           +f"Filepath: '{filename}'" + "\n\t* "
                           +f"Avail Mem: {psutil.virtual_memory().available}" + "\n\t* "
                           +f"Filesize:  {os.path.getsize(filename)}" + "\n\n\n")

        with open(filename, 'r') as f:
            txt = f.read()
        return txt
    else:
        if throw_error:
            EXC.ERROR("The %s file doesn't exist!"%filename)
        return ''

# Checks if a filepath or folderpath exists
def path_leads_somewhere(path):
    if os.path.isfile(path) or os.path.isdir(path):
        return True
    else:
        return False


#Checks if the directory exists and makes it if not
def check_mkdir(path, max_depth=2):
    path = folder_correct(path)
    lpath = path.split('/')
    act_folders = []
    for i in range(2,len(lpath)):
        sub_path = '/'.join(lpath[:i])
        if not path_leads_somewhere(sub_path):
            act_folders.append(False)
        else:
            act_folders.append(True)
    if not all(act_folders[:-max_depth]):
        EXC.ERROR("Too many folders need to be created please check the filepaths, or increase the amount of folder I am allowed to create (check_mkdir).")
    else:
        for i in range(2,len(lpath)+1):
            sub_path = '/'.join(lpath[:i])
            if not os.path.isdir(sub_path) and '.' not in sub_path[sub_path.rfind('/'):]:
               os.mkdir(sub_path)
    return True


# Returns an absolute file/folder path. Will convert the relative file/folerpaths such as ../foo -> $PATH_TO_PYTHON_MINUS_1/foo
def folder_correct(f, make_file=False):
    f = os.path.expanduser(f)
    f = f.replace("//",'/')
    folder = os.path.isdir(f)
    if '/' not in f:
        f = './'+f
    flist = f.split('/')
    clist = os.getcwd().split('/')
    if flist[0] != clist[0] and flist[1] != clist[1]:
        cind, find = 0, 0
        for i in flist:
            if i == '..':
                cind += 1
                find += 1
            if i == '.':
                find += 1
        if cind != 0:
            clist = clist[:-cind]
        flist = flist[find:]
    else:
        clist = []
    if flist[-1] != '' and folder or (not path_leads_somewhere('/'.join(clist+flist)) and '.' not in f[f.rfind('/'):]):
        flist.append('')
    f= '/'.join(clist+flist)
    if make_file:
        if not path_leads_somewhere(f):
            if folder or '.' not in f[f.rfind('/'):]:
                check_mkdir(f)
            else:
                if not path_leads_somewhere(f[:f.rfind('/')]):
                    check_mkdir(f[:f.rfind('/')])
                File = open(f, 'a+')
                File.close()
        return f
    else:
        return f


# Prints on a same line
def print_same_line(message, sys, print_func):
   if sys != False:
     sys.stdout.flush()
     print_func("\r%s"%(message), sep="", end="\r")
   else:
     print_func(message+'\r')


# Writes a single step of an xyz file (data should be a numpy array with structure [[x1,y1,z1], ... , [xn,yn,zn]] )
def xyz_step_writer(positions, atom_nums, timestep, step, filepath, conv=0.52918):
    natom = len(positions)
    positions *= conv
    if natom != len(atom_nums):
        EXC.ERROR("The length of the positions array and atomic numbers array in the xyz_writer are not the same. Please fix this to use the 'background_mols' feature.\n\tlen(positions) = %i\n\tlen(atom_nums) = %i"%(natom, len(atom_nums)))
    s = "%i\ni =  %i, time =    %.3f\n"%(natom,step,timestep)
    s += '\n'.join(['\t'.join([str(atom_nums[i])]+pos.tolist()) for i, pos in enumerate(positions.astype(str))])
    open_write(filepath, s)


# Extracts the AOM_coeffs from the AOM_coeffs file
def AOM_coeffs(filename, atoms_per_site ):
    ltxt = open_read(filename).split('\n')
    ltxt = [i.split(' ') for i in ltxt]
    ltxt = [[j for j in i if j] for i in ltxt if i]
    ltxt = [i for i in ltxt if i]
    dtxt = {}
    if any('XX' in i for i in ltxt):
        mol_info = {}
        mol_count, atom_count = 0,0
        for i,item in enumerate(ltxt):
            if item[0] != "XX":
                atom_count += 1
                mol_info[i] = mol_count
                if (atom_count%atoms_per_site == 0):
                    mol_count += 1
    else:
        mol_info = False

    Acount = 0
    for i,j in enumerate(ltxt):
         if any(at in j[0].lower() for at in ['c','h']):
            dtxt[i] = (float(j[-1]), Acount)
            Acount += 1
    return np.array([float(i[-1]) for i in ltxt]), dtxt, mol_info


# Will print out the timings info from the times dict
def times_print(times,step, max_len, tot_time=False):
   if tot_time:
      print(txt_lib.align('Function', 28, 'l'), " | ",
      txt_lib.align("CPU Time [s]",13,'r'),
      " | Percentage of Total   *", sep="")
      print("-"*28+"-|-"+"-"*13+"-|-"+"-"*22,"*",sep="")
      for i in times:
         if times[i][step] > 0:
            print(txt_lib.align(txt_lib.align(i, 28, 'l')+ " | "+
            txt_lib.align(format(times[i][step], ".3f"),13,'r')+
            " | %.2g"%(times[i][step]*100/tot_time),69,'l' )+"*")
   else:
       print(txt_lib.align('Function', 27, 'l'), " | ",
       txt_lib.align("CPU Time [s]",13,'r'), sep="")
       print("-"*27+"-|-"+"-"*13)
       for i in times:
           if times[i][step] > 0:
               print(txt_lib.align(i, 27, 'l'), " | ",
               txt_lib.align(format(times[i][step], ".2g"),13,'r'), "%")


# Will take care of saving png images
def file_handler(i, extension, step_info):
    folderpath = folder_correct(step_info['img_fold']+step_info['title'])
    check_mkdir(folderpath)
    filename = "%s.%s"%(str(i), extension)
    return folderpath, filename, folderpath+filename


# Opens and write a string to a file
def open_write(filename, message, mkdir=False, type_open='w+'):
    folder_correct(filename, mkdir)
    if not path_leads_somewhere:
        f = open(filename, 'w+')
    else:
        f = open(filename, type_open)
    f.write(message)
    f.close()


# Creates the data and img folders
def create_data_img_folders(step_info):
    if not path_leads_somewhere(step_info['data_fold']):
        print ("Creating Data folder at:\n%s"%step_info['data_fold'])
        os.mkdir(step_info['data_fold'])

    if not path_leads_somewhere(step_info['img_fold']):
        print("Making a folder for images at:\n%s"%step_info['img_fold'])
        os.mkdir(step_info['img_fold'])


# Will change all the filenames in a folder to add leading zeros to them so
# alphabetesising them preserves order.
def add_leading_zeros(folder):
    tga_files = [i for i in os.listdir(folder) if '.tga' in i]
    dts = max([float(f.replace(",",".")[:f.find('_')]) for f in tga_files])
    if dts < 0.01 and dts > -0.01:
        num_leading_zeros = 1
    elif dts > 0:
        num_leading_zeros = int(np.floor(np.log10(dts)))
    else:
        raise SystemExit("Creating negative times???")

    new_files = []
    for f in tga_files:
        dt_str = f[:f.find('_')]
        dt = float(dt_str.replace(',','.'))
        if dt != 0:
          num_zeros_needed = num_leading_zeros - int(np.floor(np.log10(dt)))
        else:
           num_zeros_needed = num_leading_zeros
        new_dt = '0'*num_zeros_needed + "%.2f"%dt

        new_file = folder+f.replace(dt_str, new_dt.replace(".",","))
        os.rename(folder+f, new_file)
        new_files.append(new_file)

    return new_files



# Find which file inputs aren't in the folder
def find_missing_files(CP2K_output_files, all_files):
    bad_files = {}
    for f in CP2K_output_files:
        if type(CP2K_output_files[f]) == str:
            if CP2K_output_files[f] in all_files:
                bad_files[f] = [True]
            else:
                bad_files[f] = [False]
        elif any(type(CP2K_output_files['pos']) == j for j in (list, range)):
            if bad_files.get(f):
                if CP2K_output_files[f] in all_files:
                    bad_files[f].append(True)
                else:
                    bad_files[f].append(False)
            else:
                if CP2K_output_files[f] in all_files:
                    bad_files[f] = [True]
                else:
                    bad_files[f] = [False]
    return bad_files


# Will find the files using fuzzy logic.
def fuzzy_file_find(path):
    all_files = np.sort(os.listdir(path))

    strs_to_match = {'pvecs' : ['.xyz', 'pvec'],
                     'pos'   : ['.xyz', 'n-pos'],
                     'coeff' : ['.xyz', 'n-coeff'],
                     'AOM'   : ['AOM', 'COEFF'],
                     'inp'   : ['run.inp']}
    fuzzy_files = {ftype:[f for f in all_files if all(j in f for j in strs_to_match[ftype]) and '.' != f[0] and '.sw' not in f] for ftype in strs_to_match}

    # Pos and Coeffs need same num of steps
    if any(len(fuzzy_files['pos']) != len(fuzzy_files[i]) for i in ['pos','coeff']):
        files_with_numbers = ["I have found %i files for the %s"%(len(fuzzy_files[i]), i) for i in fuzzy_files]
        files_with_numbers = "\n\t*"+ "\n\t*".join(files_with_numbers)
        raise SystemExit("Sorry I can't find the same number of files for pvecs, posistions and coefficients:%s"%files_with_numbers)

    # Check if we have certain files
    if any(len(fuzzy_files[i]) == 0 for i in fuzzy_files if i != 'pvecs'):
        no_files = [i for i in fuzzy_files if len(fuzzy_files[i]) == 0 ]
        raise SystemExit("Sorry I can't seem to find any files for \n\t* %s"%('\n\t* '.join(no_files)))

    if not fuzzy_files['pvecs']:
        fuzzy_files['pvecs'] = ['CREATE']

    return fuzzy_files


# Will write the original settings (with typos fixed etc...) to a settings file
def write_cleaned_orig_settings(orig_settings_dict, settings_file):
    #settings_whitelist = ['calibrate', 'load_in_vmd', 'timestep_to_render','show_img_after_vmd','path','atoms_to_plot','start_step', 'end_step','stride']
    s = ''
    for sett_name in orig_settings_dict:
        # If type is list or array then put square brackets and commas in the settings file.
        if typ.is_list_array(orig_settings_dict[sett_name][0]):
            if all(i == j for i,j in zip(orig_settings_dict[sett_name][0], dft.defaults[sett_name])):
                continue
            s += "%s = [%s] %s\n"%(sett_name, ','.join(["%.2g"%i for i in orig_settings_dict[sett_name][0]]), orig_settings_dict[sett_name][1])
            continue

        # If the sett_name is int then it is a carriage return
        if type(sett_name) == int:
            s += orig_settings_dict[sett_name][1] + '\n'
            continue

        ## Remove settings from file that are the defaults (apart from certain ones in whitelist)
        #if orig_settings_dict[sett_name][0] == dft.defaults.get(sett_name) and sett_name not in settings_whitelist:
        #    continue

        # Need to put quotations around strings
        if type(orig_settings_dict[sett_name][0]) == str:
            s += "%s = '%s' %s\n"%(sett_name, orig_settings_dict[sett_name][0].strip("'").strip('"'), orig_settings_dict[sett_name][1])
        else:
            s += '%s = %s %s\n'%(sett_name, orig_settings_dict[sett_name][0], orig_settings_dict[sett_name][1])
    with open(settings_file, 'w') as f:
        f.write(s.strip('\n'))


# Plots the data, saves the graph and then appends the 2 images side by side.
def plot(step_info, mind, files, plt, optX=[], optY=[]):
    start_plot = mind-step_info['max_graph_data']
    if start_plot < 0:
       start_plot = 0
    fig, A = plt.subplots(1,1,facecolor=(1,1,1),figsize=(8.4*step_info['graph_img_ratio'],8.4))
    A.spines['top'].set_visible(False)
    A.spines['right'].set_visible(False)
    A.grid(color=[0.8]*3,ls='-')
    A.set_ylabel(step_info['ylab'],fontsize=step_info['yfont'])
    A.set_xlabel(step_info['xlab'],fontsize=step_info['xfont'])
    highlighted_mols = step_info['highlighted_mols']
    if type(step_info['highlighted_mols']) == str:
       if 'max' in step_info['highlighted_mols']:
          highlighted_mols = [i for i in xrange(len(step_info['mol'][mind,:])) if step_info['mol'][mind,i] == np.max(step_info['mol'][mind,:])]
       if 'min' in step_info['highlighted_mols']:
          highlighted_mols = [i for i in xrange(len(step_info['mol'][mind,:])) if step_info['mol'][mind,i] == np.min(step_info['mol'][mind,:])]
       elif 'al' in step_info['highlighted_mols']:
          highlighted_mols = range(len(step_info['mol'][mind,:]))
    replace_str = ','.join([str(i+1) for i in highlighted_mols])
    G_title = step_info['graph_title'].replace("*", replace_str)
    A.set_title(G_title, fontsize=step_info['tfont'])
    mols_plotted = []
    for moli in step_info['AOM_D']:
       mol_ind = step_info['mol_info'][moli]
       if mol_ind not in mols_plotted:
          if mol_ind in highlighted_mols:
             A.plot(step_info['Mtime-steps'][start_plot:mind], step_info['mol'][start_plot:mind,mol_ind],label="mol %i"%(mol_ind+1))
          else:
             A.plot(step_info['Mtime-steps'][start_plot:mind],step_info['mol'][start_plot:mind,step_info['mol_info'][moli]], alpha=0.2)
          _,_,graph_filepath = file_handler(files['name'], "png", step_info)
          mols_plotted.append(mol_ind)
    A.plot(optX,optY)
    A.legend(loc='best')

    #plt.tight_layout()
    fig.savefig(graph_filepath, format='png')
    add_imgs_cmd = "convert %s %s +append %s"%(graph_filepath, files['tga_fold'], files['tga_fold'])
    os.system(add_imgs_cmd)

    plt.close()
    return graph_filepath
