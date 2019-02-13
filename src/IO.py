'''
This large module contains many functions that are involved in the input/output
operations in the code. These are things like reading the xyz file, writing the
cube file, writing the xyz file, a generic open_read function and open_write.
'''
from __future__ import print_function

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

# Finds the number of lines in one step of the xyz file data
def find_num_lines_in_xyz_file_step(ltxt, filename):
    first_line = ltxt[0]
    num_lines_in_step = 1
    for i in ltxt[2:]: # Loops over all the line of text
        num_lines_in_step += 1
        #If any lines are very similar to the first line then assume the step is repeating
        if dfl.SequenceMatcher(None, first_line, i).ratio() > 0.8:
            return num_lines_in_step
    EXC.ERROR("Unable to determine number of steps in:\n\n%s"%filename)

# Finds the number of title lines and number of atoms with a step
def find_num_title_lines(step): # should be the text in a step split by line
    num_title_lines = 0
    for line in step:
        if typ.is_atom_line(line):
            break
        num_title_lines += 1
    return num_title_lines

# Finds the delimeter for the time-step in the xyz_file title
def find_time_delimeter(step, filename):
    for linenum,txt in enumerate(step):
        txt = txt.lower()
        if 'time' in txt:
            break
    txt = txt[txt.lower().find('time')+4:]
    cond1 = False
    cond2 = True
    for char in txt:
        cond1 = (typ.is_num(char) or char == '.')
        if cond1 == True:
            cond2 = False
        if not any([cond1, cond2]) and char != " ":
            return char,linenum
    EXC.ERROR("Cannot find the delimeter for the time-step info in the following xyz_file:\n\n%s\n\nstep = %s"%(filename,step))

# Will get necessary metadata from an xyz file such as time step_delim, lines_in_step etc...
# This will also create the step_data dictionary with the data of each step in
def get_xyz_step_metadata(filename, ltxt=False):
    if ltxt == False:
        ltxt = open_read(filename).split('\n')
    most_stable = False
    if any('*' in i for i in ltxt[:300]):
        most_stable = True
    if not most_stable:
        num_title_lines, num_atoms = txt_lib.num_atoms_find(ltxt)
        lines_in_step = num_title_lines + num_atoms
        if len(ltxt) > lines_in_step+1: # take lines from the second step instead of first as it is more reliable
           step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
        else: #If there is no second step take lines from the first
           step_data = {1:ltxt[:lines_in_step]}
    else:
        lines_in_step = find_num_title_lines(ltxt, filename)
        step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
        num_title_lines = find_num_title_lines(step_data[1])

    nsteps = int(len(ltxt)/lines_in_step)
    time_delim, time_ind = find_time_delimeter(step_data[1][:num_title_lines], filename)
    timelines = [ltxt[time_ind+(i*lines_in_step)] for i in range(nsteps)]
    timesteps = np.array([txt_lib.string_between(line, "time = ", time_delim) for line in  timelines]).astype(np.float64)
    return {'time_delim':time_delim,
            'time_ind':time_ind,
            'lines_in_step':lines_in_step,
            'num_title_lines':num_title_lines,
            'nsteps':nsteps,
            'tsteps':timesteps}

# Reads an xyz_file
# Would like to create a mask here to avoid reading the atoms to ignore.
# This function is quite obscure and terse as this is a bottle neck in the code and has been optimised.
# It relies heavily on numpy arrays and list\dictionary comphrensions to give speed things up.
def read_xyz_file(filename, num_data_cols, min_step=0, max_step='all', stride=1, ignore_steps=[], do_timesteps=[], metadata=False):
    """
    Reads an xyz file.

    Inputs:
        * filename       =>  filename [str] -required
        * num_data_cols  =>  how many columns (on the right) are data [int] -required
        * min_step       => The minimum step to iterate over [int] -optional (default 0)
        * max_step       => The maximum step to iterate over [int, or 'all'] -optional (default 'all')
        * stride         => The stride to take when iterating over steps [int] -optional (default 1)
        * ignore_steps   => Step indices to ignore [list <int>] -optional (default [])
        * do_timesteps   => Timesteps to complete [list <float>] -optional (default [])
        * metadata       => The metadata of the xyz file [dict] -optional (default False)

    Outputs:
        tuple with data, data from the columns and the timesteps

    N.B. Will do timesteps that aren't in ignore timesteps, and are in
    do_timesteps (unless it is empty) in the list created by the min, max and
    stride.
    """
    num_data_cols = -num_data_cols
    ltxt = open_read(filename).split('\n')
    if not metadata:
        metadata = get_xyz_step_metadata(filename, ltxt)

    if max_step == 'all':              max_step = metadata['nsteps']
    if max_step > metadata['nsteps']:  max_step = metadata['nsteps']
    #First find the min, max, stride set of timesteps
    all_steps = metadata['tsteps'][np.arange(min_step, max_step, stride)]
    if len(do_timesteps) > 0:
        #Find the timesteps which are in do_timesteps and range
        common_timesteps = np.intersect1d(all_steps, do_timesteps)
        #Find the indices of these and remove steps to ignore.
        all_steps = np.searchsorted(metadata['tsteps'], common_timesteps)
    else: all_steps = range(len(all_steps)) # All steps needs to be indices here
    all_steps = [i for i in all_steps if i not in ignore_steps]
    timesteps = metadata['tsteps'][np.array(all_steps)]

    step_data = [i for i in all_steps]

    for i, step_num in enumerate(all_steps):
        step_data[i] = ltxt[step_num*metadata['lines_in_step']+metadata['time_ind']+1:(step_num+1)*metadata['lines_in_step']] # Get string data

    for i, step in enumerate(step_data):
        step_data[i] = [[k for k in j.split(' ') if k] for j in step] # Split and get seperate cols
    step_data = np.array(step_data)
    data = np.array([step[:,num_data_cols:] for step in step_data])
    data = data.astype(float)
    spare_info = [step[:,:num_data_cols] for step in step_data]
    return data, spare_info, timesteps

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

# Checks whether the tachyon path specified is the correct one.
def check_tachyon(tachyon_path, times=0):
    if not tachyon_path:
        return False
    tachyon_out =os.popen(tachyon_path, 'r').read()
    tachyon_spiel_ind = tachyon_out.lower().find("tachyon parallel/multiprocessor ray tracer")
    if tachyon_spiel_ind != -1:
        return True
    else:
       os.chmod(tachyon_path, 755)
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
        framerate = int(np.round(num_files/length,0))
        if framerate == 0:
            framerate = 1
        in_files = files_folder+files
        pre_flags = ""
    elif "*" in files and '.' in files:
        ext = files.split('.')[-1]
        print(ext)
        all_files = os.listdir(files_folder)
        all_files = [i for i in all_files if ext in i.split('.')[-1]] #removing files that don't have the correct extension
        num_files = len(all_files)
        framerate = int(np.round(num_files/length,0))
        if framerate == 0:
            framerate = 1
        pre_flags = '-pattern_type glob' # Glob type input files
        in_files = '"%s"'%(files_folder+files) #input files must be inside string
    else:
        EXC.ERROR("Input format for image files is incorrect.\nPlease input them in the format:\n\n\t'pre%0Xd.ext'\n\nwhere pre is the prefix (can be nothing), X is the number of numbers in the filename, and ext is the file extensions (e.g. png or tga).")
    if path_leads_somewhere(output_name+'.mp4'):  os.remove(output_name+'.mp4') #remove file before starting to prevent hanging on 'are you sure you want to overwrite ...'
    options = (ffmpeg_binary, pre_flags, framerate, in_files, Vcodec, Acodec, extra_flags, output_name, log_file, err_file)

    Stitch_cmd = "%s -f image2 %s -r %s -i %s -vcodec %s -acodec %s %s %s.mp4 > %s 2> %s"%options
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
    if vmd_log_text != False:
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
    # 2> %s &
    vmd_commnd = "vmd -nt -dispdev none -e %s > %s 2> %s &"%(step_info['vmd_script'][PID], step_info['vmd_junk'][PID], step_info['vmd_err'][PID])
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
def open_read(filename, throw_error=True):
    filename = folder_correct(filename)
    if path_leads_somewhere(filename):
        f = open(filename, 'r')
        txt = f.read()
        f.close()
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
            dtxt[i] = [float(j[-1]),Acount]
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
    dts = [float(f.replace(",",".")[:f.find('_')]) for f in tga_files]
    num_leading_zeros = int(np.floor(np.log10(np.max(dts))))
    for f in tga_files:
        dt_str = f[:f.find('_')]
        dt = float(dt_str.replace(',','.'))
        if dt != 0:
          num_zeros_needed = num_leading_zeros - int(np.floor(np.log10(dt)))
        else:
           num_zeros_needed = num_leading_zeros
        new_dt = '0'*num_zeros_needed + "%.2f"%dt
        os.rename(folder+f, folder+f.replace(dt_str, new_dt.replace(".",",")))

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
    if any(len(fuzzy_files['pvecs']) != len(fuzzy_files[i]) for i in ['pos','pvecs','coeff']):
        files_with_numbers = ["I have found %i files for the %s"%(len(fuzzy_files[i]), i) for i in fuzzy_files]
        files_with_numbers = "\n\t*"+ "\n\t*".join(files_with_numbers)
        raise SystemExit("Sorry I can't find the same number of files for pvecs, posistions and coefficients:%s"%files_with_numbers)
    if any(len(fuzzy_files[i]) == 0 for i in fuzzy_files):
        no_files = [i for i in fuzzy_files if len(fuzzy_files[i]) == 0 ]
        raise SystemExit("Sorry I can't seem to find any files for \n\t* %s"%('\n\t* '.join(no_files)))
    return fuzzy_files


# Will write the original settings (with typos fixed etc...) to a settings file
def write_cleaned_orig_settings(orig_settings_dict, settings_file):
    settings_whitelist = ['calibrate', 'load_in_vmd', 'calibration_step','show_img_after_vmd','path','atoms_to_plot','start_step', 'end_step','stride']
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

        # Remove settings from file that are the defaults (apart from certain ones in whitelist)
        if orig_settings_dict[sett_name][0] == dft.defaults.get(sett_name) and sett_name not in settings_whitelist:
            continue

        # Need to put quotations around strings
        if type(orig_settings_dict[sett_name][0]) == str:
            s += "%s = '%s' %s\n"%(sett_name, orig_settings_dict[sett_name][0].strip("'").strip('"'), orig_settings_dict[sett_name][1])
        else:
            s += '%s = %s %s\n'%(sett_name, orig_settings_dict[sett_name][0], orig_settings_dict[sett_name][1])
    with open(settings_file, 'w') as f:
        f.write(s.strip('\n'))

# Old Code
# # Wrapper for the full xyz_reader step for parellisation
# def perform_full_step_xyz(xyz_step_info):
#     istep = xyz_step_info[1]
#     xyz_step_info = xyz_step_info[0]
#     good_stuff, time_step = read_xyz_step(xyz_step_info['steps'], istep, xyz_step_info['time_step_delim'], xyz_step_info['atoms_to_ignore'])
#     if good_stuff:
#         xyz_step_info['atomic_numbers'].append([x[0:-xyz_step_info['num_data_cols']][0] if len(x[0:-xyz_step_info['num_data_cols']]) == 1 else x[0:-xyz_step_info['num_data_cols']] for x in good_stuff])
#         xyz_step_info['atomic_coords'].append([[x[tmp] for tmp in range(len(x)-xyz_step_info['num_data_cols'],len(x))] for x in good_stuff])
#         xyz_step_info['timesteps'].append(time_step)
#
# # Reads a single xyz step from a iterable of all steps
# def read_xyz_step(steps, istep, time_step_delim, atoms_to_ignore):
#     bad_step = False
#     time_step = txt_lib.text_search(steps[istep][0], "time", time_step_delim, error_on=False)[0].replace(' ','')
#     time_step = float(time_step.split('=')[-1])
#     current_time_step = steps[istep][1] #list of all the items in the current time-current_time_step
#     if atoms_to_ignore:
#         if type(atoms_to_ignore) == str:
#            atoms_to_ignore = list(atoms_to_ignore)
#         if type(atoms_to_ignore) != list:
#             EXC.ERROR("The variable 'atoms_to_ignore' is not a list! This should have been declared as either a list of strings or a single string")
#         good_stuff = [x for x in current_time_step if not any(atom.lower() in x.lower() for atom in atoms_to_ignore)]
#     else:
#         good_stuff = [x for x in current_time_step]
#     for item in good_stuff:
#         if "**" in item:
#             bad_step = True
#     if bad_step:
#         return False, time_step
#     return [[typ.atomic_num_convert(j, False) for j in tmp.split(' ') if j] for tmp in good_stuff],time_step
#
# #Reads the xyz_files, If the time isn't there but the iteration is then still work...
# def xyz_reader(filename, num_data_cols, atoms_to_ignore = [], time_step_delim=',', num_atomsO=False, start_step=0, end_step='all', stride=1, time_steps=False):
#     ltxt = open_read(filename).split('\n')
#     # Find necessary info about the file, how many title lines, how many atoms, and how many time-steps
#     start_atoms, num_atoms = txt_lib.num_atoms_find(ltxt)
#     if type(num_atomsO) == int:
#         num_atoms = num_atomsO
#     num_title_lines = start_atoms
#     lines_in_step = num_title_lines+num_atoms
#     max_steps = int(''.join(ltxt).lower().count("time"))
#     if type(end_step) == str:
#       if 'al' in end_step.lower():
#          end_step = max_steps
#     if end_step > max_steps:
#         end_step = max_steps
#     xyz_step_info = {'atomic_numbers':[], 'atomic_coords':[], 'timesteps':[]}
#     # Steps is a list (of tuples) containing data from all the Steps
#     #   the first item in the tuple is the title of the step, the second item is the data from that step.
#     xyz_step_info['steps'] = [(''.join(ltxt[i*lines_in_step:(i*lines_in_step)+num_title_lines]),ltxt[i*lines_in_step+num_title_lines:(i+1)*lines_in_step]) for i in range(start_step,end_step,stride)]
#     xyz_step_info['time_step_delim'] = time_step_delim
#     xyz_step_info['atoms_to_ignore'] = atoms_to_ignore
#     xyz_step_info['num_data_cols'] = num_data_cols
#     if type(time_steps) != bool:
#         xyz_step_info['steps'] = [xyz_step_info['steps'][i] for i in xyz_step_info['steps'] if float(txt_lib.text_search(xyz_step_info['steps'][i][0], "time", time_step_delim, error_on=False)[0].replace(' ','').split('=')[-1]) in time_steps]
#     for istep in range(len(xyz_step_info['steps'])): #loop over all the steps
#         perform_full_step_xyz((xyz_step_info,istep))
#     atomic_coords = np.array([np.array(i) for i in xyz_step_info['atomic_coords']])
#     atomic_numbers = np.array([np.array(i) for i in xyz_step_info['atomic_numbers']])
#     return np.array(atomic_coords), np.array(atomic_numbers), np.array(xyz_step_info['timesteps'])
