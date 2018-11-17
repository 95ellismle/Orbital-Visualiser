'''
This module contains all the functions which handle I/O in the code.
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
import Templates.permanent_settings as ps

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

# Reads an xyz_file
# Would like to create a mask here to avoid reading the atoms to ignore.
# This function is quite obscure and terse as this is a bottle neck in the code and has been optimised.
# It relies heavily on numpy arrays and list\dictionary comphrensions to give speed things up.
def read_xyz_file(filename, num_data_cols, min_step=0, max_step='all', stride=1):
    num_data_cols = -num_data_cols
    ltxt = open_read(filename).split('\n')
    most_stable = False
    if any('*' in i for i in ltxt[:300]):
        most_stable = True
    if not most_stable:
        num_title_lines, num_atoms = txt_lib.num_atoms_find(ltxt)
        lines_in_step = num_title_lines + num_atoms
        step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
    else:
        lines_in_step = find_num_title_lines(ltxt, filename)
        step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
        num_title_lines = find_num_title_lines(step_data[1])
    time_delim, time_ind = find_time_delimeter(step_data[1][:num_title_lines], filename)
    if max_step == 'all':
        max_step = int(len(ltxt)/lines_in_step)
    # The OrderedDict doesn't seem to have major overheads as dictionary access aren't the main bottleneck here.
    # It is also much easier to use!
    step_data = OrderedDict() # The OrderedDict keeps the order of the frames for saving etc...
    for i in range(min_step, max_step, stride):
        step_data[i] = ltxt[i*lines_in_step:(i+1)*lines_in_step]
        step_data[i] = (step_data[i][:num_title_lines],step_data[i][num_title_lines:])
    time_steps = np.array([txt_lib.string_between(step_data[i][0][time_ind], "time = ", time_delim) for i in step_data]).astype(float)
    for i in range(min_step, max_step, stride):
        step_data[i] = [j.split(' ') for j in step_data[i][1]]
        step_data[i] = np.array([[k for k in j if k] for j in step_data[i]])
    data = np.array([step_data[i][:,num_data_cols:] for i in step_data]).astype(float)
    spare_info = [step_data[i][:,:num_data_cols] for i in step_data]
    return data, spare_info, time_steps

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
def check_tachyon(tachyon_path):
    tachyon_out =os.popen(tachyon_path, 'r').read()
    tachyon_spiel_ind = tachyon_out.lower().find("tachyon parallel/multiprocessor ray tracer")
    if tachyon_spiel_ind != -1:
        return True
    return False

# Reads and writes the updated permanent settings file
def read_write_perm_settings(filepath, setting, value):
    txt = open_read(filepath)
    new_txt = txt_lib.change_perm_settings(txt, setting, value)
    open_write(filepath, new_txt,'w')

# Searches recursively for the VMD tachyon renderer path.
def find_tachyon(current_tachyon_path=''):
    if check_tachyon(current_tachyon_path):
        return current_tachyon_path
    vmd_path = sb.check_output("which vmd", shell=True)
    vmd_path = str(vmd_path, 'utf-8')
    for N in range(1,vmd_path.count('/')-1):
        vmd_path = txt_lib.folderpath_back_N(vmd_path,N)
        tachyon_path  = sb.check_output("find %s -name '*tachyon_*'"%vmd_path, shell=True)
        if tachyon_path:
            break
    if not tachyon_path:
        return False
    tachyon_path = str(tachyon_path, 'utf-8').replace('\n','')
    return tachyon_path

# Will stitch together an mp4 video
def stitch_mp4(files, files_folder, output_name, length, ffmpeg_binary, Acodec='aac', Vcodec='libx264', extra_flags="-pix_fmt yuv420p -preset slow", log_file="a.log", err_file="a.err"):
    if all(i in files for i in ['%','d','.']):
        ext = files.split('.')[-1]
        num_of_nums = eval(files.split('%')[-1].split('d')[0].strip('0'))
        prefix = files.split('%')[0]
        all_files = os.listdir(files_folder)
        all_files = [i for i in all_files if ext in i.split('.')[-1]] #removing files that don't have the correct extension
        all_files = [i for i in all_files if len(i[len(prefix):i.find('.')]) == num_of_nums] # only files with the correct amount of nums
        num_files = len(all_files)
        framerate = num_files/length
    else:
        EXC.ERROR("Input format for image files is incorrect.\nPlease input them in the format:\n\n\t'pre%0Xd.ext'\n\nwhere pre is the prefix (can be nothing), X is the number of numbers in the filename, and ext is the file extensions (e.g. png or tga).")
    options = (ffmpeg_binary, framerate, files_folder+files, Vcodec, Acodec, extra_flags, output_name, log_file, err_file)
    Stitch_cmd = "%s -f image2  -r %s -i %s -vcodec %s -acodec %s %s %s.mp4 > %s 2> %s"%options
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
def settings_update(step_info):
    vmd_log_text = open_read(step_info['vmd_log_file'], False)
    script_text = open_read(step_info['vmd_script'][list(step_info['vmd_script'].keys())[0]])
    if vmd_log_text != False:
         os.remove(step_info['vmd_log_file'])
         log_ltxt = txt_lib.vmd_log_file_parse(vmd_log_text, script_text, step_info)
         log_ltxt.append('\n')
         if use_prev_scaling(step_info['path']):
             open_write(step_info['tcl']['vmd_source_file'], '\n'.join(log_ltxt[1:]),type_open='a')
         else:
             open_write(step_info['tcl']['vmd_source_file'], '\n'.join(log_ltxt[1:]),type_open='w+')
         write_settings_file(step_info)
         return step_info
    else:
         EXC.WARN("VMD hasn't created a logfile!", step_info['verbose_output'])
         return step_info

# Writes the settings file with updated info from the step_info dict
def write_settings_file(step_info):
    settings_to_write = ''
    for i in step_info['orig_settings']:
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
    print("\n\n\nVMD_SCRIPT = ",step_info['vmd_script'][PID], "\n\n\n")
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
        return False

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
def file_handler(i, Type, step_info):
    folderpath = folder_correct(step_info['img_fold']+step_info['Title'])
    check_mkdir(folderpath)
    filename = "%s.%s"%(str(i),Type)
    return folderpath, filename, folderpath+filename

# Concatenates a string up to a specified substring
def remove_substr_after_str(txt, substr):
    ind = txt.find(substr)
    if ind < 0:
        ind = len(txt) + 1 + ind
    return txt[:ind]

# Opens and write a string to a file
def open_write(filename, message, mkdir=False, type_open='w+'):
    folder_correct(filename, mkdir)
    if not path_leads_somewhere:
        f = open(filename, 'w+')
    else:
        f = open(filename, type_open)
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
        return False

# Creates the data and img folders
def create_data_img_folders(step_info):
    if not path_leads_somewhere(step_info['data_fold']):
        print ("Creating Data folder at:\n%s"%step_info['data_fold'])
        os.mkdir(step_info['data_fold'])

    if not path_leads_somewhere(step_info['img_fold']):
        print("Making a folder for images at:\n%s"%step_info['img_fold'])
        os.mkdir(step_info['img_fold'])

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
