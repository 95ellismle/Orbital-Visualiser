#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:32:22 2019

@author: oem
"""

import re
import os
import difflib as dfl
from collections import OrderedDict
import numpy as np
from scipy.stats import mode

# Returns the substring between 2 substrings within a string
def string_between(Str, substr1, substr2):
    Str = Str[Str.find(substr1)+len(substr1):]
    Str = Str[:Str.find(substr2)]
    return Str

# Checks if a line of text is an atom line in a xyz file
def is_atom_line(line):
    line = [i for i in line.split(' ') if i]
    if not len(line):
        return False
    fline = [len(re.findall("[0-9]\.[0-9]", i)) > 0 for i in line]
    percentFloats = fline.count(True) / float(len(fline))
    if percentFloats < 0.5:
        return False
    else:
        return True

# Checks whether a number is a float or integer
def is_float(num):
   if type(num) == str:
      if is_num(num):
         if '.' in num:
            return True
   return False

# Checks whether a string can be a number
def is_num(Str):
    try:
        float(Str)
        return True
    except:
        return False

# Will determine the number of atoms in an xyz file
def num_atoms_find(ltxt):
    start_atoms, finish_atoms = 0,0
    for i,line in enumerate(ltxt):
        if (is_atom_line(line)) == True:
            start_atoms = i
            break
    for i,line in enumerate(ltxt[start_atoms:],start=start_atoms):
        if (is_atom_line(line) == False):
            finish_atoms=i
            break
    return start_atoms, finish_atoms-start_atoms

# Finds the number of lines in one step of the xyz file data
def find_num_lines_in_xyz_file_step(ltxt, filename):
    first_line = ltxt[0]
    num_lines_in_step = 1
    for i in ltxt[2:]: # Loops over all the line of text
        num_lines_in_step += 1
        #If any lines are very similar to the first line then assume the step is repeating
        if dfl.SequenceMatcher(None, first_line, i).ratio() > 0.8:
            return num_lines_in_step
    raise SystemExit("Unable to determine number of steps in:\n\n%s"%filename)

# Finds the number of title lines and number of atoms with a step
def find_num_title_lines(step): # should be the text in a step split by line
    num_title_lines = 0
    for line in step:
        if is_atom_line(line):
            break
        num_title_lines += 1
    return num_title_lines

# Finds the delimeter for the time-step in the xyz_file title
def find_time_delimeter(step, filename):
    for linenum, txt in enumerate(step):
        txt = txt.lower()
        if 'time' in txt:
            break
    else:
        raise SystemExit ("Can't find the word 'time' in this data:\n\n%s\n\n\tFilename:%s"%(str(step), filename) )
    prev_char, count = False, 0
    txt = txt[txt.find("time"):]
    for char in txt.replace(" ",""):
        isnum = (char.isdigit() or char == '.')
        if isnum != prev_char:
            count += 1
        prev_char = isnum
        if count == 2:
            break
    if char.isdigit(): return '\n', linenum
    else: return char, linenum
    raise SystemExit("Cannot find the delimeter for the time-step info in the following xyz_file:\n\n%s\n\nstep = %s"%(filename,step))


def get_num_data_cols(ltxt, filename, num_title_lines, lines_in_step):
   """
   Will get the number of columns in the xyz file that contain data. This isn't a foolproof method
   so if there are odd results maybe this is to blame.
   """
#   numIter = len(ltxt) // lines_in_step
   dataTxt = [ltxt[num_title_lines + (i*lines_in_step) : (i+1)*lines_in_step]
              for i in range(len(ltxt) // lines_in_step)]

   num_data_cols_all = []
   for step in dataTxt[:20]:
      for line in step:
         splitter = line.split()
         count = 0
         for item in splitter[-1::-1]:
            if not is_float(item):
               num_data_cols_all.append(count)
               break
            count += 1

   num_data_cols = mode(num_data_cols_all)[0][0]
   return num_data_cols


# Reads a file and closes it
def open_read(filename, throw_error=True):
    if os.path.isfile(filename):
        f = open(filename, 'r')
        txt = f.read()
        f.close()
        return txt
    else:
        if throw_error:
            raise SystemExit("The %s file doesn't exist!" % filename)
        return False


# Will get necessary metadata from an xyz file such as time step_delim, lines_in_step etc...
# This will also create the step_data dictionary with the data of each step in
def get_xyz_step_metadata(ltxt, filename):
    most_stable = False
    if any('*' in i for i in ltxt[:300]):
        most_stable = True
    if not most_stable:
        num_title_lines, num_atoms = num_atoms_find(ltxt)
        lines_in_step = num_title_lines + num_atoms
        if len(ltxt) > lines_in_step+1: # take lines from the second step instead of first as it is more reliable
           step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
        else: #If there is no second step take lines from the first
           step_data = {1:ltxt[:lines_in_step]}
    else:
        lines_in_step = find_num_title_lines(ltxt)
        step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
        num_title_lines = find_num_title_lines(step_data[1])
    time_delim, time_ind = find_time_delimeter(step_data[1][:num_title_lines], filename)
    num_data_cols = get_num_data_cols(ltxt, filename, num_title_lines, lines_in_step)
    return time_delim, time_ind, lines_in_step, num_title_lines, num_data_cols


# Will get necessary metadata from an xyz file such as time step_delim, lines_in_step etc...
# This will also create the step_data dictionary with the data of each step in
def get_xyz_step_metadata2(filename, ltxt=False):
    if ltxt == False:
        ltxt = open_read(filename).split('\n')
    # Check whether to use the very stable but slow parser or quick slightly unstable one
    most_stable = False
    if any('*' in i for i in ltxt[:300]):
        most_stable = True

    if not most_stable:
        num_title_lines, num_atoms = num_atoms_find(ltxt)
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
    time_delim, time_ind = find_time_delimeter(step_data[1][:num_title_lines],
                                               filename)
    num_data_cols = get_num_data_cols(ltxt, filename, num_title_lines, lines_in_step)
    return {'time_delim': time_delim,
            'time_ind': time_ind,
            'lines_in_step': lines_in_step,
            'num_title_lines': num_title_lines,
            'num_data_cols': num_data_cols,
            'nsteps': nsteps}


def splitter(i):
    return [j.split() for j in i]


# Reads an xyz_file
def read_xyz_file(filename, num_data_cols=False,
                  min_time=0, max_time='all', stride=1,
                  ignore_steps=[], metadata=False):
    """
    Will read 1 xyz file with a given number of data columns.

    Inputs:
        * filename => the path to the file to be read
        * num_data_cols => the number of columns which have data (not metadata)
        * min_time => time to start reading from
        * max_time => time to stop reading at
        * stride => what stride to take when reading
        * ignore_steps => a list of any step numbers to ignore.
        * metadata => optional dictionary containing the metadata

    Outputs:
        * data, cols, timesteps = the data, metadata and timesteps respectively
    """
    # Quick type check of param fed into the func
    if type(stride) != int and isinstance(min_step, (int, float)):
            print("min_time = ", min_time, " type = ", type(min_time))
            print("max_time = ", max_time, " type = ", type(max_time))
            print("stride = ", stride, " type = ", type(stride))
            raise SystemExit("Input parameters are the wrong type!")

    # Get bits of metadata
    if num_data_cols is not False:
       num_data_cols = -num_data_cols
    ltxt = [i for i in open_read(filename).split('\n') if i]
    if metadata is False:
        metadata = get_xyz_step_metadata2(filename, ltxt)
        if num_data_cols is False:
           num_data_cols = -metadata['num_data_cols']
    lines_in_step = metadata['lines_in_step']
    num_title_lines = metadata['num_title_lines']
    time_ind = metadata['time_ind']
    time_delim = metadata['time_delim']
    num_steps = len(ltxt) / metadata['lines_in_step']

    # Ignore any badly written steps at the end
    badEndSteps = 0
    for i in range(num_steps, 0, -1):
      stepData = ltxt[(i-1)*lines_in_step:(i)*lines_in_step][num_title_lines:]
      badStep = False
      for line in stepData:
         if '*********************' in line:
            badEndSteps += 1
            badStep = True
            break
      if badStep is False:
         break
            

    # The OrderedDict is actually faster than a list here.
    #   (time speedup at the expense of space)
    step_data = OrderedDict()  # keeps order of frames -Important
    all_steps = [i for i in range(0, num_steps-badEndSteps, stride)
                 if i not in ignore_steps]

    # Get the timesteps
    timelines = [ltxt[time_ind+(i*lines_in_step)] for i in all_steps]
    timesteps = [string_between(line, "time = ", time_delim)
                 for line in timelines]
    timesteps = np.array(timesteps)
    timesteps = timesteps.astype(float)

    # Get the correct steps (from min_time and max_time)
    all_steps = np.array(all_steps)
    mask = timesteps >= min_time
    if type(max_time) == str:
      if 'all' not in max_time.lower():
         msg = "You inputted max_time = `%s`\n" % max_time
         msg += "Only the following are recognised as max_time parameters:\n\t*%s" % '\n\t*'.join(['all'])
         print(msg)
         raise SystemExit("Unknown parameter for max_time.\n\n"+msg)
    else:
      mask = mask & (timesteps <= max_time)
    all_steps = all_steps[mask]
    timesteps = timesteps[mask]
      
    # Get the actual data (but only up to the max step)
    min_step, max_step = min(all_steps), max(all_steps)+1
    all_data = np.array(ltxt)[min_step*metadata['lines_in_step']:max_step*metadata['lines_in_step']]

    # get the data from each step in a more usable format
    step_data = np.reshape(all_data, (len(all_steps), lines_in_step))
    step_data = step_data[:, num_title_lines:]

    
    # This bit is the slowest atm and would benefit the most from optimisation
    step_data = np.apply_along_axis(splitter, 1, step_data)
    data = step_data[:, :, num_data_cols:].astype(float)

    # If there is only one column in the cols then don't create another list!
    if (len(step_data[0, 0]) + num_data_cols) == 1:
        cols = step_data[:, :, 0]
    else:
        cols = step_data[:, :, :num_data_cols]

    return data, cols, timesteps
