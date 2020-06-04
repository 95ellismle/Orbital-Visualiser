#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:32:22 2019

@author: oem
"""

from src import text as txt_lib


import os
import difflib as dfl
from collections import OrderedDict
import numpy as np


# Checks if a line of text is an atom line in a xyz file
def is_atom_line(line):
    line = [i for i in line.split(' ') if i]
    if len(line) < 3:
        return False
    fline = [is_num(i) for i in line]
    if not sum(fline[-3:]) == 3:
        return False
    else:
        return True

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
    for linenum,txt in enumerate(step):
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
    return time_delim, time_ind, lines_in_step, num_title_lines


# Will get necessary metadata from an xyz file such as time step_delim, lines_in_step etc...
# This will also create the step_data dictionary with the data of each step in
def get_xyz_step_metadata2(filename, ltxt=False):
    if ltxt == False:
        ltxt = open_read(filename).split('\n')
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
    return {'time_delim': time_delim,
            'time_ind': time_ind,
            'lines_in_step': lines_in_step,
            'num_title_lines': num_title_lines,
            'nsteps': nsteps}

def _get_timesteps_(ltxt, all_steps, metadata, do_timesteps=[]):
    """
    Will get the timesteps from the xyz file and ammend the all steps array.

    Inputs:
        * ltxt <list<str>>           => The file txt split by '\n'
        * all_steps  <list<int>>     => The step numbers
        * metadata <dict>            => The xyz file metadata
        * do_timesteps <list<float>> => Which timesteps to do
    """
    # Get some metadata vars
    lines_in_step = metadata['lines_in_step']
    num_title_lines = metadata['num_title_lines']
    time_ind = metadata['time_ind']
    time_delim = metadata['time_delim']

    # Get timesteps
    timelines = np.array([ltxt[time_ind+(i*lines_in_step)] for i in all_steps])
    timesteps = [txt_lib.string_between(line, "time = ", time_delim)
                 for line in timelines]
    timesteps = np.array(timesteps)
    timesteps = timesteps.astype(np.float32)

    # Apply the do_timesteps array
    if len(do_timesteps) > 0:
        mask = [i in do_timesteps for i in timesteps]
        if sum(mask) == 0:
            raise SystemExit("Can't find any steps in the 'do_timesteps' list.\n\t* "
                            +f"do_timesteps: {do_timesteps}.")
        timesteps = timesteps[mask]
        all_steps = all_steps[mask]

    return all_steps, timesteps

# Reads an xyz_file
def read_xyz_file(filename, num_data_cols,
                  min_step=0, max_step='all', stride=1,
                  ignore_steps=[], do_timesteps=[], metadata=False):
    """
    Will read 1 xyz file with a given number of data columns.


    Inputs:
        * filename => the path to the file to be read
        * num_data_cols => the number of columns which have data (not metadata)
        * min_step => step to start reading from
        * max_step => step to stop reading at
        * stride => what stride to take when reading
        * ignore_steps => a list of any step numbers to ignore.
        * do_timesteps   => Timesteps to complete [list <float>] -optional (default [])
        * metadata => optional dictionary containing the metadata

    Outputs:
        * data, cols, timesteps = the data, metadata and timesteps respectively
    """
    if not all(isinstance(j, (int, np.integer)) for j in (max_step, min_step, stride)):
        if type(max_step) != str:
            print("min_step = ", min_step, " type = ", type(min_step))
            print("max_step = ", max_step, " type = ", type(max_step))
            print("stride = ", stride, " type = ", type(stride))
            raise SystemExit("Input parameters are the wrong type!")

    num_data_cols = -num_data_cols
    ltxt = open_read(filename).split('\n')
    if metadata is False:
        metadata = get_xyz_step_metadata2(filename, ltxt)
    lines_in_step = metadata['lines_in_step']
    num_title_lines = metadata['num_title_lines']

    abs_max_step = int(len(ltxt)/lines_in_step)
    if max_step == 'all' or max_step > abs_max_step:
        max_step = abs_max_step

    # The OrderedDict is actually faster than a list here.
    step_data = OrderedDict()  # keeps order of frames -Important
    all_steps = np.array([i for i in range(min_step, max_step, stride)
                          if i not in ignore_steps])

    # Get the timesteps
    all_steps, timesteps = _get_timesteps_(ltxt, all_steps, metadata, do_timesteps)

    # Get the actual data
    for i in all_steps:
        step_data[i] = ltxt[i*lines_in_step:(i+1)*lines_in_step]
        step_data[i] = (step_data[i][:num_title_lines],
                        step_data[i][num_title_lines:])

    for i in all_steps:
        step_data[i] = [j.split() for j in step_data[i][1]]
        step_data[i] = np.array(step_data[i])

    step_data = np.array(list(step_data.values()))
    data = step_data[:, :, num_data_cols:].astype(float)

    # If there is only one column in the cols then don't create another list!
    if (len(step_data[0, 0]) + num_data_cols) == 1:
        cols = step_data[:, :, 0]
    else:
        cols = step_data[:, :, :num_data_cols]

    return data, cols, timesteps


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
