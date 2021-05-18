#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:32:22 2019

@author: oem
"""

from src import text as txt_lib
from src import IO as gen_io
from src import type as type_check

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


def atom_find_more_rigorous(ltxt):
    """
    Will try to find where the atom section starts and ends using some patterns in the file.

    The idea behind this function is we split the first 100 lines into words and find their
    length. The most common length is then assumed to be the length of the atom line and any
    line with this num of words is assumed to be an atom line.

    Inputs:
      * ltxt <list<str>> => The file_txt split by lines
    Outputs:
      * <int>, <int> Where the atom section starts and ends
    """
    if type_check.is_num(ltxt[0]):
        if type_check.is_int(float(ltxt[0])):
            nat = int(ltxt[0])
            if len(ltxt[1].split()) == len(ltxt[2].split()):
                line2_types = (type(type_check.eval_type(i)) for i in ltxt[1].split())
                line3_types = (type(type_check.eval_type(i)) for i in ltxt[2].split())
                for t1, t2 in zip(line2_types, line3_types):
                    if t1 != t2: return 2, int(ltxt[0])
                return 1, int(ltxt[0])
            return 2, int(ltxt[0])

    # Get the length of the last line -this will be the length of atom line.
    first_100_line_counts = [len(line.split()) for line in ltxt[:100]]
    unique_vals = set(first_100_line_counts)
    last_line_len = 0
    for i in first_100_line_counts[-1:0:-1]:
        if i != 0:
            last_line_len = i
            break

    # This means either we have 1 title line, 2 title lines but 1 has the same num words as the atom lines
    #    or 2 title lines and they both have the same length.
    # If this function needs to be more rigorous this can be modified.
    if len(unique_vals) == 2:
      pass

    # This means we haven't found any difference in any lines.
    if len(unique_vals) == 1:
       raise SystemError("Can't find the atom section in this file!")

    start = False
    for line_num, line in enumerate(ltxt):
        len_words = len(line.split())

        # Have started and len words changed so will end
        if len_words != last_line_len and start is True:
            atom_end = line_num
            break

        # Haven't started and len words changed so will start
        if len_words == last_line_len and start is False:
            start = True
            prev_line = False
            atom_start = line_num
    else:
      atom_end = len(ltxt)

    return atom_start, atom_end - atom_start


def get_num_data_cols(ltxt, filename, num_title_lines, lines_in_step):
    """
    Will get the number of columns in the xyz file that contain data. This isn't a foolproof method
    so if there are odd results maybe this is to blame.

    Inputs:
        * ltxt <list<str>> => A list with every line of the input file as a different element.
        * filename <str> => The path to the file that needs opening
        * num_title_lines <int> => The number of non-data lines
        * lines_in_step <int> => How many lines in 1 step of the xyz file
    Outputs:
        <int> The number of columns in the data section.
    """
    dataTxt = [ltxt[num_title_lines + (i*lines_in_step) : (i+1)*lines_in_step]
               for i in range(len(ltxt) // lines_in_step)]

    num_data_cols_all = []
    for step in dataTxt[:20]:
       for line in step:
          splitter = line.split()
          count = 0
          for item in splitter[-1::-1]:
             if not type_check.is_float(item):
                num_data_cols_all.append(count)
                break
             count += 1

    num_data_cols = max(set(num_data_cols_all), key=num_data_cols_all.count)
    return num_data_cols


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

    raise SystemError("Cannot find the delimeter for the time-step info in the following xyz_file:\n\n%s\n\nstep = %s"%(filename,step))

# Will get necessary metadata from an xyz file such as time step_delim, lines_in_step etc...
# This will also create the step_data dictionary with the data of each step in
def get_xyz_metadata(filename, ltxt=False):
    """
    Get metadata from an xyz file.

    This function is used in the reading of xyz files and will retrieve data necessary for reading
    xyz files.

    Inputs:
       * filename <str> => the path to the file that needs parsing
       * ltxt <list<str>> OPTIONAL => the parsed txt of the file with each line in a separate element.

    Outputs:
       <dict> A dictionary containing useful parameters such as how many atom lines, how to get the timestep, num of frames.
    """
    if ltxt == False:
        ltxt = gen_io.open_read(filename, max_size=0.3).split('\n')
    ltxt = list(filter(None, ltxt))

    # Check whether to use the very stable but slow parser or quick slightly unstable one
    most_stable = False
    if any('**' in i for i in ltxt[:300]):
        most_stable = True

    #if not most_stable:
    num_title_lines, num_atoms = atom_find_more_rigorous(ltxt)
    lines_in_step = num_title_lines + num_atoms

    if len(ltxt) > lines_in_step+1: # take lines from the second step instead of first as it is more reliable
       step_data = {i: ltxt[i*lines_in_step:(i+1)*lines_in_step] for i in range(1,2)}
    else: #If there is no second step take lines from the first
       step_data = {1:ltxt[:lines_in_step]}


    nsteps = int(len(ltxt)/lines_in_step)
    time_delim, time_ind = find_time_delimeter(step_data[1][:num_title_lines],
                                               filename)
    timelines = [ltxt[time_ind+(i*lines_in_step)] for i in range(nsteps)]
    timesteps = np.array([txt_lib.string_between(line, "time = ", time_delim) for line in  timelines]).astype(np.float64)
    num_data_cols = get_num_data_cols(ltxt, filename, num_title_lines, lines_in_step)
    return {'time_delim': time_delim,
            'time_ind': time_ind,
            'tsteps': timesteps,
            'lines_in_step': lines_in_step,
            'num_title_lines': num_title_lines,
            'num_data_cols': num_data_cols,
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
        metadata = get_xyz_step_metadata(filename, ltxt)
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
