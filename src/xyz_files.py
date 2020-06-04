#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:32:22 2019

A module containing methods relevant to xyz files.

The class XYZ_File at the top will store the data read from an xyz file and overload operators
in order to make the data manipulatable.

To read an xyz file use the function `read_xyz_file(filepath <str>)`

To write an xyz file use the function `write_xyz_file(xyz_data <np.array | list>, filepath <str>)`
"""

import re
import os
from collections import OrderedDict, Counter
import numpy as np

from src.io_utils import general_io as gen_io
from src.parsing import general_parsing as gen_parse
from src.system import type_checking as type_check


class XYZ_File(gen_io.DataFileStorage):
    """
    A container to store xyz data in.

    This class will store data from an xyz file and also overload the in_built
    mathematical operators e.g. adding, subtracting, multiplying etc... to allow
    easy manipulation of the data without loosing any metadata.

    Inputs/Attributes:
        * xyz_data <numpy.array> => The parsed xyz data from an xyz file.
        * cols <numpy.array> => The parsed column data from the xyz file.
        * timesteps <numpy.array> => The parsed timesteps from the xyz file.
    """
    metadata = {'file_type': 'xyz'}
    # write_precision = 5
    def __init__(self, filepath):
        super().__init__(filepath)

    def _parse_(self):
        """
        Will call the 'read_xyz_file' function to parse an xyz file.

        For more info on the specifics of reading the xyz file see 'read_xyz_file'
        below.
        """
        self.xyz_data, self.cols, self.timesteps = read_xyz_file(self.filepath)

        # Make sure the data are in numpy arrays
        self.cols, self.xyz_data = np.array(self.cols), np.array(self.xyz_data)
        self.timesteps = np.array(self.timesteps)

        # Get some metadata
        self.nstep = len(self.xyz_data)
        self.natom = self.xyz_data.shape[1]
        self.ncol = self.xyz_data.shape[2]

        # Set the metadata
        self.metadata['number_steps'] = self.nstep
        self.metadata['number_atoms'] = self.natom

    # Overload the str function (useful for displaying data).
    def __str__(self):
        # Create an array of spaces/newlines to add between data columns in str
        space = ["    "] * self.natom

        # Convert floats to strings (the curvy brackets are important for performance here)
        xyz = self.xyz_data.astype(str)
        xyz = (['    '.join(line) for line in step_data] for step_data in xyz)
        cols = np.char.add(self.cols[0], space)
        head_str = '%i\ntime = ' % self.natom
        s = (head_str + ("%.3f\n" % t) + '\n'.join(np.char.add(cols, step_data)) + "\n"
             for step_data, t in zip(xyz, self.timesteps))

        # Create the str
        return ''.join(s)


class Write_XYZ_File(gen_io.Write_File):
      """
      Will handle the writing of xyz files.

      The parent class gen_io.Write_File will handle the actual writing and
      this just creates an xyz file string.

     Inputs:
        * Data_Class <class> => The class containing all the data to be written
        * filepath <str>     => The path to the file to be written.
      """
      def __init__(self, Data_Class, filepath):
          # If we can set the xyz variables in the data class then set them
          required_data_fncs = ('get_xyz_data', 'get_xyz_cols', 'get_xyz_timesteps',)

          if all(j in dir(Data_Class) for j in required_data_fncs):
              self.xyz_data = Data_Class.get_xyz_data()
              self.cols = Data_Class.get_xyz_cols()
              self.timesteps = Data_Class.get_xyz_timesteps()
          else:
              non_implemented = ""
              for i in required_data_fncs:
                  if i not in dir(Data_Class): non_implemented += f"'{i}' "
              raise SystemError(f"\n\n\nPlease implement the methods {non_implemented} in {type(Data_Class)}\n\n")

          # Run standard file writing procedure
          super().__init__(Data_Class, filepath)

      def create_file_str(self):
          """
          Will create the string that contains an xyz file, this is save as self.file_txt.
          """
          all_lists = (self.cols, self.xyz_data, self.timesteps)
          if len(np.shape(self.xyz_data)) == 4:
  
              if len(self.cols) != len(self.xyz_data) != self.timesteps:
                raise SystemError("\n\nThe length of the cols, xyz_data and timesteps arrays are different.\n\n"
                                  + "These arrays should all be the same length and should contain info for each file to write.")
              
              all_file_strings = [self.create_single_file_str(xyz_data, cols, timesteps)
                                  for cols, xyz_data, timesteps in zip(self.cols, self.xyz_data, self.timesteps)]  

          # Create the str
          else:
              all_file_strings = self.create_single_file_str(self.xyz_data, self.cols, self.timesteps)


          return all_file_strings

      def create_single_file_str(self, xyz_data, cols, timesteps):
          """Will create the xyz file string for a single file."""

          # Create an array of spaces/newlines to add between data columns in str
          natom = len(cols[0])
          space = ["    "] * natom
  
          # Convert floats to strings (the curvy brackets are important for performance here)
          xyz = xyz_data.astype(str)
          xyz = (['    '.join(line) for line in step_data] for step_data in xyz)
          cols = np.char.add(cols[0], space)
          head_str = '%i\ntime = ' % natom

          s = (head_str + ("%.3f\n" % t) + '\n'.join(np.char.add(cols, step_data)) + "\n"
               for step_data, t in zip(xyz, timesteps))
  
          return ''.join(s)

def string_between(Str, substr1, substr2):
    """
    Returns the string between 2 substrings within a string e.g. the string between A and C in 'AbobC' is bob.

    Inputs:
      * line <str> => A line from the xyz file

    Outputs:
      <str> The string between 2 substrings within a string.
    """
    Str = Str[Str.find(substr1)+len(substr1):]
    Str = Str[:Str.find(substr2)]
    return Str

def is_atom_line(line):
    """
    Checks if a line of text is an atom line in a xyz file

    Inputs:
      * line <str> => A line from the xyz file

    Outputs:
      <bool> Whether the line contains atom data
    """
    words = line.split()
    # If the line is very short it isn't a data line
    if not len(words):
        return False

    # Check to see how many float there are compared to other words
    fline = [len(re.findall("[0-9]\.[0-9]", i)) > 0 for i in words]
    percentFloats = fline.count(True) / float(len(fline))
    if percentFloats < 0.5:
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
    # Get the most common length of line -this will be the length of atom line.
    first_100_line_counts = [len(line.split()) for line in ltxt[:100]]
    unique_vals = set(first_100_line_counts)
    modal_val = max(unique_vals, key=first_100_line_counts.count)

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
        if len_words != modal_val and start is True:
            atom_end = line_num
            break

        # Haven't started and len words changed so will start
        if len_words == modal_val and start is False:
            start = True
            prev_line = False
            atom_start = line_num
    else:
      atom_end = len(ltxt)

    return atom_start, atom_end - atom_start



# Finds the number of title lines and number of atoms with a step
def find_num_title_lines(step): # should be the text in a step split by line
    """
    Finds the number of title lines and number of atoms with a step

    Inputs:
       * step <list<str>> => data of 1 step in an xyz file.

    Outputs:
       <int> The number of lines preceeding the data section in an xyz file.
    """
    num_title_lines = 0
    for line in step:
        if is_atom_line(line):
            break
        num_title_lines += 1

    return num_title_lines


# Finds the delimeter for the time-step in the xyz_file title
def find_time_delimeter(step, filename):
    """
    Will find the delimeter for the timestep in the xyz file title.

    Inputs:
       * step <list<str>> => data of 1 step in an xyz file.
       * filename <str> => the name of the file (only used for error messages)

    Outputs:
       <str> The time delimeter used to signify the timestep in the xyz file.
    """
    # Check to see if we can find the word time in the title lines.
    for linenum, txt in enumerate(step):
        txt = txt.lower()
        if 'time' in txt:
            break
    else:
        return [False] * 2

    # find the character before the first number after the word time.
    prev_char, count = False, 0
    txt = txt[txt.lower().find("time"):]
    for char in txt.replace(" ",""):
        isnum = (char.isdigit() or char == '.')
        if isnum != prev_char:
            count += 1
        prev_char = isnum
        if count == 2:
            break
    if char.isdigit(): return '\n', linenum
    else: return char, linenum
    return [False] * 2


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
        ltxt = gen_io.open_read(filename).split('\n')
    # Check whether to use the very stable but slow parser or quick slightly unstable one
    most_stable = False
    if any('**' in i for i in ltxt[:300]):
        most_stable = True

    if not most_stable:
        num_title_lines, num_atoms = atom_find_more_rigorous(ltxt)
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
    """
    Splits each string in a list of strings.
    """
    return [j.split() for j in i]


def read_xyz_file(filename, num_data_cols=False,
                  min_time=0, max_time='all', stride=1,
                  ignore_steps=[], do_timesteps=[], metadata=False):
    """
    Will read 1 xyz file with a given number of data columns.

    Inputs:
        * filename => the path to the file to be read
        * num_data_cols => the number of columns which have data (not metadata)
        * min_time => time to start reading from
        * max_time => time to stop reading at
        * stride => what stride to take when reading
        * ignore_steps => a list of any step numbers to ignore.
        * do_timesteps   => Timesteps to complete [list <float>] -optional (default [])
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
    ltxt = [i for i in gen_io.open_read(filename).split('\n') if i]

    if metadata is False:
        metadata = get_xyz_metadata(filename, ltxt)
        if num_data_cols is False:
           num_data_cols = -metadata['num_data_cols']
    lines_in_step = metadata['lines_in_step']
    num_title_lines = metadata['num_title_lines']
    get_timestep = metadata['time_ind'] is not False
    if get_timestep:
       time_ind = metadata['time_ind']
       time_delim = metadata['time_delim']
    num_steps = metadata['nsteps']

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
    if get_timestep:
       timelines = (ltxt[time_ind+(i*lines_in_step)] for i in all_steps)
       timesteps = (string_between(line, "time = ", time_delim)
                    for line in timelines)
       timesteps = (gen_parse.get_nums_in_str(i, True) for i in timesteps)
       timesteps = np.array([i[0] if len(i) == 1 else 0.0 for i in timesteps])
       timesteps = timesteps.astype(float)

    else:
       print("***********WARNING***************\n\nThe timesteps could not be extracted\n\n***********WARNING***************")
       timesteps = [0] * len(all_steps)

    # Get the correct steps (from min_time and max_time)
    all_steps = np.array(all_steps)
    if get_timestep:
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
    # tmp = np.array([[len(i.split()) for i in j] for j in step_data])
    # print(step_data[tmp != 3])
    step_data = np.apply_along_axis(splitter, 1, step_data)
    data = step_data[:, :, num_data_cols:].astype(float)

    # If there is only one column in the cols then don't create another list!
    if (len(step_data[0, 0]) + num_data_cols) == 1:
        cols = step_data[:, :, 0]
    else:
        cols = step_data[:, :, :num_data_cols]

    return data, cols, timesteps
