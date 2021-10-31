'''
Contains functions used to manipulate/handle text. This is quite a loose
collection of function and probably needs tidying up.

These include functions like
a fuzzy variable translator, remove substr after string, comment remove, setting
typos check etc...
'''

from src import type as typ
from src import EXCEPT as EXC
from src import consts

import difflib as dfl
import numpy as np
import sys
if sys.version_info[0] > 2:
    xrange = range

# Adds a 3D vector to the tcl_info dictionary (just adds 3 different entries)
def tcl_3D_input(data, dims, tcl_info, start_str):
    for i,dim in enumerate(dims):
        tcl_info["%s%s"%(start_str, dim)] = str(data[i])
    return tcl_info

# Will fix typos in a line of the settings file.
def setting_typo_check(line, defaults, setting_file_settings, replacer_settings):
    """
    Inputs:
        * line <str> => ...
        * setting_file_settings <?> => ...
        * replacer_settings <?> => ...
    """
    sett = line.split('=')[0].strip()
    poss_setts = fuzzy_variable_translate(sett, list(defaults),False, False,0.6)
    if sum(poss_setts) == 1:
        new_sett = defaults[poss_setts]
        if len(new_sett) == 1:
            new_sett = new_sett[0]
        if new_sett != sett:
            line = line.replace(sett, new_sett)
            setting_file_settings.append(sett)
            replacer_settings.append(new_sett)
    elif sum(poss_setts) > 1:
        EXC.WARN("There are too many possible settings for '%s'. These are:\n\t* %s.\n\nI do not want to assume which one it is, please correct it in the input file!"%(sett, '\n\t* '.join(defaults[poss_setts])), True)
    elif sum(poss_setts) < 1 and sett != 'path':
        EXC.WARN("There are too many possible settings for '%s'. These are:\n\t* %s.\n\nI do not want to assume which one it is, please correct it in the input file!"%(sett, '\n\t* '.join(defaults[poss_setts])), True)
    return line

def fuzzy_variable_translate(variable, poss_variables, verbose_output, throw_error=True, min_tol=0.3):
    """
    Uses a fuzzy finder to correct spelling mistakes in variables.

    Inputs:
        * variable <str> => The variable to be checked for typos.
        * poss_variables <list<str>> => The possible (correct) names of the variables.
        * verbose_output <bool> => Whether to output lots of info or not.
        * throw_error <bool>  (default True) => Whether to throw an error for no hits.
        * min_tol <float> (default 0.3) => The tolerance for classifying a 'hit'.

    Outputs:
        Returns a list of bools, these tell which element of the input list poss_variables are
         possible correct strings.
    """
    if type(variable) != str:
       return False
    a = [dfl.SequenceMatcher(None, variable.lower(), i.lower()).ratio() for i in poss_variables]
    if all(i < min_tol for i in a) and throw_error:
        EXC.ERROR("I don't know what variable '%s' means. \nValid Options are:%s"%(variable,'\n\t*'+'\n\t*'.join(poss_variables))  )
    if all(i < min_tol for i in a):
        return [False]*len(poss_variables)
    temp_array = [False]*len(poss_variables)
    temp_array = [True if i == max(a) else False for i in a ]
    if all(i < 0.95 for i in a) and verbose_output:
        EXC.WARN("Assuming '%s' means you want me to use %s" % (variable, np.array(poss_variables)[temp_array][0]))
    return temp_array

def fuzzy_variable_helper(variable, poss_var, just_1_var=True,
                          tol=0.3, verbose_out=True, throw_error=True):
    """
    A function to make the fuzzy_variable_translate slightly easier to use.

    Inputs:
        * variable <str> => The variable to be checked for typos.
        * poss_variables <list<str>> => The possible (correct) names of the variables.
        * just_1_var <bool> (default False) => Only allow 1 variable or not.
        * tol <float> (default 0.3) => The tolerance for classifying a 'hit'.
        * verbose_out <bool> => Whether to output lots of info or not.
        * throw_error <bool>  (default True) => Whether to throw an error for no hits.

    Outputs:
        Returns a list of strings that can be the variable. If
    """
    poss_var_inds = fuzzy_variable_translate(variable, poss_var, verbose_out, throw_error, tol)
    poss_vars = np.array(poss_var)
    poss_vars = poss_vars[poss_var_inds]

    if len(poss_vars) < 1: EXC.ERROR("I don't know what '%s' is supposed to mean." % variable)

    elif len(poss_vars) == 1:
        return poss_vars[0]

    elif len(poss_vars) > 1:
        if just_1_var:
            EXC.ERROR("I don't know what you mean by: '%s'.\n" % variable
                    + "You could mean:\n\t* %s" % '\n\t* '.join(poss_var))

        else: return poss_vars

# Take the product of a list of transformations in vmd such as: scaling; x,y,z rotations etc...
def combine_vmd_scalings(ltxt):
    prods = np.prod([typ.str_to_num(i.split('by')[1]) for  i in ltxt if 'scale' in i])
    return prods

# Sums a list of transformations in vmd such as: translations etc...
def combine_vmd_translations(ltxt):
    Tot_trans = np.array([0,0,0], dtype=np.float64)
    for line in ltxt:
        if 'translate' in line:
            trans = line.split('by')[1]
            trans = np.array([eval(i) for i in trans.split(' ') if i])
            Tot_trans += trans
    return Tot_trans

# Finds any variables in a string that starts with "$"
def find_tcl_variable(string, variables=[]):
    start_ind = string.find("$")
    end_ind = np.min([string[start_ind:].find(i)+1 for i in ['\n',' ']])
    if end_ind == 0:
      end_ind = len(string)
    variables.append(string[start_ind:end_ind])
    if start_ind != -1:
      find_tcl_variable(string[end_ind+1:],variables)
    variables = [i.replace('\n','').strip() for i in variables if i]
    return [i for i in list(set(variables)) if i and "$" in i]

# Removes any data already in the TCL script from the log file so it isn't repeated
def vmd_log_clean(log_txt, script_txt):
    script_ltxt = comment_remove(script_txt).split('\n')
    log_ltxt = comment_remove(log_txt).split('\n')
    # Find lines that look very much like the last line of the script
    poss_ends = [i for i in range(len(log_ltxt)) if dfl.SequenceMatcher(None, script_ltxt[-1], log_ltxt[i]).ratio() > 0.9]
    if len(poss_ends) == 1: #Only 1 line that could possibly be the end line
        start_of_new_log = poss_ends[0]+1
    #else:
    # remove lines that appear in the MainProcess.tcl script.
    log_ltxt = log_ltxt[start_of_new_log:]
    return log_ltxt

# Concatenates a string up to a specified substring
def remove_substr_after_str(txt, substr):
    ind = txt.find(substr)
    if ind < 0:
        ind = len(txt) + 1 + ind
    return txt[:ind]

# Removes a folder away from the filepath
def folderpath_back_N(folderpath, N=1):
   folderpath = str(folderpath)
   folderpath = folderpath[folderpath.find('/'):]
   for tmp in range(N):
      if folderpath[-1] == '/':
          folderpath = folderpath[:-1]
      folderpath = folderpath[:folderpath.rfind('/')] + '/'
   return folderpath

# Changes variables in the permanent settings file
def change_perm_settings(settings_txt, setting, value):
    """
    Changes variables in the permanent settings file, should ignore any triple
    string docstrings
    """
    if type(value) == str:
        value = "'%s'"%value.strip()
    ltxt = ltxt_clean(settings_txt.split('\n')) # clean up the file
    for i, line in enumerate(ltxt):
        if '=' in line:
            old_setting = line.split('=')[0].replace(' ','')
            if old_setting == setting:
                ltxt[i] = "%s = %s"%(setting, value)
                return '\n'.join(ltxt)
    else:
        raise SystemExit("""
Couldn't find the correct setting in the Templates/permanent settings.py file.

The file has probably been corrupted somehow. Please try deleting it and
restarting the code.
""")

# Removes any comments from some text
def comment_remove(string, cmt_str='#'):
    x = [i for i in string.split('\n') if i]
    x = [i for i in x if i[0] != cmt_str]
    x = [i[:i.find(cmt_str)] if i.find(cmt_str) != -1 else i for i in x ]
    string = '\n'.join(x)
    return string

# Will determine the number of atoms in an xyz file
def num_atoms_find(ltxt):
    start_atoms, finish_atoms = 0,0
    for i,line in enumerate(ltxt):
        if (typ.is_atom_line(line)) == True:
            start_atoms = i
            break
    for i,line in enumerate(ltxt[start_atoms:],start=start_atoms):
        if (typ.is_atom_line(line) == False):
            finish_atoms=i
            break
    return start_atoms, finish_atoms-start_atoms

# Will search for a piece of text in a string.
def text_search(txt, start_find, end_find="\n", error_on=True):
    start_ind = txt.find(start_find)
    if start_ind != -1:
        txt = txt[start_ind:]
        end_ind = txt.find(end_find)
        if end_ind != -1:
            txt = txt[:end_ind]
        else:
            txt = txt[0:20]
        return txt, start_ind, end_ind
    if error_on:
        EXC.WARN("No instance of '%s' in the txt!"%start_find)
    return False

# Will find keywords in the inp file
def inp_keyword_finder(inp_text, keyword):
    txt = text_search(inp_text, keyword, "\n")[0].split(' ')
    txt = [i for i in txt if typ.is_num(i)]
    return [float(i) for i in txt]

# Will align text left or right
def align(string, len_line=5, rl='r'):
    num_spaces = len_line-len(string)
    if rl == 'r':
        return "".join([" " for i in xrange(num_spaces)]) +  string
    elif rl == 'l':
        return string + "".join([" " for i in xrange(num_spaces)])

# Creates a string of the data in a cube file format
def cube_file_text(data, coords, at_nums, origin,
                   N_vec, basis_vec):
    """Return a string with the cube file text

    Args
        data: cube data in 3D array (Nx, Ny, Nz)
        coords: the atomic coords (Natom<x, y, z>)
        at_nums: atomic numbers (Natom)
        origin: the origin of the cube file
        N_vec: number of basis vecs in size (Nx, Ny, Nz)
        basis_vec: the basis vec of system
    """
    # HEADER FILE WRITING
    tab = "    "
    natom = len(at_nums)
    s = 'Cube file -generated by Orbital Movie Maker\n\n'
    s += f'{str(natom).ljust(4)}{tab}{tab.join((f"{i: .6f}" for i in  origin))}' + '\n'
    for i in range(3):
        str_bv = tab.join((f'{j: .6f}' for j in basis_vec[i]))
        s += f"{str(N_vec[i]).ljust(4)}{tab}{str_bv}" + '\n'

    for atom in range(natom):
        at_pos = tab.join(f'{i: .6f}' for i in coords[atom])
        s += f'{str(at_nums[atom]).ljust(4)}{tab} {0.0:.6f}{tab}{at_pos}' + '\n'

    # DATA SECTION WRITING.
    data = data.flatten()
    s += '\n'.join(tab.join(map(str, data[i:i+6]))
                   for i in range(0, len(data), 6))
    return s

# Puts all the instances of a triple string quotation on one line to be read in main.py
def triple_string_clean(ltxt):
    """
    Will concatenate all instances of a triple string in the list of lines to a
    single item.
    """
    triple_stings = ['"""',"'''"]
    str_mark = triple_stings[0]
    txt = '\n'.join(ltxt)

    # Create a unique identifier
    unique_replace_mark = "*54987&&*4637"
    if unique_replace_mark in txt:
        import random
        while (unique_replace_mark in txt):
            unique_replace_mark = ''.join([consts.alphabet[random.randint(0,25)] for i in range(15)])

    # Find all triple string sections and replace \n with unique identifier
    for str_mark in triple_stings:
        first_trip_str = txt.find(str_mark)
        if first_trip_str == -1: break
        next_trip_str  = txt[first_trip_str+len(str_mark):].find(str_mark)
        if next_trip_str == -1:
            raise SystemExit("Sorry I think there is no end to a string in the following text:\n\n%s"%txt)
        # Grab the bit of text between triple strings and replace carriage returns with a unique identifier
        partial_txt = txt[first_trip_str : next_trip_str+len(str_mark)*2]
        partial_txt = partial_txt.replace("\n", unique_replace_mark)
        txt = txt[:first_trip_str] + partial_txt + txt[next_trip_str+len(str_mark)*2:]
        # Make sure the carriage returns in the triple string are preserved
    ltxt = [i.replace(unique_replace_mark, '\n') for i in txt.split('\n')]
    return ltxt

    # trip_str_instances = False
    # txt = '\n'.join(ltxt)
    # for str_mark in triple_stings:
    #     if str_mark in txt:
    #         trip_str_instances = True
    #         break
    # if not trip_str_instances:
    #     return ltxt
    # count = 0
    # L = []
    # end_strs = []
    # begin_strs = []
    # for str_mark in triple_stings:
    #     for linei, line in enumerate(ltxt):
    #         if line.count(str_mark) > 2:
    #             break
    #         if line.count(str_mark) == 2:
    #             L.append(line)
    #             begin_strs.append(linei)
    #             end_strs.append(linei+1)
    #         if line.count(str_mark) == 1:
    #             line_split = line.split(str_mark)
    #             if count == 0:
    #                 if '=' in line_split[0]:
    #                     eq_split = [i.strip() for i  in line.split('=')]
    #                     if eq_split[0].count(' ') == 0:
    #                         count = 1
    #                         start_i = linei
    #                         begin_strs.append(linei)
    #             if count == 1:
    #                 for nlinei, nline in enumerate(ltxt[start_i+1:]):
    #                     if nline.count(str_mark):
    #                         L.append('\n'.join(ltxt[start_i:nlinei+start_i+2]))
    #                         end_strs.append(nlinei+start_i+2)
    #                         break
    #                 count = 0
    #     ltxt_tmp = ltxt[:begin_strs[0]]
    #     for i in range(len(begin_strs)-1):
    #         ltxt_tmp += ltxt[end_strs[i]:begin_strs[i+1]]
    #     ltxt_tmp += ltxt[end_strs[-1]:]
    #
    #     for line in ltxt_tmp:
    #         if str_mark not in ltxt_tmp and line not in L:
    #             L.append(line)
    # return L

# Returns the substring between 2 substrings within a string
def string_between(Str, substr1, substr2):
    Str = Str[Str.find(substr1)+len(substr1):]
    Str = Str[:Str.find(substr2)]
    return Str

# Cleans up the ltxt by putting all the lines that end with & onto the same line
#   Inputs:
#       ltxt = text in ltxt format (text split by '\n') [list]
#   Outpus:
#       the cleaned ltxt
def ltxt_clean(ltxt, end_line=',', joiner=','):
    ltxt = [i.strip() for i in ltxt]
    i, lim = 0, len(ltxt)
    while (i<lim):
        line = ltxt[i]
        if line != '':
            if line[-1] == end_line:
                ltxt[i] = ','.join([ltxt[i][:-1], ltxt[i+1]])
                del(ltxt[i+1])
                lim = len(ltxt)
                i -= 1
        i += 1
    ltxt = [j.strip() for j in ltxt if j.strip()]
    ltxt = triple_string_clean(ltxt)
    return ltxt


# #Combines rotations and gives back Euler angles
# def combine_rotations(ltxt, step_info):
#     I = np.array([[1,0,0],
#                   [0,1,0],
#                   [0,0,1]])
#     curr_rot = step_info['clean_settings_dict']['rotation']
#     print("I = ", I, "\n\n")
#     I = np.matmul(MT.create_X_rot_mat(curr_rot[0]),np.matmul(MT.create_Y_rot_mat(curr_rot[1]),MT.create_Z_rot_mat(curr_rot[2])))
#     print("I = ", I, "\n\n")
#     # Can't keep combining like this, will have to calculate Euler angles at each step.
#     for i in ltxt:
#         if "rotate" in i:
#             theta = eval(i.split('by')[1])
#             if 'rotate x' in i:
#                 print("I = ", I, "\n\n")
#                 X_rot_mat = MT.create_X_rot_mat(theta)
#                 print("The transformation = ",X_rot_mat, "\n\n")
#                 I = np.matmul(X_rot_mat,I)
#                 print("The final matrix = ",X_rot_mat, "\n\n")
#             if 'rotate y' in i:
#                 I = np.matmul(MT.create_Y_rot_mat(theta),I)
#             if 'rotate z' in i:
#                 I = np.matmul(MT.create_Z_rot_mat(theta),I)
#     final_rotation_euler = MT.rot_mat_to_euler_zyx(I)
#     return final_rotation_euler
