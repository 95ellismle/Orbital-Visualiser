"""
Contains functions involved in type checking or changing. E.g. converting
seconds to hours and minutes, converting atomic number to element, checking if a
str is a number etc...

N.B. periodic_table belongs in consts.py
"""

import numpy as np
from src import EXCEPT as EXC
import difflib as dfl

periodic_table = {'XX':0,'Ac':89,'Ag':47,'Al':13,'Am':95,'Ar':18,'As':33,'At':85,'Au':79,'B':5,'Ba':56,'Be':4,'Bh':107,'Bi':83,'Bk':97,'Br':35,'C':6,'Ca':20,'Cd':48,'Ce':58,'Cf':98,'Cl':17,'Cm':96,'Co':27,'Cr':24,'Cs':55,'Cu':29,'Db':105,'Ds':110,'Dy':66,'Er':68,'Es':99,'Eu':63,'F':9,'Fe':26,'Fm':100,'Fr':87,'Ga':31,'Gd':64,'Ge':32,'H':1,'He':2,'Hf':72,'Hg':80,'Ho':67,'Hs':108,'I':53,'In':49,'Ir':77,'K':19,'Kr':36,'La':57,'Li':3,'Lr':103,'Lu':71,'Md':101,'Mg':12,'Mn':25,'Mo':42,'Mt':109,'N':7,'Na':11,'Nb':41,'Nd':60,'Ne':10,'Ni':28,'No':102,'Np':93,'O':8,'Os':76,'P':15,'Pa':91,'Pb':82,'Pd':46,'Pm':61,'Po':84,'Pr':59,'Pt':78,'Pu':94,'Ra':88,'Rb':37,'Re':75,'Rf':104,'Rg':111,'Rh':45,'Rn':86,'Ru':44,'S':16,'Sb':51,'Sc':21,'Se':34,'Sg':106,'Si':14,'Sm':62,'Sn':50,'Sr':38,'Ta':73,'Tb':65,'Tc':43,'Te':52,'Th':90,'Ti':22,'Tl':81,'Tm':69,'U':92,'Uub':112,'Uuh':116,'Uuo':118,'Uup':115,'Uug':114,'Uus':117,'Uut':113,'V':23,'W':74,'Xe':54,'Y':39,'Yb':70,'Zn':30,'Zr':40,'Actinium':89,'Silver':47,'Aluminium':13,'Americium':95,'Argon':18,'Arsenic':33,'Astatine':85,'Gold':79,'Boron':5,'Barium':56,'Beryllium':4,'Bohrium':107,'Bismuth':83,'Berkelium':97,'Bromine':35,'Carbon':6,'Calcium':20,'Cadmium':48,'Cerium':58,'Californium':98,'Chlorine':17,'Curium':96,'Cobalt':27,'Chromium':24,'Caesium':55,'Copper':29,'Dubnium':105,'Darmstadtium':110,'Dysprosium':66,'Erbium':68,'Einsteinium':99,'Europium':63,'Fluorine':9,'Iron':26,'Fermium':100,'Francium':87,'Gallium':31,'Gadolinium':64,'Germanium':32,'Hydrogen':1,'Helium':2,'Hafnium':72,'Mercury':80,'Holmium':67,'Hassium':108,'Iodine':53,'Indium':49,'Iridium':77,'Potassium':19,'Krypton':36,'Lanthanum':57,'Lithium':3,'Lawrencium':103,'Lutetium':71,'Mendelevium':101,'Magnesium':12,'Manganese':25,'Molybdenum':42,'Meitnerium':109,'Nitrogen':7,'Sodium':11,'Niobium':41,'Neodymium':60,'Neon':10,'Nickel':28,'Nobelium':102,'Neptunium':93,'Oxygen':8,'Osmium':76,'Phosphorus':15,'Protactinium':91,'Lead':82,'Palladium':46,'Promethium':61,'Polonium':84,'Praseodymium':59,'Platinum':78,'Plutonium':94,'Radium':88,'Rubidium':37,'Rhenium':75,'Rutherfordium':104,'Roentgenium':111,'Rhodium':45,'Radon':86,'Ruthenium':44,'Sulphur':16,'Antimony':51,'Scandium':21,'Selenium':34,'Seaborgium':106,'Silicon':14,'Samarium':62,'Tin':50,'Strontium':38,'Tantalum':73,'Terbium':65,'Technetium':43,'Tellurium':52,'Thorium':90,'Titanium':22,'Thallium':81,'Thulium':69,'Uranium':92,'Ununbium':112,'Ununhexium':116,'Ununoctium':118,'Ununpentium':115,'Ununquadium':114,'Ununseptium':117,'Ununtrium':113,'Vanadium':23,'Tungsten':74,'Xenon':54,'Yttrium':39,'Ytterbium':70,'Zinc':30,'Zirconium':40}
periodic_table = {i.upper():periodic_table[i] for i in periodic_table}
reverse_periodic_table = {periodic_table[i]:i for i in periodic_table if len(i) < 3}

def eval_type(String):
    """
    Will convert a string to a number if possible. If it isn't return a string.

    Inputs:
        * String  =>  Any string

    Outpus:
        * If the string can be converted to a number it will be with ints being
          preferable to floats. Else will return the same string
    """
    String = str(String)

    if is_float(String):
        return float(String)
    elif is_num(String):
        if '.' in String:
            return float(String)
        return int(String)
    elif String == "True":
        return True
    elif String == "False":
        return False
    else:
        return String


def is_float(Str):
    """
    Check whether a string can be represented as a non-integer float

    Inputs:
      * Str <str> => A string to check

    Outputs:
      <bool> Whether the string can be represented as a non-integer float or not.
    """
    if type(Str) == str:
        if is_num(Str):
            if not float(Str).is_integer():
                return True
    return False


# Converts letters to atomic numbers
def atomic_num_convert(name, warn=True):
    try:
        an = periodic_table[name.upper().strip()]
    except KeyError:
        an = False
    return an

#Seconds to minutes and hours
def seconds_to_minutes_hours(seconds, time_elapsed_str=False):
   minutes, seconds = divmod(seconds, 60)
   hours, minutes   = divmod(minutes, 60)
   if type(time_elapsed_str) == str:
     if hours > 0:
        time_elapsed_str += "%i hrs"%int(hours)
     if minutes > 0:
       time_elapsed_str += "%i mins"%int(minutes)
     time_elapsed_str += "%.2g s"%seconds
     return time_elapsed_str
   return [i for i in (hours, minutes, seconds) ]

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

# Checks whether a float can be an integer
def is_int(Float):
    if is_between(int(Float),Float):
        return int(Float)
    else:
        return False

# Checks whether an object is a list or array etc...
def is_list_array(array):
    return type(array) == type([1,2]) or type(array) == type(np.array([1,2]))

# The resolution needs to divide neatly into the cell size otherwise the wavefunction ends up being translated into another place as there can only be an integer number of voxels in each direction.
def int_res_marry(nums, res, cell_sze):
    nums = [np.ceil(i) for i in nums]
    new_nums = []
    for i, num in enumerate(nums):
        while (type(is_int(num / res)) != int
                and num <= (cell_sze[i]) / res):
            num += 1
        new_nums.append(int(num/res))
    return new_nums

# Will translate a user input to a boolean
def translate_to_bool(val, name):
    if type(val) == str:
        if 'y' in val.lower():
            return True
        if 'n' in val.lower() or not val:
            return False
        else:
            print ("Sorry I don't understand the variable %s\nValid options are, 'yes', 'no', True & False."%name)
            return 'BREAK THE CODE AS %s HASN\'T BEEN SET PROPERLY!'%name
    else:
        try:
            val = bool(val)
        except:
            pass
        if val == True:
            return True
        if val == False:
            return False
        else:
            print( "Sorry I don't understand the variable %s\nValid options are, 'yes', 'no', True & False."%name)
            return 'BREAK THE CODE AS %s HASN\'T BEEN SET PROPERLY!'%name

# Checks whether a float is near another float
def is_between(num, comparison, *args):
    lower = comparison - 1e-12
    upper = comparison + 1e-12
    try:
        lower = comparison - args[0]
        try:
            upper = comparison + args[1]
        except IndexError:
            pass
    except IndexError:
        pass
    if num > lower and num < upper:
        return True
    return False

# Checks whether a number is normalised or not
def is_norm(Cnum):
    if is_between(np.linalg.norm(Cnum), 1, 0.01, 0.01) or np.linalg.norm(Cnum) == 0:
        return True
    return False

# Converts a string into a number (either int or float depending on the number)
def str_to_num(string, ret_str=False):
    try:
        fl= float(string)
        if int(fl) == fl:
            return int(fl)
        return fl
    except:
        if ret_str:
            return string
        return False
