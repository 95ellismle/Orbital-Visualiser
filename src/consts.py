"""
Defines constants used throughout the code.

e.g. img_folderpath, data_folderpath, template_folderpath, pi, img_prefix etc...
"""
pi = 3.14159265358979323846
ang2bohr = 1.88971616463207
bohr2ang = 0.52918
Orig_img_prefix = '$fs_img'
end_of_vmd_file = """
rotate x by 360.000000
rotate x by -360.000000
scale by 1.000000
scale by 1.000000
"""
alphabet = 'abcdefghijklmnopqrstuvwxyz'

phaseMasks = {'PosReal': (-pi/4., pi/4.),
              'NegReal': ((3./4.*pi, pi), (-pi, -3./4.*pi)),
              'PosImag': (pi/4., 3./4.*pi),
              'NegImag': (-3./4.*pi, -pi/4.),
             }

img_folderpath      = './img/'
data_folderpath     = './data/'
template_folderpath = './Templates'
settings_filepath = './settings.inp'

