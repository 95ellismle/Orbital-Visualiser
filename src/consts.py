import numpy as np

pi = np.pi
ang2bohr = 1.88971616463207
bohr2ang = 0.52918
Orig_img_prefix = '$fs_img'
end_of_vmd_file = """
rotate x by 360.000000
rotate x by -360.000000
scale by 1.000000
scale by 1.000000
"""

img_folderpath      = './img/'
data_folderpath     = './data/'
template_folderpath = './Templates'
settings_filepath = './settings.inp'
