"""
 This is where all the default settings are stored.
 Each setting that can appear in the settings.inp file must be declared here (apart from the path setting).
 this means that all settings should have a default by design.

 Declaring a default setting should follow a few simple rules:
  *  The default setting names should all be lowercase
  *  The default setting should have a comment str after it explaining what it does (this is for documentation)
                  @ The comment str should contain: @
     * A description | A list of possible values | whether it has been tested



 N.B The tested declaration in the comment str isn't currently used so just put 'not-tested' for now!
"""




defaults = {
## Calibration
'calibrate'          : True , # Performs a single `calibration' step and save the image in the folder ./img/Calibration/ | ['yes','no'] | 'not-tested'
'timestep_to_render' : 0 , # Chooses which timestep to perform the calibration on. This is subject to the 'missing_pos_steps' setting. | [float, 'last'] | 'not-tested'
'show_img_after_vmd' : False, # Loads the calibration img after the code has ran | ['yes','no'] | 'not-tested'
'load_in_vmd'        : False, # Loads the calibration img in VMD. This is useful for setting the position of the image as the code will save any rotations, scalings and translations performed within VMD | ['yes', 'no'] | 'not-tested'
'isosurface_to_plot' : 0.007, # Which isosurface to plot during the simulation, a smaller number will show a larger isosurface | [float] | 'not-tested'
## Movie Creation
"title"             : None, # This will change the name of the movie and the folder it is saved in. Will save the movie in ./img/Title/ | [str] | 'not-tested'
'end_time'          : 'all', # Which <strong>time-step</strong> to iterate up to for the movie. This is inclusive e.g. an end_time of 20.0 would include the frame for 20.0fs. | ['all', float] | 'not-tested'
'stride'            : 1, # What stride to use to iterate with for the movie (a stride of 2 would render every 2nd frame). Refers to the step index in the file -not the MD step. | [int] | 'not-tested'
'start_time'        : 0.0, # Which <strong>time-step</strong> to start iterating on. This is inclusive e.g. a start_time of 0.0fs would include the frame for 0.0fs.| [float (zero is the first)] | 'not-tested'
'verbose_output'    : True, # Shows lots of info about what is happening | ['yes','no'] | 'not-tested'
'movie_length'      : 6, # The length of the finished movie (will adjust the framerate to match the time) | [int] | 'not-tested'
'movie_format'      : "mp4", # The format of the created movie | ['mp4'] | 'not-tested'
'missing_pos_steps' : "closest", # Will decide what to do with any position steps that are missing. Option details: <br><ul><li>'skip': Will skip any missing position steps.</li><li>'closest': Use the closest available position step in time (default).</li><li>'use N': Use position step N as the only step in the full visualisation. <emph>N.B. this is the step index rather than a timestep.</emph></li></ul> | ['skip', 'closest', 'use N'] | 'not-tested'
'restart_vis'       : False, # Will detect if you are using the same title as any in the img/ folder. If you are it will only visualise the steps not completed and stitch all imgs together into a movie at the end | ['yes', 'no'] | 'not-tested'
'img_size' : 'auto', # The size of the image rendered if 'auto' is chosen the image will be 1000x1000 for calibration and 650x650 for a movie | ['auto', [int, int]] | 'not-tested'
## File Handling
'aom_file'      : 'AOM_COEFF.include', # The AOM coefficients file required for the simulation | ['str'] | 'not-tested'
'lumo_file'     : 'LUMO_AOM_COEFF.include', # The LUMO AOM coefficients file for transition state plots | ['str'] | 'not-tested'
'homo_file'     : 'HOMO_AOM_COEFF.include', # The HOMO AOM coefficients file for transition state plots | ['str'] | 'not-tested'
'cp2k_inp_file' : 'run.inp', # The cp2k input file required for the simulation | ['str'] | 'not-tested'
'coeff_file'    : 'run-coeff-1.xyz', # The molecular (diabatic) coefficients file required for the simulation | ['str'] | 'not-tested'
'pvecs_file'    : 'run-pvecs-1.xyz', # The pvecs file required for the simulation | ['str'] | 'not-tested'
'pos_file'      : 'run-pos-1.xyz', # The positions (xyz) file required for the simulation | ['str'] | 'not-tested'
'decomp_file'   : 'DECOMP.include', # The DECOMP (active mol list) file used in the simulation | ['str'] | 'not-tested'
'keep_cube_files'   : False, # Keeps the data files after the simulation | ['yes','no'] | 'not-tested'
'keep_img_files'    : True, # Converts the outputted tga files to the default *img_format* and saves them | ['yes','no'] | 'not-tested'
'keep_tga_files'    : False, # Keeps the tga files after the simulation | ['yes','no'] | 'not-tested'
'img_format'        : "jpg", # The format to store the img files in | [str (e.g. "png" or "jpg") ] | 'not-tested'
## Wavefunction Shape
'type_of_wavefunction' : "phase", # The type of visualisation | ['density', 'phase'] | 'not-tested'
'atomic_number_orbital_map' : {1: (1, 's'), 6: (2, 'p'), 7: (2, 'p'), 16: (3, 'p')}, # The map of orbitals for each atom type | [] | 'not-tested'
## Transition State
'do_transition_state' : False, # Will plot the transition state density from 2 AOM coeff files | ['yes', 'no'] | 'not-tested'
## Positioning
'zoom_value'        : 1, # How much to scale the image by | [float, int] | 'not-tested'
'rotation'          : 'auto', # How much to rotate the image by: 'Auto' will let the code try and align the long axis along the x axis, you can also use manually set the angle of rotation (as Euler angles), this will rotate in the order z,y,x. You can also turn it off by setting this variable to 'no' | ['auto', list [x,y,z], 'no'] | 'not-tested'
'translate_by'      : (0,0,0), # How much to translate the image by | [list [x,y,z]] | 'not-tested'
## colors and Materials
'pos_real_iso_col'  : (0,0,1), # The color of positive real isosurfaces | [list [red, green, blue]] | 'not-tested'
'neg_real_iso_col'  : (1,0,0), # The color of negative real isosurfaces | [list [red, green, blue]] | 'not-tested'
'neg_imag_iso_col'  : (0,0.4,1), # The color of positive imaginary isosurfaces | [list [red, green, blue]] | 'not-tested'
'pos_imag_iso_col'  : (1,0.4,0), # The color of negative imaginary isosurfaces | [list [red, green, blue]] | 'not-tested'
'density_iso_col'   : (0.3, 0.32,0.3), # The color of the isosurface in a density visualisation | ['list [red,green,blue]'] | 'not-tested'
'carbon_color'      : 'silver', # The color of the Carbon atoms | [str] | 'not-tested'
'hydrogen_color'    : 'white', # The color of the Hydrogen atoms | [str] | 'not-tested'
'background_color'  : (1,1,1), # The color of the background | [str, list (rgb values e.g. [1,0,0])] | 'not-tested'
'neon_color'        : 'yellow', # The color of the Neon atoms | [str (VMD colors)] | 'not-tested'
'mol_style'         : 'CPK', # The visualisation style of the molecules (see VMD) | [str] | 'not-tested'
'mol_material'      : "Edgy", # The visualisation material used for the molecules (see VMD) | [str] | 'not-tested'
'iso_material'      : "BrushedMetal", # The visualisation material used for the isosurface (see VMD) | [str] | 'not-tested'
## Dimensions of system
'xdims'             : [-100000,100000], # Apply a cuttoff on the atoms in the image | [list [min coord, max coord]] | 'not-tested'
'ydims'             : [-100000,100000], # Apply a cuttoff on the atoms in the image | [list [min coord, max coord]] | 'not-tested'
'zdims'             : [-100000,100000], # Apply a cuttoff on the atoms in the image | [list [min coord, max coord]] | 'not-tested'
## Atoms to plot
'background_mols'   : False, # Plot the background, inactive, molecules in a light color | ['yes','no'] | 'not-tested'
'background_mols_end_extend' : 15, # How much to extend the background molecules past the last inactive atom | [int, float]  | 'not-tested'
'atoms_to_plot'     : 'auto', # Which atoms to plot in the image ('auto' will plot from 0 - max num of active molecules) | ['all', 'min_active', 'auto', 'list',int] | 'not-tested'
'atoms_to_ignore'   : [] , # Ignore any atoms by element, set to elemental symbol or name | [list [int], str] | 'not-tested'
'ignore_inactive_atoms' : False, # Will ignore all inactive atoms (including hydrogen on active molecules) | ['yes','no'] | 'not-tested'
'ignore_inactive_mols' : True, # Will ignore all inactive whole molecules | ['yes', 'no'] | 'not-tested'
'show_box'          : False, # Show the bounding box around the molecules | ['yes','no'] | 'not-tested'
## Optimisation
'min_abs_mol_coeff' : 5e-4, # A threshold for the minimum molecular population, will ignore any molecule with a population less than this. This probably shouldn't be changed if you don't fully understand how it works. The code will 'learn' the optimum value as it runs so a lower value here won't result in a loss of performance. | [float] | 'not-tested'
'nn_cutoff'         : 12, # The cutoff for constructing the nearest neighbour list. This decides how many neighbouring mols contribute to the wavefunction | ['integer [bohr?]'] | 'not-tested'
'resolution'        : 0.3, # Changes the resolution of the cube file data | [float] | 'not-tested'
'bounding_box_scale' : 12, # What to add to the dimension of the bounding box surrounding the atoms | [int, list [x,y,z]] | 'not-tested'
'at_box_size'       : 13, # How large a box to draw around each atom to calculate the orbital in | [int] | 'not-tested'
'dynamic_bounding_box' : True, # Will dynamically change the size of the bounding box depending on the size of the wavefunction | ['yes','no'] | 'not-tested'
## Text on the image
'draw_time'         : False, # <span style="color: red; text-transform: uppercase; font-weight: bold;"> Doesn't Work</span> Will draw a label with the time stamp on the image | ['yes', 'no'] | 'not-tested'
'pos_time_label'    : 'auto', # <span style="color: red; text-transform: uppercase; font-weight: bold;"> Doesn't Work</span> The position of the time-stamp | ['auto', list [x,y,z]] | 'not-tested'
'time_label_text'   : "Time = * fs", # <span style="color: red; text-transform: uppercase; font-weight: bold;"> Doesn't Work</span> The time-label text, the code will replace any * symbols with the current time-step | [str] | 'not-tested'
## Miscellaneous
'vmd_exe' : 'vmd', # Where to find the vmd binary. Should be a filepath pointing to vmd bin. | str | 'not-tested'
'vmd_timeout'       : 400, # How long to wait before assuming there is an error with the VMD script | [int, float] | 'not-tested'
'num_cores'         : 'half', # DEPRECATED | 'not-tested'
'time_step'         : False , # DEPRECATED | 'not-tested'
}





#'combination_rule' : 'L*H', # How to combine the LUMO and HOMO to create the transition state density. <br>This is a string the possible values are: <ul> <li>'L*H' -> LUMO*HOMO</li> <li>'L+H' -> LUMO+HOMO</li> <li>'H-L' -> HOMO - LUMO</li> <li>'L/H' -> LUMO - HOMO</li> | ['(L or H)*(L or H)'<br> '(L or H)+(L or H)'<br> '(L or H)-(L or H)'<br> '(L or H)/(L or H)'] | 'not-tested'
#'find_fuzzy_files'  : True, # Uses a fuzzy file finder to find files within a folder | [ 'yes', 'no' ] | 'not-tested'

