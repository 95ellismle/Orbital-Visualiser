"""
 This is where all the default settings are stored.
 Each setting that can appear in the settings.inp file must be declared here (apart from the path setting).
 this means that all settings should have a default by design.

 Declaring a default setting should follow a few simple rules:
  *  The default setting names should all be lowercase
  *  The default setting should have a comment string after it explaining what it does (this is for documentation)
                  @ The comment string should contain: @
     * A description | A list of possible values | whether it has been tested



 N.B The tested declaration in the comment string isn't currently used so just put 'not-tested' for now!
"""




defaults = {
## Calibration
'calibrate'         : True , # Performs a single `calibration' step and save the image in the folder ./img/Calibration/ | ['yes','no'] | 'not-tested'
'calibration_step'  : 10 , # Chooses which (nuclear) timestep to perform the calibration on. Floats (less than 1) will be converted to a percentage of total time (e.g. 0.1 will give 10% time) | ['integer', 'float', '"half"', '"last"'] | 'not-tested'
'show_img_after_vmd' : False, # Loads the calibration img after the code has ran | ['yes','no'] | 'not-tested'
'load_in_vmd'       : False, # Loads the calibration img in VMD. This is useful for setting the position of the image as the code will save any rotations, scalings and translations performed within VMD | ['yes', 'no'] | 'not-tested'
'isosurface_to_plot' : 0.003, # Which isosurface to plot during the simulation, a smaller number will show a larger isosurface | ['float'] | 'not-tested'
## Movie Creation
"title"             : None, # This will change the name of the movie and the folder it is saved in. Will save the movie in ./img/Title/ | ['string'] | 'not-tested'
'end_step'         : 'all', # Which step to iterate up to for the movie | ['all', 'integer'] | 'not-tested'
'stride'            : 1, # What stride to use to iterate with for the movie (a stride of 2 would render every 2nd frame) | ['integer'] | 'not-tested'
'start_step'        : 0, # Which step to start iterating on | ['integer (zero is the first)'] | 'not-tested'
'verbose_output'    : False, # Shows lots of info about what is happening | ['yes','no'] | 'not-tested'
'movie_length' : 6, # The length of the finished movie (will adjust the framerate to match the time) | ['integter'] | 'not-tested'
'movie_format'      : "mp4", # The format of the created movie | ['mp4'] | 'not-tested'
'restart_vis'       : False, # Will detect if you are using the same title as any in the img/ folder. If you are it will only visualise the steps not completed and stitch all imgs together into a movie at the end | ['yes', 'no'] | 'not-tested'
## File Handling
'CP2K_output_files' : {'AOM':'AOM_COEFF.include', 'inp':'run.inp', 'mol_coeff':'run-coeff-1.xyz', 'pvecs':'run-pvecs-1.xyz', 'xyz':'run-pos-1.xyz'} , # The output files required for the simulation | ['dictionary (see default value)'] | 'not-tested'
'keep_cube_files'   : False, # Keeps the data files after the simulation | ['yes','no'] | 'not-tested'
'keep_img_files'    : True, # Converts the outputted tga files to the default *img_format* and saves them | ['yes','no'] | 'not-tested'
'keep_tga_files'    : False, # Keeps the tga files after the simulation | ['yes','no'] | 'not-tested'
'img_format'        : "jpg", # The format to store the img files in | ['str (e.g. "png" or "jpg" '] | 'not-tested'
'find_fuzzy_files'  : True, # Uses a fuzzy file finder to find files within a folder | [ 'yes', 'no' ] | 'not-tested'
## Replicas
'num_reps'          : 1, # The number of replicas in the system | ['integer'] | 'not-tested'
'rep_comb_type'     : 'mean', # What type of replica set-up. 'mean' will mean the data from each replica. | ['mean'] | 'not-tested'

## Positioning
'zoom_value'        : 1, # How much to scale the image by | ['float', 'integer'] | 'not-tested'
'rotation'          : 'auto', # How much to rotate the image by: 'Auto' will let the code try and align the long axis along the x axis, you can also use manually set the angle of rotation (as Euler angles), this will rotate in the order z,y,x. You can also turn it off by setting this variable to 'no' | ['auto', 'list [x,y,z]', 'no'] | 'not-tested'
'translate_by'      : (0,0,0), # How much to translate the image by | ['list [x,y,z]'] | 'not-tested'
## colors and Materials
'pos_real_iso_col'  : (1,0,0), # The color of positive real isosurfaces | ['list [red, green, blue]'] | 'not-tested'
'neg_real_iso_col'  : (0,0,1), # The color of negative real isosurfaces | ['list [red, green, blue]'] | 'not-tested'
'neg_imag_iso_col'  : (0,0.4,1), # The color of positive imaginary isosurfaces | ['list [red, green, blue]'] | 'not-tested'
'pos_imag_iso_col'  : (1,0.4,0), # The color of negative imaginary isosurfaces | ['list [red, green, blue]'] | 'not-tested'
'density_iso_col'   : (0.3, 0.32,0.3), # The color of the isosurface in a density visualisation | ['list [red,green,blue]'] | 'not-tested'
'carbon_color'     : 'black', # The color of the Carbon atoms | ['string'] | 'not-tested'
'hydrogen_color'   : 'white', # The color of the Hydrogen atoms | ['string'] | 'not-tested'
'background_color' : (1,1,1), # The color of the background | ['string', 'list (rgb values e.g. [1,0,0])'] | 'not-tested'
'neon_color'        : 'yellow', # The color of the Neon atoms | ['string (VMD colors)'] | 'not-tested'
'mol_style'         : 'CPK', # The visualisation style of the molecules (see VMD) | ['string'] | 'not-tested'
'mol_material'      : "Edgy", # The visualisation material used for the molecules (see VMD) | ['string'] | 'not-tested'
'iso_material'      : "BrushedMetal", # The visualisation material used for the isosurface (see VMD) | ['string'] | 'not-tested'
## Dimensions of system
'xdims'             : [-100000,100000], # Apply a cuttoff on the atoms in the image | ['list [min coord, max coord]'] | 'not-tested'
'ydims'             : [-100000,100000], # Apply a cuttoff on the atoms in the image | ['list [min coord, max coord]'] | 'not-tested'
'zdims'             : [-100000,100000], # Apply a cuttoff on the atoms in the image | ['list [min coord, max coord]'] | 'not-tested'
## Atoms to plot
'background_mols'   : False, # Plot the background, inactive, molecules in a light color | ['yes','no'] | 'not-tested'
'background_mols_end_extend' : 15, # How much to extend the background molecules past the last inactive atom | ['integer', 'float']  | 'not-tested'
'atoms_to_plot'     : 'auto', # Which atoms to plot in the image ('auto' will plot from 0 - max num of active molecules) | ['all', 'min_active', 'auto', 'list','integer'] | 'not-tested'
'atoms_to_ignore'   : [] , # Ignore any atoms by element, set to elemental symbol or name | ['list', 'str'] | 'not-tested'
'ignore_inactive_atoms' : False, # Will ignore all inactive atoms (including hydrogen on active molecules) | ['yes','no'] | 'not-tested'
'ignore_inactive_mols' : True, # Will ignore all inactive whole molecules | ['yes', 'no'] | 'not-tested'
'show_box'          : False, # Show the bounding box around the molecules | ['yes','no'] | 'not-tested'
## Optimisation
'min_abs_mol_coeff' : 5e-4, # A threshold for the minimum molecular population, will ignore any molecule with a population less than this. This probably shouldn't be changed if you don't fully understand how it works. The code will 'learn' the optimum value as it runs so a lower value here won't result in a loss of performance. | ['float'] | 'not-tested'
'nn_cutoff'         : 12, # The cutoff for constructing the nearest neighbour list. This decides how many neighbouring mols contribute to the wavefunction | ['integer [bohr?]'] | 'not-tested'
'resolution'        : 0.4, # Changes the resolution of the cube file data | ['float'] | 'not-tested'
'bounding_box_scale' : 7, # What to add to the dimension of the bounding box surrounding the atoms | ['integer', 'list [x,y,z]'] | 'not-tested'
'dynamic_bounding_box' : True, # Will dynamically change the size of the bounding box depending on the size of the wavefunction | ['yes','no'] | 'not-tested'
## Text on the image
'draw_time'         : False, # Will draw a label with the time stamp on the image | ['yes', 'no'] | 'not-tested'
'pos_time_label'    : 'auto', # The position of the time-stamp | ['auto', 'list [x,y,z]'] | 'not-tested'
'time_label_text'   : "Time = * fs", # The time-label text, the code will replace any * symbols with the current time-step | ['string'] | 'not-tested'
## Graph
'side_by_side_graph' : False, # Plots a graph of the molecular coefficients beside the visualisation | ['yes','no'] | 'not-tested'
'mols_to_highlight' : 0, # Which molecules to highlight in the graph (shows as a opaque line instead a transparent one) | ['max', 'min', 'all', 'integer (mol index)'] | 'not-tested'
'ylabel'            : r'|u$_i$|$^2$', # What to put on the y axis (graph) | ['string'] | 'not-tested'
'xlabel'            : 'time [fs]', # What to put on the x axis (graph) | ['string'] | 'not-tested'
'yfontsize'         : 18, # Size of the y axis font (graph) | ['string'] | 'not-tested'
'xfontsize'         : 17, # Size of the x axis font (graph) | ['string'] | 'not-tested'
'graph_title'       : 'Probability Density, Molecule * being highlighted', # The title for the graph | ['string'] | 'not-tested'
'title_fontsize'    : 20, # Size of the title font (graph) | ['string'] | 'not-tested'
'graph_to_vis_ratio' : 1, # The ratio of sizes (x dimension) for the graph and the visualisation (e.g. 2 would mean the graph would be 2 times as big as the visualisation) | ['integer'] | 'not-tested'
'max_data_in_graph' : 450, # Maximum amount of data points in the graph before refreshing them | ['integer'] | 'not-tested'
## Miscellaneous
'vmd_timeout'       : 400, # How long to wait before assuming there is an error with the VMD script | ['integer', 'float'] | 'not-tested'
'type_of_wavefunction' : "phase", # The type of visualisation | ['density', 'phase'] | 'not-tested'
'num_cores'         : 'half', # DEPRECATED | 'not-tested'
'time_step'         : False , # DEPRECATED | 'not-tested'
}
