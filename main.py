"""Will import the python3 print function."""
from __future__ import print_function

import numpy as np
import time
import os
import subprocess
import sys

from src import EXCEPT as EXC
from src import geometry as geom
from src import type as typ
from src import math as MT
from src import text as txt_lib
from src import IO as io
from src import consts

from init import INIT
all_settings = INIT.all_settings


"""
 This is the file that runs everything. The MainLoop class is responsible for
 actually carrying out the visualisation once everything has been initialised.

 The initialisation is carried out by files in the init folder. By importing
 the init module we run these methods and initialise (read all the files and
 parse them into a computer friendly format).

 After this we create an instance of the MainLoop class. This class handles all
 the visualisation stuff, from creating the data, to visualising in vmd and
 stitching together images into a movie.

 To get a feel for what the MainLoop is doing a good place to start is the
 do_step method which is the function which makes an image from the data.
"""

if sys.version_info[0] > 2:
    xrange = range
    raw_input = input


class MainLoop(object):

    """ This will carry out all the main functions and actually run the code.
         It will create the data, save it as a cube file, and make vmd render
         it      """

    def __init__(self, all_settings, all_steps, errors):
        self.tcl_dict_ind = 0
        self.errors = errors
        self.all_settings = all_settings
        self.neg_iso_cols = {}
        self.pos_iso_cols = {}
        self.PID = "MainProcess"
        for step in all_steps:  # Loop over all steps and visualise them
            self.step = step
            # Find the phase of the first mol as a reference.
            # Phase = angle in complex plane
            self.theta1 = np.angle(self.all_settings['mol'][self.step][0])
            tmp = consts.Orig_img_prefix.replace(
                    "$fs",
                    "%.2f" % self.all_settings['Ntime-steps'][self.step])
            self.all_settings['img_prefix'] = tmp.replace(".", ",")
            start_time = time.time()
            self.do_step(step)  # Do a visualisation step
            # Pretty print timings
            self.__print_timings(step, len(all_steps), start_time)
        self._finalise(len(all_steps))

    # Completes 1 step
    def do_step(self, step):
        """
        Will carry out a single visualisation step. This involves creating the
        wavefunction data, writing this to a file and rendering it with VMD.

        Inputs:
            * step  =>  Which step to visualise.
        """
        self._find_active_molecules()
        self.data_files_to_visualise = []
        self._vmd_filename_handling()
        if self.all_settings['background_mols']:
            self._write_background_mols()
        self.theta1 = np.angle(self.all_settings['mol'][self.step][0])

        for mol_i, mol_id in enumerate(self.active_step_mols):
            self._find_active_atoms(mol_id)
            self._create_wf_data(mol_id, step)
            self._set_wf_colors()
            self._save_wf_colors()
            self._create_cube_file_txt(step)
            self._write_cube_file(step, mol_id)
        self._vmd_visualise(step)  # run the vmd script and visualise the data
        if self.all_settings['side_by_side_graph']:  # (Not supported)
            self._plot(step)  # Will plot a graph to one side (Not supported)

    # Finds a dynamic bounding box scale. Makes the bounding box smaller
    #  when the mol_coeff is smaller
    # Can calculate the mol_C_abs here instead of in the create_wf_data
    #  function
    def _dynamic_bounding_box_scale(self, mol_ind, BBS):
        """
        Calculates the new bounding box scale according to how much population
        there is on the molecule. Uses a tanh function to vary scale. This is
        an optimisation (makes bounding box smaller for smaller densities).

        Inputs:
            BBS      =>  Original bounding box scale
            mol_ind  =>  The index of the molecule
        """
        w = 0.4  # When do we first start getting there
        c = 4  # minimum bounding box scale
        pop = np.abs(self.all_settings['pops'][self.step][mol_ind])
        new_bb_scale = np.tanh(pop/w)
        new_bb_scale *= (BBS - c)
        new_bb_scale += c
        new_bb_scale = np.ceil(new_bb_scale)
        return int(new_bb_scale)

    # Will handle where to save the various vmd files created
    def _vmd_filename_handling(self):
        """
        Will handle the setting of the filepaths involved in the vmd part of
        the process.
        """
        tmp = self.all_settings['vmd_script_folder'] + self.PID + ".tcl"
        self.all_settings['vmd_script'][self.PID] = tmp
        tmp = self.all_settings['vmd_script_folder'] + self.PID + '.out'
        self.all_settings['vmd_junk'][self.PID] = tmp
        tmp = self.all_settings['vmd_script_folder'] + self.PID + '.error'
        self.all_settings['vmd_err'][self.PID] = tmp
        self.all_settings['delete_these'].append(
                                        self.all_settings['vmd_junk'][self.PID]
                                                )
        self.all_settings['delete_these'].append(
                                         self.all_settings['vmd_err'][self.PID]
                                                )

    # Will save the background molecules in an xyz file to be loaded by vmd
    def _write_background_mols(self):
        """
        Will write the background molecules as a seperate xyz file. The
        background molecules are the inactive FIST molecules as part of the
        wider crystal.
        """
        # Dealing with the background molecules
        largest_dim = np.argmax(
                          [np.max(self.all_settings['coords'][self.step][:, i])
                           for i in range(3)]
                               )
        # dims = [Xdims, Ydims, Zdims][largest_dim]
        # Find maximum coordinate
        at_crds = self.all_settings['coords'][self.step]
        at_crds = at_crds[self.all_settings['atoms_to_plot']][:, largest_dim]
        max_coord = np.max(at_crds)
        max_coord += self.all_settings['background_mols_end_extend']

        # Find coordinates within below this max_coord
        mask = self.all_settings['coords']
        mask = mask[self.step][:, largest_dim] < max_coord

        # Apply the mask to get data. Find metadata.
        background_mols_pos = self.all_settings['coords'][self.step][mask]
        background_mols_at_num = self.all_settings['at_num'][mask]
        backgrnd_mols_filepath = self.all_settings['data_fold'] + \
            "bckgrnd_mols-%s.xyz" % self.PID

        # Write the background mols
        io.xyz_step_writer(background_mols_pos,
                           background_mols_at_num,
                           self.all_settings['Mtime-steps'][self.step],
                           self.step,
                           backgrnd_mols_filepath,
                           consts.bohr2ang)

        tcl_load_xyz_cmd = 'mol new {%s}' % backgrnd_mols_filepath
        tcl_load_xyz_cmd += ' type {xyz} first 0 last -1 step 1 waitfor 1'
        self.all_settings['tcl']['backgrnd_mols'] = tcl_load_xyz_cmd

    # Finds how many molecules have a significant charge to visualise
    # Probably isn't actually that useful!
    # The min tolerance thing is actually more useful.
    def _localisation(self):
        """
        Finds the number of molecules that the charge is localised over.

        I was going to use this to set a cutoff for the number of molecules to
        be visualised as an optimisation. But it didn't go anywhere.
        """
        localisation = MT.IPR(self.all_settings['mol'][self.step])
        if localisation > 1.00001:
            localisation *= 1+np.exp(-localisation)
        localisation = int(np.ceil(localisation))
        print("Localisation = ", localisation, len(self.active_step_mols))

    # Will find the active molecules which need looping over
    def _find_active_molecules(self):
        """
        Will find which molecules are active and which can be ignored.
        """
        # Find mols with enough population (more than min_abs_mol_coeff)
        mask = (self.all_settings['pops'][self.step] >
                self.all_settings['min_abs_mol_coeff'])
        self.active_step_mols = np.arange(0, self.all_settings['nmol'])[mask]
        self.all_settings['mols_plotted'] = len(self.active_step_mols)
        if self.all_settings['mols_plotted'] == 0:
            self.active_step_mols = np.array([0])
            if self.all_settings['verbose_output']:
                msg = "No molecules found that have a high enough molecular "
                msg += "coefficient to be plotted for"
                msg += " trajectory %i" % self.step
                EXC.WARN(msg)
        return self.active_step_mols

    # Will find the active atoms to loop over
    def _find_active_atoms(self, mol_id):
        """
        Find which atoms are active (on a molecule?) according to the
        AOM_COEFF.include file.

        Inputs:
            * mol_id  =>  The molecule to find active atoms for
        """
        # Find active coordinates (from active atom index)
        atMask = [i for i in self.all_settings['active_atoms_index'][mol_id]]
        self.active_coords = self.all_settings['coords'][self.step][atMask]
        self.active_coords = np.array(self.active_coords)

        if len(self.active_coords) <= 0:
            # Check if any molecules are past the number of molecules being
            #  visualised
            max_plot_mol = self.all_settings['num_mols_active']
            max_act_mol = max(self.active_step_mols)
            if max_act_mol > max_plot_mol:
                msg = "The charge is no longer contained by the molecules"
                msg += " shown.\nPlease extend the range to allow for this!"
                msg += "\n\nMax charged molecule = %i" % max_plot_mol
                msg += "\tMax molecule plotted = %i" % max_act_mol
                EXC.WARN(msg, True)
            else:
                msg = "Something went wrong and I don't know what sorry!"
                msg += "\nThe length of the active_coords array "
                msg += "is %i. It should be >0" % len(self.active_coords)
                SystemExit(msg)
                return False

    # Will create the wavefunction data
    def _create_wf_data(self, mol_id, step):
        """
        Will create the wavefunction data. This involves creating a bounding
        box big enough to encapsulate the wf and creating the p-orbital data.

        Inputs:
            * mol_id  =>  The molecule index
            * step    =>  The step number
        """
        start_data_create_time = time.time()

        # Drawing a bounding box around the active atoms to prevent creating
        #  unecessary data
        if self.all_settings['dyn_bound_box']:
            BBS_dyn = [self._dynamic_bounding_box_scale(mol_id, i)
                       for i in self.all_settings['bounding_box_scale']]
        else:
            BBS_dyn = self.all_settings['bounding_box']
        act_crds = [self.active_coords[:, k] for k in range(3)]
        trans, active_size = geom.min_bounding_box(act_crds, BBS_dyn)
        self.sizes = typ.int_res_marry(active_size,  # How many grid points
                                       self.all_settings['resolution'],
                                       [1, 1, 1])

        # Create wf data
        scale_factors = [size*self.all_settings['resolution']
                         for size in self.sizes]
        scale_factors = np.array(scale_factors)

        # Actually create the data
        self.data = np.zeros(self.sizes, dtype=complex)
        self.origin = scale_factors/-2 + trans
        self.mol_C = self.all_settings['mol'][self.step][mol_id]
        mol_C_abs = np.absolute(self.mol_C)**2

        for j in self.all_settings['AOM_D']:  # loop over atoms
 
            molIsActive = self.all_settings['mol_info'][j] == mol_id
            if molIsActive:  # if active molecule

                at_crds = self.all_settings['coords'][self.step][j] - trans
                self.atom_I = self.all_settings['AOM_D'][j][1]
                pvecs = self.all_settings['pvecs'][self.step][self.atom_I]
                AOM = self.all_settings['AOM_D'][j][0]

                self.data += MT.dot_3D(MT.SH_p(self.sizes[0],
                                               self.sizes[1],
                                               self.sizes[2],
                                               self.all_settings['resolution'],
                                               at_crds),
                                       pvecs) * AOM
        # Density plot shows |psi|^2
        if self.all_settings['color_type'] == 'density':
            self.data *= self.mol_C
            self.data *= np.conjugate(self.data)
        else:
            # Phase plot shows psi = |u|^2 * SOMO
            self.data *= mol_C_abs
        end_time = time.time() - start_data_create_time
        self.all_settings['times']['Create Wavefunction'][step] += end_time

    # Creates the cube file to save
    def _create_cube_file_txt(self, step):
        """
        Creates the cube file as a string. This is created as a string first
        then written to a file as this is much more efficient than writing each
        line to a file on the fly
        """
        start_cube_create_time = time.time()
        # Probably not too bad creating this tiny list here at every step.
        xyz_basis_vectors = [[self.all_settings['resolution'], 0, 0],
                             [0, self.all_settings['resolution'], 0],
                             [0, 0, self.all_settings['resolution']]]
        xyz_basis_vectors = np.array(xyz_basis_vectors)

        self.cube_txt = txt_lib.cube_file_text(
                              self.data.real,
                              vdim=self.sizes,
                              mol_info=self.all_settings['mol_info'],
                              orig=self.origin,
                              Ac=self.all_settings['coords'][self.step],
                              An=self.all_settings['at_num'],
                              tit=self.all_settings['title'],
                              atoms_to_plot=self.all_settings['atoms_to_plot'],
                              basis_vec=xyz_basis_vectors
                                              )

        end_time = time.time() - start_cube_create_time
        self.all_settings['times']['Create Cube Data'][step] += end_time

    # Handles the saving the wf colors in a dictionary of the wavefunction.
    def _set_wf_colors(self):
        """
        Will determine the colour of the wavefunction depending on the setting
        chosen. If density is chosen then the wavefunction will all be one
        colour. If real-phase is chosen the wavefunction will be one colour for
        positive values and another for negative values. If full phase is
        chosen the colour will be dependent on which quadrant in the complex
        plane the coefficient appears in.
        """
        thetai = np.angle(self.mol_C*self.atom_I) - self.theta1
        # Could optimise (and tidy) this, the code doesn't need to do all
        #  this at every step
        if self.all_settings['color_type'] == 'density':
            self.neg_iso_cols[self.tcl_dict_ind] = 22
            self.pos_iso_cols[self.tcl_dict_ind] = 22

        elif self.all_settings['color_type'] == 'real-phase':
            self.neg_iso_cols[self.tcl_dict_ind] = 21
            self.pos_iso_cols[self.tcl_dict_ind] = 20

        elif self.all_settings['color_type'] == 'phase':
            if -np.pi/4 < thetai <= np.pi/4:  # Pos Real Quadrant
                self.neg_iso_cols[self.tcl_dict_ind] = 21
                self.pos_iso_cols[self.tcl_dict_ind] = 20
            elif np.pi/4 < thetai <= 3*np.pi/4:  # Pos Imag Quadrant
                self.neg_iso_cols[self.tcl_dict_ind] = 19
                self.pos_iso_cols[self.tcl_dict_ind] = 18
            elif 3*np.pi/4 < thetai <= 5*np.pi/4:  # Neg Real Quadrant
                self.neg_iso_cols[self.tcl_dict_ind] = 20
                self.pos_iso_cols[self.tcl_dict_ind] = 21
            else:  # Neg imag Quadrant
                self.neg_iso_cols[self.tcl_dict_ind] = 18
                self.pos_iso_cols[self.tcl_dict_ind] = 19
        self.tcl_dict_ind += 1

    # Saves the wavefunction coloring in the tcl dictionary
    def _save_wf_colors(self):
        """
        Saves the wf colours in the tcl dictionary to be visualised by vmd
        """
        replacers = [(',', ''),
                     ('[', '{'),
                     (']', '}'),
                     (':', ''),
                     ("'", "")]

        # tmp -> store temporary string for TCL command
        tmp = str(self.neg_iso_cols)
        for find, replace in replacers:
            tmp = tmp.replace(find, replace)
        neg_col_dict_str = "set Negcols " + tmp

        tmp = str(self.pos_iso_cols)
        for find, replace in replacers:
            tmp = tmp.replace(find, replace)
        pos_col_dict_str = "set Poscols " + tmp

        self.all_settings['tcl']['neg_cols'] = neg_col_dict_str
        self.all_settings['tcl']['pos_cols'] = pos_col_dict_str

    # Visualises the vmd data and adds timings to the dictionary
    def _vmd_visualise(self, step):
        """
        Visualises the data. This fills in the variables in the vmd template,
        writes the script and runs it in vmd.
        """
        start_vmd_time = time.time()
        for i in self.all_settings['tcl']['cube_files'].split(' '):

            if not io.path_leads_somewhere(i.strip()):
                msg = "Sorry I couldn't find the following cube file:"
                msg += "\n%s" % i.strip()
                EXC.ERROR(msg)

        self.all_settings['tcl']['pic_filename'][self.PID] = self.tga_filepath
        io.vmd_variable_writer(self.all_settings, self.PID)
        # check if the file exists
        tmp = os.path.isfile(self.all_settings['vmd_script'][self.PID])
        if not tmp:
            msg = "Sorry I can't find the vmd script!"
            msg += "It hasn't been created (or created in the wrong place)."
            EXC.ERROR(msg)

        cond = 'tga' not in self.all_settings['files_to_keep']
        cond *= not all_settings['calibrate']
        if cond:
            self.all_settings['delete_these'].append(self.tga_filepath)

        io.VMD_visualise(self.all_settings, self.PID)

        end_time = time.time() - start_vmd_time
        self.all_settings['times']['VMD Visualisation'][step] += end_time

    # Handles the writing of the necessary files
    def _write_cube_file(self, step, mol_id):
        """
        Converts each molecular wavefunction to a cube file to be loaded in vmd
        """
        start_data_write_time = time.time()
        if all_settings['keep_cube_files']:
            data_filename = "%i-%s.cube" % (step, mol_id)
        else:
            data_filename = "tmp%i-%s.cube" % (mol_id, self.PID)
        data_filepath = self.all_settings['data_fold'] + data_filename
        if not all_settings['keep_cube_files']:
            self.all_settings['delete_these'].append(data_filepath)
        self.data_files_to_visualise = [data_filepath] + \
            self.data_files_to_visualise

        self.all_settings['tcl']['cube_files'] = \
            ' '.join(self.data_files_to_visualise)

        self.tga_folderpath, _, self.tga_filepath = io.file_handler(
                                               self.all_settings['img_prefix'],
                                               'tga',
                                               self.all_settings)
        if all_settings['draw_time']:
            replace = str(self.all_settings['Mtime-steps'][self.step])
            tLabelTxt = self.all_settings['time_lab_txt'].replace("*", replace)
            self.all_settings['tcl']['time_step'] = '"%s"' % (tLabelTxt)
        self.all_settings['tcl']['cube_files'] = ' '.join(
                                                   self.data_files_to_visualise
                                                         )
        io.open_write(data_filepath, self.cube_txt)

        end_time = time.time() - start_data_write_time
        self.all_settings['times']['Write Cube File'][step] += end_time

    # Handles the plotting of the side graph.
    def _plot(self, step):
        """
        Unsupported.

        Will plot a graph along side the visualisation.
        """
        # Plotting if required
        start_plot_time = time.time()
        import matplotlib.pyplot as plt
        files = {'name': "G%i" % step, 'tga_fold': self.tga_filepath}
        self.all_settings['delete_these'].append(io.plot(self.all_settings,
                                                         self.step,
                                                         files,
                                                         plt))
        end_time = time.time() - start_plot_time
        self.all_settings['times']['Plot and Save Img'][step] += end_time

    # Runs the garbage collection and deals with stitching images etc...
    def _finalise(self, num_steps):
        """
        Updates the setting file with changes that occured in the vmd file.
        Will also display the img or stitch the movie. It will finally collect
        garbage.
        """
        io.settings_update(self.all_settings)
        if not all_settings['calibrate']:
            self._stitch_movie()
        else:
            self._display_img()
        self._store_imgs()
        self._garbage_collector()

    # Show the image in VMD or load the image in a default image viewer
    def _display_img(self):
        """
        Displays the created image in the default viewer. Only works in linux!
        """
        if self.all_settings['mols_plotted'] > 0:
            if self.all_settings['load_in_vmd']:
                self.all_settings['tcl']['pic_filename'][self.PID] = \
                    self.tga_filepath
                io.vmd_variable_writer(self.all_settings, self.PID)
                os.system("vmd -nt -e %s" % (
                                     self.all_settings['vmd_script'][self.PID]
                                            )
                          )
                io.settings_update(self.all_settings)
            if self.all_settings['show_img_after_vmd']:
                open_pic_cmd = "xdg-open %s" % (self.tga_filepath)
                subprocess.call(open_pic_cmd, shell=True)
        else:
            EXC.WARN("There were no wavefunctions plotted on the molecules!")

    # Handles Garbage Collection
    def _garbage_collector(self):
        """
        Deletes temporary files.
        """
        self.all_settings['delete_these'].append(
                                            io.folder_correct('./vmdscene.dat')
                                                )
        # Garbage collection
        self.all_settings['delete_these'].append(self.all_settings['f.txt'])
        for f in self.all_settings['delete_these']:
            if io.path_leads_somewhere(f):
                os.remove(f)

    # Handles converting the image to another img format for storing
    def _store_imgs(self):
        """
        Will convert images from tga to jpg (for storage). jpg is smaller than
        tga.
        """
        # Convert all .tga to .img
        if 'img' in self.all_settings['files_to_keep']:
            cnvt_command = "mogrify -format %s %s*.tga" % (
                                               self.all_settings['img_format'],
                                               self.tga_folderpath
                                                          )
            subprocess.call(cnvt_command, shell=True)

    # Need to change all the filenames of the tga files to add leading zeros
    # Stitches the movie together from other files
    def _stitch_movie(self):
        """
        Stitches the individual images together into a movie using the ffmpeg
        binary in the bin/ folder.
        """
        io.add_leading_zeros(self.tga_folderpath)
        files = "*.tga"
        # Creating the ffmpeg and convert commands for stitching
        if self.all_settings['movie_format'] == 'mp4':
            title_path = self.tga_folderpath + self.all_settings['title']
            Stitch_cmd, tmp, _ = io.stitch_mp4(
                                             files,
                                             self.tga_folderpath,
                                             title_path,
                                             self.all_settings['movie_length'],
                                             self.all_settings['ffmpeg_bin']
                                              )
            self.all_settings['delete_these'].append(tmp)
            self.all_settings['delete_these'].append(_)

        subprocess.call(Stitch_cmd, shell=True)  # Actually stitch the movie

    # Prints the timing info
    def __print_timings(self, step, num_steps, start_step_time):
        """
        Will pretty print the timings
        """
        tmpTime = time.time() - start_step_time
        timeTaken = typ.seconds_to_minutes_hours(tmpTime, "CPU: ")
        timeStep = self.all_settings['Mtime-steps'][self.step]
        msg = "Trajectory %i/%i    %s    Timestep %s" % (step + 1,
                                                         num_steps,
                                                         timeTaken,
                                                         timeStep)
        traj_print = "\n"+txt_lib.align(msg, 69, "l") + "*"

        if self.all_settings['verbose_output']:
            print("*"*70)
            print(traj_print)
            io.times_print(self.all_settings['times'], step, 70, tmpTime)
        else:
            io.print_same_line(traj_print, sys, print)
        if self.all_settings['verbose_output']:
            print("*"*70, "\n")
        self.all_settings['times_taken'].append(time.time() - start_step_time)


all_settings['img_prefix'] = consts.Orig_img_prefix.replace("$fs_", "")

tgaFiles = [io.file_handler(all_settings['img_prefix'],
                            'tga',
                            all_settings)[2]
            for step in INIT.all_steps]

all_settings['to_stitch'] = '\n'.join(tgaFiles)

errors = {}
step_data = MainLoop(INIT.all_settings, INIT.all_steps, errors)
