from __future__ import print_function
from __future__ import   division
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

import sys
if sys.version_info.major < 3:
   raise SystemExit("This code no longer supports python2! Please use python3 via: `python3 main.py`")

import numpy as np
import time
import os
import subprocess
import sys
import shutil
import multiprocessing as mp

from src import EXCEPT as EXC
from src import geometry as geom
from src import type as typ
from src import math as MT
from src import text as txt_lib
from src import IO as io
from src import consts

from init import INIT
all_settings = INIT.all_settings


if sys.version_info[0] > 2:
    xrange = range
    raw_input = input


class MainLoop(object):

    """ This will carry out all the main functions and actually run the code.
         It will create the data, save it as a cube file, and make vmd render
         it. I put all this in a class as I thought I might try and incorporate
         PyQt or Django with it.
    """

    def __init__(self, all_settings, all_steps, errors):
        self.errors = errors
        self.all_settings = all_settings
        self.neg_iso_cols = {}
        self.pos_iso_cols = {}
        self.tcl_color_dict_count = 0
        self.PID = "MainProcess"
        self.calc_pvecs = self.all_settings['pvecs'] is False

        print("Init complete. Entering mainloop -%i steps to complete." % len(all_steps))
        for step in all_steps:  # Loop over all steps and visualise them
            self.step = step
            self.posStepInd = all_settings['pos_step_inds'][step]

            # Find the phase of the first mol as a reference.
            # phase = angle in complex plane
            self.thetaRef = np.angle(self.all_settings['mol'][self.step][0])
            tmp = consts.Orig_img_prefix.replace(
                    "$fs",
                    "%.2f" % self.all_settings['Mtime-steps'][self.step])
            self.all_settings['img_prefix'] = tmp.replace(".", ",")
            start_time = time.time()

            self._do_step(step)  # Do a visualisation step

            # Pretty print timings
            self.__print_timings(step, len(all_steps), start_time)

        self._finalise(len(all_steps))

    # Completes 1 step
    def _do_step(self, step):
        """
        Will carry out a single visualisation step. This involves creating the
        wavefunction data, writing this to a file and rendering it with VMD.
        This is different to the density step as it creates 2 cube files. 1
        which is negative and 1 which is positive.

        Inputs:
            * step  =>  Which step to visualise.
        """
        # Reset variables
        self.neg_iso_cols = {}
        self.pos_iso_cols = {}
        self.tcl_color_dict_count = 0
        self.__count__ = 0
        self.data_files_to_visualise = []

        # Carry out meat of step
        self._find_active_molecules()
        self._vmd_filename_handling()
        if self.all_settings['background_mols']:
            self._write_background_mols()
        self._nearestNeighbourKeys()  # Find nearest neighbour list

        localisation = MT.IPR(self.all_settings['mol'][self.step])
        print(f"IPR = {localisation}")
        print(f"{len(self.active_step_mols)} Mols have enough population")

        # Ordered by mol coeff max to min
        const = 0.01/self.all_settings['isosurface_to_plot']
        loc = const*localisation if const >= 1.2 else localisation * 1.2
        for count, molID in enumerate(self.active_step_mols):
            if count > loc: break
            self._create1Mol_(molID)

        print("Visualising in VMD")
        self._vmd_visualise(step)  # run the vmd script and visualise the data
        # if self.all_settings['side_by_side_graph']:  # (Not supported)
        #     self._plot(step)  # Will plot a graph to one side (Not supported)

    def _create1Mol_(self, molID):
        molPop = self.all_settings['pops'][self.step][molID]
        if molPop < self.all_settings['min_abs_mol_coeff']:
            return None

        self._findActiveAtoms(molID)
        print('\r'f"(mol {molID}) Found {self.active_coords.shape[1]} active Atoms, now creating data    ", end="\r")
        self._createWfData(molID, self.step)
        print('\r'f"(mol {molID}) Created WF Data, now setting colours     ", end="\r")
        self._reapplyPhase(molID)
        self._setWfCol()
        print('\r'f"(mol {molID}) Set colours, now writing cube files     ", end="\r")
        self._writeCubeFile(self.step, molID)
        self.__count__ += self.writeImagCube + self.writeRealCube

    def __findCubesToWrite(self, molID):
        """
        Will find whether the real or imaginary data has enough data (at least
        0.3%) above the isosurface that is to be plotted.

        This function will also use the percentage useful data in the real and
        imaginary data containers to update the minimum absolute coefficient
        threshold for visualising the molecule.
        """
        # What percentage of the data is visible
        #tol = 0.001 if self.all_settings['do_transition_state'] else 0.0001

        size = float(self.RealData.size)
        isoToPlot = self.all_settings['isosurface_to_plot']
        relevantReal = np.sum(np.abs(self.RealData) > isoToPlot)
        relevantReal /= size
        self.writeImagCube = False
        if all_settings['color_type'] == "phase":
            relevantImag = np.sum(np.abs(self.ImagData) > isoToPlot)
            relevantImag /= size
            self.writeImagCube = False #relevantImag > tol  # At least tol% is visible


        self.writeRealCube = True #relevantReal > tol  # At least tol% of the data structure is visible is visible
        if not self.writeImagCube and not self.writeRealCube:
            molPop = self.all_settings['pops'][self.step][molID]
            if molPop > self.all_settings['min_abs_mol_coeff']:
                self.all_settings['min_abs_mol_coeff'] = molPop

        #if all_settings['color_type'] == "real-phase": self.writeImagCube = False

    def _reapplyPhase(self, molID):
        """
        Will handle the processing of the data after being created. For the
        phase option this involves finding which quadrant the data points lie
        within and then correcting the density information to align with that.

        We create an array that store the phase at each grid point in the cube
        file. With this phase array we then determine whether the point is
        mostly imaginary or real. We also determine whether the point is in a
        negative or positive quadrant. This information is then added in later.

        To add in whether something is negative or positive we simply set all
        the points that lie in the negative quadrant to -'ve.

        To add in the complex or imaginary information we make 2 data arrays.
        One contains the imaginary points one contains the real points. These
        are then plotted later.
        """
        start_data_create_time = time.time()
        if self.all_settings['color_type'] == 'density':
            self.RealData = self.data * np.conjugate(self.data)
            self.RealData = np.sqrt(self.RealData)
            self.ImagData = False  # We don't have imaginary data here

        elif self.all_settings['color_type'] == "phase":
            # Get phase info
            self.phase = np.angle(self.data)
            # Get the density
            self.RealData = self.data * np.conjugate(self.data)
            self.RealData = np.sqrt(self.RealData)

            # Find which quadrants the phase fits into
            #  Positive Real Quadrant
            self.PRmask = self.phase < consts.phaseMasks['PosReal'][1]
            self.PRmask *= self.phase > consts.phaseMasks['PosReal'][0]
            #  Negative Real Quadrant
            # The negative real quadrant falls in the -pi section that is split
            # between 2 octants (see consts.phaseMasks)
            NRmask1 = self.phase < consts.phaseMasks['NegReal'][0][1]
            NRmask1 *= self.phase > consts.phaseMasks['NegReal'][0][0]
            NRmask2 = self.phase < consts.phaseMasks['NegReal'][1][1]
            NRmask2 *= self.phase > consts.phaseMasks['NegReal'][1][0]
            self.NRmask = NRmask1 + NRmask2
            #  Positive Imaginary Quadrant
            self.PImask = self.phase < consts.phaseMasks['PosImag'][1]
            self.PImask *= self.phase > consts.phaseMasks['PosImag'][0]
            #  Negative Imaginary Quadrant
            self.NImask = self.phase < consts.phaseMasks['NegImag'][1]
            self.NImask *= self.phase > consts.phaseMasks['NegImag'][0]

            # Create the imaginary data storage
            self.ImagData = np.copy(self.RealData)

            # Add phase back into the imaginary and real data
            self.RealData[self.NRmask] = -self.RealData[self.NRmask]  # make neg
            self.RealData[self.NImask + self.PImask] = 0  # Set imag part to 0
            self.ImagData[self.NImask] = -self.ImagData[self.NImask]  # make neg
            self.ImagData[self.NRmask + self.PRmask] = 0  # Set real part to 0

        elif self.all_settings['color_type'] == "real-phase":
            # Get phase info
            self.negMask = self.data < 0
            # Get the density
            self.RealData = self.data * np.conjugate(self.data)
            self.RealData = np.sqrt(self.RealData)
            self.RealData[self.negMask] *= -1



        self.__findCubesToWrite(molID)
        end_time = time.time() - start_data_create_time
        self.all_settings['times']['WF Post Processing'][self.step] += end_time

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
        new_bb_scale = np.tanh(float(pop)/float(w))
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
        tmp = self.all_settings['vmd_script_fold'] / f'{self.PID}.tcl'
        self.all_settings['vmd_script'][self.PID] = tmp
        tmp = self.all_settings['vmd_script_fold'] / f'{self.PID}.out'
        self.all_settings['vmd_junk'][self.PID] = tmp
        tmp = self.all_settings['vmd_script_fold'] / f'{self.PID}.error'
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
                          [np.max(self.all_settings['coords'][self.posStepInd][:, i])
                           for i in range(3)]
                               )
        # dims = [Xdims, Ydims, Zdims][largest_dim]
        # Find maximum coordinate
        at_crds = self.all_settings['coords'][self.posStepInd]
        at_crds = at_crds[self.all_settings['atoms_to_plot']][:, largest_dim]
        max_coord = np.max(at_crds)
        max_coord += self.all_settings['background_mols_end_extend']

        # Find coordinates within below this max_coord
        mask = self.all_settings['coords']
        mask = mask[self.step][:, largest_dim] < max_coord

        # Apply the mask to get data. Find metadata.
        background_mols_pos = self.all_settings['coords'][self.posStepInd][mask]
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
        Will find which molecules are active and which can be ignored. This will
        also order the mol inds by the population. This is so we converge to the
        min_abs_mol_coeff via the fastest route.
        """
        # Find mols with enough population (more than min_abs_mol_coeff)
        mask = (self.all_settings['pops'][self.step] >
                self.all_settings['min_abs_mol_coeff'])
        self.active_step_mols = np.arange(0, self.all_settings['nmol'])[mask]
        pops = [self.all_settings['pops'][self.step][i]
                for i in self.active_step_mols]
        sortedMols = reversed(sorted(zip(pops, self.active_step_mols)))
        self.active_step_mols = np.array([i[1] for i in sortedMols])

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
    def _findActiveAtoms(self, molID):
        """
        Find which atoms are active according to the AOM_COEFF.include file.
        These are atoms on a molecule.

        Inputs:
            * molID  =>  The molecule to find active atoms for
        """
        # Find active coordinates (from active atom index)
        atMask = [i for i in self.all_settings['active_atom_index'][molID]]
        self.active_coords = self.all_settings['coords'][self.posStepInd][atMask]
        self.active_coords = [self.active_coords[:, k] for k in range(3)]
        self.active_coords = np.array(self.active_coords)

        # Error Checking
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
    def _createWfData(self, molID, step):
        """
        Will create the wavefunction data. This involves creating a bounding
        box big enough to encapsulate the wf and creating the SOMO (p orbital
        stuff). In this for each molecule we loop over it and its nearest
        neighbours to account for overlap between them. This is an optimisation
        and could probably be improved with a cutoff distance etc...

        The equation for the molecular wf is:
          Psi = \sum_{l} u_l \sum_{v} AOM_{v}[\sum_{i=x,y,z} PVEC_{i, v}|p_{i}>]

        Inputs:
            * molID  =>  The molecule index
            * step    =>  The step number
        """
        start_data_create_time = time.time()

        # Drawing a bounding box around the active atoms to prevent creating
        #  unecessary data
        if self.all_settings['dyn_bound_box']:
            BBS_dyn = [self._dynamic_bounding_box_scale(molID, i)
                       for i in self.all_settings['bounding_box_scale']]
        else:
            BBS_dyn = self.all_settings['bounding_box_scale']

        # act_crds = [self.active_coords[:, k] for k in range(3)]
        bb_center, bb_spans = geom.min_bounding_box(self.active_coords)
        bb_spans += BBS_dyn
        self.sizes = typ.int_res_marry(bb_spans,  # How many grid points
                                       self.all_settings['resolution'],
                                       [1, 1, 1])

        # Create wf data
        self.sizes = np.array(self.sizes)
        scale_factors = self.sizes  * self.all_settings['resolution']
        self.origin = scale_factors/-2. + bb_center

        # Actually create the data
        self.data = np.zeros(self.sizes, dtype=np.complex64)

        # Get AOM coeffs
        if self.all_settings['do_transition_state']:
            aom_coeffs = [(self.all_settings['LUMO'][i].coeff, self.all_settings['HOMO'][i].coeff)
                          for i in self.all_settings['active_atom_index'][molID]]
        else:
            aom_coeffs = [(self.all_settings['AOM'][i].coeff,)
                         for i in self.all_settings['active_atom_index'][molID]]
        aom_coeffs = np.swapaxes(aom_coeffs, 0, 1)

        u_l = self.all_settings['pops'][self.step, molID]

        do_neighbours = self.all_settings['pops'][self.step][molID] > 0.07
        for molNum in self.nearestNeighbours[molID]:  # loop nearest mols
            # Only use nearest neighbours if there's a large enough population
            if molNum != molID and not do_neighbours:
                continue

            u_l = self.all_settings['mol'][self.step][molNum]
            if self.all_settings['pops'][self.step][molNum] > 0.1:
                all_at_data = self.__calc_SOMO(molID, bb_center) * u_l

                if self.all_settings['do_transition_state']:
                    self.data += (np.conjugate(np.sum(all_at_data * aom_coeffs[0, :, None, None, None], axis=0))
                                  * np.sum(all_at_data * aom_coeffs[1, :, None, None, None], axis=0))
                else:
                    self.data += np.sum(all_at_data * aom_coeffs[0, :, None, None, None], axis=0)

        end_time = time.time() - start_data_create_time
        self.all_settings['times']['Create Wavefunction'][step] += end_time

    def __calc_SOMO(self, molID, translation):
        """
        Will create the SOMO for 1 molecule. This involves looping over all
        active atoms in one molecule and creating a p orbtial on each one. This
        is orientated via the pvecs, and it's size and (real) phase is
        determined by AOM_COEFF.

        Args:
            * molID => The ID of the molecule
            * translation => Where to move the spherical harmonic to make it fit on the atom

        Returns
            The pvecs * spherical harmonic data for each atom
            in an array of shape (n_act_atom, size_x, size_y, size_z)
        """
        # Loop over current molecules atoms
        act_ats = self.all_settings['active_atom_index'][molID]
        tmpData = np.zeros((len(act_ats), *self.sizes), dtype=np.complex64)
        if self.calc_pvecs:
            start_time = time.time()

            mol_ats = np.array(self.all_settings['reversed_mol_info'][molID])
            ats = self.all_settings['coords'][self.posStepInd]
            ats = ats[mol_ats]
            act_ats = self.all_settings['active_atom_index'][molID]
            assert all(molID == np.array(act_ats) // all_settings['atoms_per_site'])
            first_at_in_mol_ind = molID * all_settings['atoms_per_site']

            pvecs_all = geom.calc_pvecs_1mol(ats, act_ats - first_at_in_mol_ind)

            self.all_settings['times']['Create Pvecs'][self.step] += time.time() - start_time
        else:
            if 'Create Pvecs' in self.all_settings['times']:
                self.all_settings['times'].pop("Create Pvecs")
            pvecs_all = self.all_settings['pvecs'][self.step][self.all_settings['reversed_mol_info']]

        # Get the spherical harmonic functions
        if self.all_settings['do_transition_state']:
            aom_funcs = [self.all_settings['LUMO'][i].orb_func
                         for i in self.all_settings['active_atom_index'][molID]]
        else:
            aom_funcs = [self.all_settings['AOM'][i].orb_func
                         for i in self.all_settings['active_atom_index'][molID]]

        # Loop over atoms that belong to molecule molID
        center_of_mol_box = (self.sizes / 2).round().astype(int)
        at_chunk_size = self.all_settings['at_box_size']
        for i, iat in enumerate(act_ats):
            at_crds = self.all_settings['coords'][self.posStepInd][iat] - translation
            pvecs = pvecs_all[i]
            if pvecs is None: continue
#            tmpData[i] = MT.dot_3D(
#                                   aom_funcs[i](self.sizes[0],
#                                                self.sizes[1],
#                                                self.sizes[2],
#                                                self.all_settings['resolution'],
#                                                at_crds),
#                                   pvecs)

            # How many units from the center of the molecular (cube file) box is the atom
            chunks_from_center = (at_crds // self.all_settings['resolution']).astype(int)
            diff = ((at_crds / self.all_settings['resolution']) - chunks_from_center)/2
            # Where is the atom's box within the molecular box
            box_loc = chunks_from_center - center_of_mol_box
            # Where in the array should we write the data
            mins = box_loc - at_chunk_size
            maxs = box_loc + at_chunk_size
            spans = maxs - mins

            tmpData[i,
                    mins[0] : maxs[0],
                    mins[1] : maxs[1],
                    mins[2] : maxs[2],
                    ] = MT.dot_3D(aom_funcs[i](spans[0],
                                         spans[1],
                                         spans[2],
                                         self.all_settings['resolution'],
                                         diff),
                                  pvecs)

        return tmpData

    # Creates the cube file to save
    def __createCubeFileTxt(self, step, molID):
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

        # Error Checking
        msg = "The data has an imaginary component and it shouldn't!"
        msg += "\nThis is a problem with the code let Matt know.\n\n"
        msg += "\n\n(Keep the input files and the version of the movie"
        msg += " maker in order for him to fix it)"
        if np.sum(self.RealData.imag) > 1e-12:
            raise SystemExit(msg + '    (Bad Real Data)')
        if self.writeImagCube and type(self.ImagData) == type(np.array(1)):
            if np.sum(self.ImagData.imag) > 1e-12:
                raise SystemExit(msg + '    (Bad Imaginary Data)\n\n')

        act_ats = self.all_settings['reversed_mol_info'][molID]
        coords = self.all_settings['coords'][self.posStepInd, act_ats]
        at_nums = self.all_settings['at_num'][act_ats]

        if self.writeRealCube:
            self.RCubeTxt = txt_lib.cube_file_text(
                                  data = self.RealData.real,
                                  coords = coords,
                                  N_vec = self.sizes,
                                  origin = self.origin,
                                  at_nums = at_nums,
                                  basis_vec = xyz_basis_vectors
            )

        if self.writeImagCube and type(self.ImagData) == type(np.array(1)):
            self.ICubeTxt = txt_lib.cube_file_text(
                                       data = self.ImagData.real,
                                       coords = coords,
                                       N_vec = self.sizes,
                                       origin = self.origin,
                                       at_nums = at_nums,
                                       basis_vec = xyz_basis_vectors
            )

        end_time = time.time() - start_cube_create_time
        self.all_settings['times']['Create Cube Data'][step] += end_time

    # Handles the saving the wf colors in a dictionary of the wavefunction.
    def _setWfCol(self, numCube=-1):
        """
        Will determine the colour of the wavefunction depending on the setting
        chosen. If density is chosen then the wavefunction will all be one
        colour. If real-phase is chosen the wavefunction will be one colour for
        positive values and another for negative values. If full phase is
        chosen the colour will be dependent on which quadrant in the complex
        plane the coefficient appears in.

        22 = density color
        phase Colors:
                           Imag | Real
                      Neg | 19  |  21
                      Pos | 18  |  20
        """
        # Could optimise (and tidy) this, the code doesn't need to do all
        #  this at every step
        if self.all_settings['color_type'] == 'density':
            self.neg_iso_cols[self.tcl_color_dict_count] = 22
            self.pos_iso_cols[self.tcl_color_dict_count] = 22
            self.tcl_color_dict_count += 1

        # Includes real-phase and full phase
        elif 'phase' in self.all_settings['color_type']:
            # First real cube file
            if self.writeRealCube:
                self.pos_iso_cols[self.tcl_color_dict_count] = 20
                self.neg_iso_cols[self.tcl_color_dict_count] = 21
                self.tcl_color_dict_count += 1
            # Then imaginary cube file
            if self.writeImagCube:
                self.pos_iso_cols[self.tcl_color_dict_count] = 18
                self.neg_iso_cols[self.tcl_color_dict_count] = 19
                self.tcl_color_dict_count += 1

        self._saveWfCol()

    # Saves the wavefunction coloring in the tcl dictionary
    def _saveWfCol(self):
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
        neg_col_dict_str = tmp

        tmp = str(self.pos_iso_cols)
        for find, replace in replacers:
            tmp = tmp.replace(find, replace)
        pos_col_dict_str = tmp

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
    def _writeCubeFile(self, step, molID, numCube=-1):
        """
        Converts each molecular wavefunction to a cube file to be loaded in vmd
        """
        self.__createCubeFileTxt(step, molID)
        start_data_write_time = time.time()
        if all_settings['keep_cube_files']:
            RDataFName = "%s-%i-%i.cube" % ('Real', step, molID)
            IDataFName = "%s-%i-%i.cube" % ('Imag', step, molID)
        else:
            RDataFName = "tmp%s-%i.cube" % ('Real', molID)
            IDataFName = "tmp%s-%i.cube" % ('Imag', molID)
        RDataFPath = self.all_settings['data_fold'] / RDataFName
        IDataFPath = self.all_settings['data_fold'] / IDataFName

        if not all_settings['keep_cube_files']:
            self.all_settings['delete_these'].append(RDataFPath)
            self.all_settings['delete_these'].append(IDataFPath)

        if self.writeRealCube:
            self.data_files_to_visualise.append(RDataFPath)
        if self.writeImagCube:
            self.data_files_to_visualise.append(IDataFPath)

        self.all_settings['tcl']['cube_files'] = \
            ' '.join(map(str, self.data_files_to_visualise))

        self.tga_folderpath, _, self.tga_filepath = io.file_handler(
                                               self.all_settings['img_prefix'],
                                               'tga',
                                               self.all_settings)
        # if all_settings['draw_time']:
        #     replace = str(self.all_settings['Mtime-steps'][self.step])
        #     tLabelTxt = self.all_settings['time_lab_txt'].replace("*", replace)
        #     self.all_settings['tcl']['time_step'] = '"%s"' % (tLabelTxt)
        self.all_settings['tcl']['cube_files'] = ' '.join(
                                                          map(str, self.data_files_to_visualise)
                                                         )
        if self.writeRealCube:
            io.open_write(RDataFPath, self.RCubeTxt)
        if self.writeImagCube:
            io.open_write(IDataFPath, self.ICubeTxt)

        end_time = time.time() - start_data_write_time
        self.all_settings['times']['Write Cube File'][step] += end_time

    def _nearestNeighbourKeys(self):
        """
        Will return a dictionary with the molecular indices of the molecules
        that are within the cutoff. The cutoff is given in the settings file.
        """
        self.nearestNeighbours = {}
        act_step_mols = set(self.active_step_mols)

        # Get the atom indices corresponding to the ones on the mol
        mol_at_inds = list(self.all_settings['reversed_mol_info'].values())
        mol_coords = self.all_settings['coords'][self.step, mol_at_inds]
        mol_avg_pos = np.mean(mol_coords, axis=1)
        print(self.all_settings['coords'].shape)

        # Loop over active mols and get nearest neighbours
        for imol in act_step_mols:
            pos = mol_avg_pos[imol]
            dist_between = np.linalg.norm(mol_avg_pos - pos, axis=1)
            dist_mask = np.abs(dist_between) < self.all_settings['nn_cutoff']
            close_mols = all_settings['active_mols'][dist_mask]
            self.nearestNeighbours[imol] = list(act_step_mols.intersection(close_mols))

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
        self._copy_settings_file()
        self._store_imgs()
        if not all_settings['calibrate']:
            self._stitch_movie()
        else:
            self._display_img()

        self._garbage_collector()

    # Copy settings.inp to img folder
    def _copy_settings_file(self):
      """
      Will copy the settings file to wherever the image is stored. Will
      also name it something sensible (to match the image or movie)
      """
      settFilePath = self.all_settings['settings_file']
      incFilePath = self.all_settings['tcl']['vmd_source_file']
      imgFilePath = self.all_settings['tcl']['pic_filename'][self.PID]
      newSettingsFolder = str(imgFilePath).rstrip("_img.tga")
      startBit = newSettingsFolder[:newSettingsFolder.rfind('/')+1]
      if self.all_settings['calibrate']:
          endBit = newSettingsFolder[newSettingsFolder.rfind('/')+1:]
      else:
          endBit = self.all_settings['title']
      newSettingsFolder = startBit + "Settings_for_" + endBit + '/'
      newSettingsFilePath = newSettingsFolder + "settings.inp"
      newIncludeFilePath = newSettingsFolder + "include.vmd"

      if not os.path.isdir(newSettingsFolder):
          os.makedirs(newSettingsFolder)
      shutil.copyfile(src=settFilePath, dst=newSettingsFilePath)
      shutil.copyfile(src=settFilePath, dst=newIncludeFilePath)

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

                vmd_bin = self.all_settings['vmd_exe']
                os.system(f"{vmd_bin} -nt -e {self.all_settings['vmd_script'][self.PID]}")
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
        #self.all_settings['delete_these'].append(self.all_settings['f.txt'])
        for f in all_settings['delete_these']:
            if os.path.isfile(f):
                os.remove(f)

    # Handles converting the image to another img format for storing
    def _store_imgs(self):
        """
        Will convert images from tga to jpg (for storage). jpg creates smaller images
        than tga. We also add any leading zeros to files as this makes them
        easier to stitch together later on.
        """
        # First add leading zeros to files
        new_files, tga_files = io.add_leading_zeros(self.tga_folderpath)
        cond = 'tga' not in self.all_settings['files_to_keep']
        cond *= not all_settings['calibrate']
        if cond: all_settings['delete_these'].extend(set(new_files))

        # Convert all .tga to .<img>
        if 'img' in self.all_settings['files_to_keep']:
            cnvt_command = ["mogrify", "-format", self.all_settings['img_format'],
                            f'{self.tga_folderpath}/*.tga']
            subprocess.run(cnvt_command)

    # Need to change all the filenames of the tga files to add leading zeros
    # Stitches the movie together from other files
    def _stitch_movie(self):
        """
        Stitches the individual images together into a movie using the ffmpeg
        binary in the bin/ folder.
        """
        files = "*.%s" % self.all_settings['img_format']
        # Creating the ffmpeg and convert commands for stitching
        if self.all_settings['movie_format'] == 'mp4':
            os.chmod(self.all_settings['ffmpeg_bin'], int("755", base=8))
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

        else:
            raise SystemExit("Unrecognised `movie_format`. Please choose from:\n\t* mp4")

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


if __name__ == '__main__':
    all_settings['img_prefix'] = consts.Orig_img_prefix.replace("$fs_", "")

    tgaFiles = [io.file_handler(all_settings['img_prefix'],
                                'tga',
                                all_settings)[2]
                for step in INIT.all_steps]

    all_settings['to_stitch'] = '\n'.join(map(str, tgaFiles))

    errors = {}
    step_data = MainLoop(INIT.all_settings, INIT.all_steps, errors)

