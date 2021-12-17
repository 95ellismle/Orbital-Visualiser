"""
Contains functions that read the pvecs, the coordinates and the coefficients.

N.B. Maybe the AOM_COEFF file reader should be here...
"""
from src import IO as io
from src import load_xyz as XYZ
from src import type as typ
from src import consts
from src import geometry as geom
from src import math

import io
import numpy as np
import pandas as pd
import re


class File:
    """Abstract class to save typing"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.ftxt = self._read()
        self._parse()

    def _read(self):
        with open(self.filepath, 'r') as f:
            txt = f.read()
        return txt

    def _parse(self):
        """to override"""
        print(self.ftxt)
        pass


class DecompFile(File):
    """Will read and parse a DECOMP file."""
    active_mols = set()
    num_active_ats = 0

    def _parse(self):
        # Parse active mols
        _ = re.findall('INDEX_MOL_DECOMP [0-9 ]+', self.ftxt)
        if _:
            self.active_mols = set(map(int, _[0][16:].split()))
            self.active_mols = {i-1 for i in self.active_mols}  # CP2K is 1 indexed
        else:
            raise SystemExit("Can't parse 'INDEX_MOL_DECOMP' from decomp file.")

        # Parse num active mols
        _ = re.findall('NUM_ACTIVE_ATOMS [0-9]+', self.ftxt)
        if _:
            self.num_active_ats = int(_[0][16:])
        else:
            raise SystemExit("Can't parse 'NUM_ACTIVE_ATOMS' from decomp file.")


class AOMAtom:
    element = None
    atomic_number = None
    orbital = None
    s_coeff = None
    p_coeff = None
    coeff = None
    orb_func = None

    def __init__(self, element, at_num,
                 orbital, s_coeff, p_coeff):
        self.element = element
        self.atomic_number = int(at_num)
        self.orbital = int(orbital)
        self.s_coeff = float(s_coeff)
        self.p_coeff = float(p_coeff)

        if abs(self.p_coeff) > 1e-5:
            self.coeff = self.p_coeff
            assert abs(self.s_coeff) <= 1e-5
        elif abs(self.s_coeff) > 1e-5:
            self.coeff = self.s_coeff
            assert abs(self.p_coeff) <= 1e-5
        else:
            self.coeff = 0

        self._get_orb_func()

    def _get_orb_func(self):
        """Will set the correct orbital function"""
        try:
            orb_map = consts.at_num_orb_map[self.atomic_number]
        except KeyError:
            raise SystemExit(f"Can't find the orbital type for element with atomic number {self.atomic_number}")
        self.orb_func = math.spherical_harmonics.get(orb_map[0], {}) \
                                                .get(orb_map[1], None)

    def __repr__(self):
        return f'Element: {self.element}, at num: {self.atomic_number}, p_coeff: {self.p_coeff}'

    def __eq__(self, obj):
        if not isinstance(obj, AOMAtom):
            return False
        return all((self.element == obj.element,
					self.atomic_number == obj.atomic_number))


class AOMFile(File):
    """Will read and parse the AOM coefficients file"""
    _active_atoms = None
    _active_mols = None
    _at_data = None
    _single_mol = None

    def __init__(self, filepath, ats_per_site, number_molecules):
        self.ats_per_site = ats_per_site
        self.nmol = number_molecules

        self._at_data = {}
        self._active_atoms = set()
        self._active_mols = set()
        self._single_mol = False

        super().__init__(filepath)
        if self._single_mol:
            self.atom_is_active = True

    def atom_is_active(self, atom):
        """Will check if an atom is active or not"""
        return atom in self._active_atoms

    def get_active_atoms(self):
        """Getter for active atoms"""
        return self._active_atoms

    def get_active_mols(self):
        """Getter for active atoms"""
        return self._active_mols

    def __getitem__(self, key):
        """Will get an atom's properties"""
        if self._single_mol:
            key = key % self.ats_per_site
        return self._at_data[key]

    def __iter__(self):
	    for i in self._at_data:
		    yield i

    def set_active_mol(self, act_mol=None):
        """Will set the active molecules set"""
        if act_mol is not None:
            self._active_mols = set(act_mol)

        elif self._single_mol:
            self._active_mols = set(range(self.nmol))

        else:
            self._active_mols = {i // self.ats_per_site
                                 for i in self._active_atoms}
        self._active_atoms = {j + (i * self.ats_per_site)
                                for i in self._active_mols
                                for j in range(self.ats_per_site)}

    def _parse(self):
        """parse the file into AOMAtoms"""
        ltxt = self.ftxt.strip('\n').split('\n')
        nlines = len(ltxt)
        if nlines == self.ats_per_site: # If we only have 1 mol's worth of data
            self._single_mol = True

        # Parse
        for iatom, line in enumerate(ltxt):
            splitter = line.split()
            if splitter == 'XX': continue
            self._active_atoms.add(iatom)
            self._at_data[iatom] = AOMAtom(*splitter)

        self.set_active_mol()


class XYZFile:
    """
    Will parse an xyz file.

    Args:
        filepath: the path to the xyz file
        num_data_cols: how many columns hold data in the of the data sections
        metadata: a dict containing timesteps, num steps, num atoms etc...
        tsteps_to_read: which steps to parse
        atoms_to_read: which atoms to parse

    N.b. num_data_cols = how many columns have data
    """
    atoms_to_read = None
    filepath = None
    num_data_cols = None
    tsteps_to_read = None
    _max_step_to_read = None
    metadata = None

    def __init__(self, filepath, num_data_cols,
                 metadata, tsteps_to_read=None,
                 atoms_to_read=None):
        self.filepath = filepath
        self.num_data_cols = num_data_cols
        self.tsteps_to_read = tsteps_to_read
        if atoms_to_read is not None:
            self.atoms_to_read = set(atoms_to_read)

        # Set some attributes from the metadata
        for key in ('num_atoms', 'time_steps', 'num_lines_in_step',
                    'num_title_lines', 'num_cols', 'num_steps'):
            setattr(self, key, metadata[key])

        # Convert timesteps to step inds
        if self.tsteps_to_read is None:
            self.steps_to_read = set(range(self.num_steps))
        else:
            self.tsteps_to_read = set(self.tsteps_to_read)
            self.steps_to_read = {i for i in range(self.num_steps)
                                  if self.time_steps[i] in self.tsteps_to_read}
            self._max_step_to_read = max(self.tsteps_to_read)

        self._parse()

    def _parse(self):
        """Open file and parse it"""
        cols = []
        data = []
        timings = {}

        if self.atoms_to_read is not None:
            assert isinstance(self.atoms_to_read, set)
            ats_to_read = [i in self.atoms_to_read for i in range(self.num_atoms)]

        with open(self.filepath, 'r') as f:
            # If we want to read all atoms
            for istep in range(self.num_steps):
                if istep not in self.steps_to_read:
                    if istep > self._max_step_to_read:
                        break
                    # Skip that whole step (with loop unrolling)
                    for i in range(self.num_lines_in_step // 5):
                        next(f)
                        next(f)
                        next(f)
                        next(f)
                        next(f)
                    for i in range(self.num_lines_in_step % 5):
                        next(f)
                    continue

                # Skip the title lines
                for i in range(self.num_title_lines): next(f)

                # Create the filetxt string to be read in
                if self.atoms_to_read is None:
                    filetxt = ''.join((next(f) for i in range(self.num_atoms)))
                else:
                    l = []
                    for i in ats_to_read:
                        if i: l.append(next(f))
                        else: next(f)
                    filetxt = ''.join(l)

                # Read 1 step
                headers = list(map(str, range(self.num_cols)))
                dtypes = {i: object for i in headers}
                data_cols = headers[-self.num_data_cols:]
                col_cols = headers[:-self.num_data_cols]
                for i in data_cols:
                    dtypes[i] = np.float32
                step = pd.read_csv(io.StringIO(filetxt),
                                   names=headers,
                                   delim_whitespace=True,
                                   dtype=dtypes)

                # Split data and other cols
                if len(col_cols) == 1:
                    cols.append(step.loc[:, col_cols[0]].to_numpy())
                else:
                    cols.append(step.loc[:, col_cols].to_numpy())
                data.append(step.loc[:, data_cols].to_numpy())

        self.data = np.array(data)
        self.cols = np.array(cols)


# Will read the coordinates and combine them in the specified way
def read_coords(all_settings):
    """Read the positions and save them in the all_settings"""
    if all_settings['do_transition_state']:
        act_ats = all_settings['LUMO'].get_active_atoms()
    else:
        act_ats = all_settings['AOM'].get_active_atoms()

    # If we want to write the background mols then read all positions
    if all_settings['background_mols']:
        act_ats = None

    pos_data = XYZFile(all_settings['pos_file'],
                       num_data_cols=3,
                       metadata=all_settings['pos_metadata'],
                       tsteps_to_read=all_settings['nucl_tsteps_to_read'],
                       atoms_to_read=act_ats,
                       )

    # Reading the nuclear positions
    all_settings['coords'] = pos_data.data * consts.ang2bohr
    # We only read the atom numbers for the first step and assume they're always the same
    all_settings['at_num'] = np.array([typ.atomic_num_convert(i) for i in pos_data.cols[0]])
    all_settings['Ntime-steps'] = all_settings['nucl_tsteps_to_read']


def read_coeffs(all_settings):
    """Will read the coefficient files"""
    # Reading the mol coeffs
    all_mol_data = XYZFile(all_settings['coeff_file'],
                           num_data_cols=2,
                           metadata=all_settings['coeff_metadata'],
                           tsteps_to_read=all_settings['coeff_tsteps_to_read'],
                           )

    all_settings['Mtime-steps'] = all_settings['coeff_tsteps_to_read']
    all_settings['pops'] = np.linalg.norm(all_mol_data.data, axis=-1)
    all_settings['mol'] = np.zeros(all_mol_data.data.shape[:-1], dtype=complex)
    all_settings['mol'].real = all_mol_data.data[:, :, 0]
    all_settings['mol'].imag = all_mol_data.data[:, :, 1]


# Will read the pvecs file
def read_pvecs(all_settings):
    """
    Always create our own pvecs.

    Maybe I'll fix the reader one day -though it's very cheap to re-create them.
    """
    # If there is no pvecs file and the code has been instructed to create them.
    all_settings['pvecs'] = False

    crd_shape = np.shape(all_settings['coords'])
    nmol = crd_shape[1] / float(all_settings['atoms_per_site'])
    if int(nmol) != nmol:
      raise SystemExit("The number of 'atoms per site' doesn't divide perfectly into the 'number of atoms'.")
    nmol = int(nmol)

    all_settings['mol_nums'] = np.arange(crd_shape[1]) // all_settings['atoms_per_site']
    all_settings['mol_coords'] = np.reshape(all_settings['coords'], (crd_shape[0],
                                                                     nmol,
                                                                     all_settings['atoms_per_site'],
                                                                     3))
    return
#
#    # Read all pvecs
#    all_pvecs = [XYZ.read_xyz_file(f,
#                                   num_data_cols=3,
#                                   do_timesteps=all_settings['coeff_steps_to_read'].union(all_settings['nuclear_steps_to_read']),
#                                   metadata=all_settings['pvecs_metadata'])[0]
#             for f in all_settings['CP2K_output_files']['pvecs']]
#
#    # Check the pvecs (sometimes the initial step in the pvecs file gives all zeros)
#    if all_settings['calibrate'] and np.sum(all_pvecs[0][0]) == 0:
#        all_pvecs = [io.read_xyz_file(f,num_data_cols=3,
#                                      min_step=1, max_step=2, stride=1)[0]
#                     for f in all_settings['CP2K_output_files']['pvecs']]
#
#    # Average all replicas
#    if all_settings['mean_rep']:
#        all_settings['pvecs'] = np.mean(all_pvecs, axis=0)
#        if len(all_settings['pvecs']) == 0: raise SystemExit("Haven't read any pvecs!")
#
#    # Same check as before but this time we'll use the combined pvecs
#    if np.sum(all_settings['pvecs'][0]) == 0 and not all_settings['calibrate']:
#        all_settings['pvecs'][0] = all_settings['pvecs'][1]
#
#    if not (len(all_settings['pvecs']) == len(all_settings['mol']) and len(all_settings['pvecs']) == len(all_settings['coords'])):
#        raise SystemExit("""Sorry something has gone wrong with the reading of the data.
#
#The number of timesteps parsed in the pvecs array isn't equal to the coefficients or the coords.
#You can let Matt know and he will make the code work for different length arrays (or you could make sure that
#you print the pvecs and coefficients for every timestep you print the coordinates.)
#
#    * len(pvecs) = %i
#    * len(coeffs) = %i
#    * len(positions) = %i
#"""%(len(all_settings['pvecs']), len(all_settings['mol']), len(all_settings['coords'])))
