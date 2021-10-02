# Change these 4 parameters to control the program


# The file which contains the molecular coeffs -this is used to determine how many molecules are in the system
coeff_file="run-coeff-1.xyz"

# Each AOM file that needs changing
AOM_files=("CP2K_HOMO_AOM_COEFF.include", "CP2K_LUMO_AOM_COEFF.include")

# How many atoms are in each molecule (can only have 1 species)
atoms_per_molecule=24

# Either 'all' or 'same'.
#   'all' will make all molecules active.
#   'same' will keep whichever molecules are active in the original AOM file active and all other inactive.
repeat_type="all"





#################################################################################
# First get num mols in coeff file
with open(coeff_file, 'r') as f:
    ltxt = f.read().split("\n")

count = 0
first_2_i=[]
for line_num, line in enumerate(ltxt):
    if 'i =' in line:
        first_2_i.append(line_num)
        count += 1

    if count == 2: break
num_mols = first_2_i[1] - first_2_i[0] - 2


#############################################################################
# Now re-write the AOM_COEFF so the same num of mols are active but
#  we have enough coeffs for each atom.
for fp in AOM_files:
    with open(fp, 'r') as f:
        ltxt = f.read().split("\n")

    # Get which mols are active and the AOM coeffs for 1 active mol
    mols_active = set()
    for i in range(len(ltxt)//atoms_per_molecule):
        splitter = ltxt[i*atoms_per_molecule].split()
        if (splitter[0] != 'XX'):
            mols_active.add(i)

    first_mol = mols_active.pop()
    mols_active.add(first_mol)
    active_AOM = "\n".join(ltxt[first_mol*atoms_per_molecule: (first_mol+1)*atoms_per_molecule])
    inactive_AOM = "XX   1    0   0.0   0.0\n" * atoms_per_molecule
    inactive_AOM = inactive_AOM.rstrip("\n")

    # Construct the AOM coeff file with inactive mols where none where previously
    full_AOM = []
    if repeat_type == "same_mols":
        for i in range(num_mols):
            if i in mols_active:
                full_AOM.append(active_AOM)
            else:
                full_AOM.append(inactive_AOM)

    elif repeat_type == "all":
        full_AOM = [active_AOM] * num_mols

    with open(fp+"_adjusted", 'w') as f:
        f.write('\n'.join(full_AOM))


