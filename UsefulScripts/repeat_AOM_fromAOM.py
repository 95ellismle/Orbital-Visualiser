# Change these 4 parameters to control the program


# The file which contains the AOM_COEFF file used in the CP2K simulation.
AOM_File_Template="AOM_COEFF.include"

# Each AOM file that needs changing
AOM_files=("CP2K_HOMO_AOM_COEFF.include", "CP2K_LUMO_AOM_COEFF.include")

# How many atoms are in each molecule (can only have 1 species)
atoms_per_molecule=24

# Either 'all' or 'same'.
#   'all' will make all molecules active.
#   'same' will keep whichever molecules are active in the original AOM file active and all other inactive.
repeat_type="same"





#################################################################################
# First get num mols in coeff file
with open(AOM_File_Template, 'r') as f:
    ltxt = f.read().split("\n")
if repeat_type == "all":
    active_mols_template = [True] * int(len(ltxt)//atoms_per_molecule)
elif repeat_type == "same":
    active_mols_template = [line.split()[0] != 'XX' for line in ltxt[::atoms_per_molecule] if line]
else:
    raise SystemExit("Don't understand the repeat type. Please check your code.")

#############################################################################
# Now re-write the AOM_COEFF so the same num of mols are active but
#  we have enough coeffs for each atom.
for fp in AOM_files:
    with open(fp, 'r') as f:
        ltxt = f.read().split("\n")


    for i, line in enumerate(ltxt[::atoms_per_molecule]):
        if line.split()[0] != "XX":
            active_AOM = "\n".join(ltxt[i*atoms_per_molecule: (i+1)*atoms_per_molecule])
            break
    else:
        raise SystemExit("I can't find any active molecules in the input AOM file '%s' " %fp)

    inactive_AOM = "XX   1    0   0.0   0.0\n" * atoms_per_molecule
    inactive_AOM = inactive_AOM.rstrip("\n")

    # Construct the AOM coeff file with inactive mols where none where previously
    full_AOM = []
    for i in active_mols_template:
        if i:
            full_AOM.append(active_AOM)
        else:
            full_AOM.append(inactive_AOM)

    with open(fp+"_adjusted", 'w') as f:
        f.write('\n'.join(full_AOM))


