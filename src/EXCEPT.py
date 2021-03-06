"""
Contains code used in exceptions/errors.
"""

# Handles general errors with a message (for some reason standard python XXError("..") won't work)
def ERROR(message, line_number=False):
    if type(line_number) != int:
        raise SystemError(message)
    else:
        raise SystemError(message+"\n\nAt line %i"%line_number)
    return 0



# Handles warnings that could cause the code to give wrong results but don't need to stop it necessarilly
def WARN(message, line_number=False):

    if type(line_number) != int:
        print("\n\n-vv-WARNING-vv-\n")
        print(message)
        print("\n-^^-WARNING-^^-\n\n")
    else:
        print("\n\n-vv-WARNING-vv-\n")
        print(message)
        print("\nAt line %i"%line_number)
        print("\n-^^-WARNING-^^-\n\n")
    return 0




# Will handle the missing permanent_settings file error
def replace_perm_settings():
    """
    If the Templates/permanent_settings.py file is deleted
    for any reason this function will create the file and
    populate it with defaults.
    """

    # The default settings
    s = '''"""
This file contains settings that are needed between runs. This is created and
changed by the code. If this file is corrupted simply delete it and re-run the
code.
"""

previous_path = ''
previous_runtime = '01/01/1900 01:01:01'
tachyon_path = './bin/tachyon_LINUXAMD64'
time_format = "%d/%m/%y %M:%H:%S"
created_docs = False
previous_calibrate = False'''
    with open("./Templates/permanent_settings.py", "w") as f:
        f.write(s)
