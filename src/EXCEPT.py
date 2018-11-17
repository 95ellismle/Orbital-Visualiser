# Handles general errors with a message (for some reason standard python XXError("..") won't work)
def ERROR(message, line_number=False):
    if type(line_number) != int:
        raise SystemExit(message)
    else:
        raise SystemExit(message+"\n\nAt line %i"%line_number)
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
    s = """previous_path = ''
previous_runtime = '01/01/1900 01:01:01'
tachyon_path = ''
time_format = "%d/%m/%y %M:%H:%S"
created_docs = False
previous_calibrate = False"""
    with open("./Templates/permanent_settings.py", "w") as f:
        f.write(s)
