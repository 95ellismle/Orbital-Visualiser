import os

# Handles general errors with a message (for some reason standard python XXError("..") won't work)
def ERROR(message, line_number=False):
    if type(line_number) != int:
        print(message)
        os._exit(0)
    else:
        print(message)
        print("\nAt line %i"%line_number)
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
