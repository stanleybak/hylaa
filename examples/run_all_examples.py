'''
Stanley Bak
Run all the examples (useful as a sort of integration test)
April 2017
'''

import imp
import time
import os

def main():
    'main code'

    script_dir = os.path.dirname(os.path.realpath(__file__))
    python_files = get_files(script_dir, '.py')

    start = time.time()

    for filepath in python_files:
        if "counterexample" in filepath: # skip some files
            continue

        # this script itself will also be a member of the python_files list... skip it
        filename = os.path.split(__file__)[-1]
        if filename in os.path.split(filepath):
            continue

        print(f"\nRunning example: {filepath}")

        run_example(filepath)

    diff = time.time() - start
    print("\nDone! Ran all examples in {:.1f} seconds".format(diff))

def get_files(filename, extension='.py'):
    '''recursively get all the files with the given extension in the passed-in directory'''
    rv = []

    if os.path.isdir(filename):
        # recursive case
        file_list = os.listdir(filename)

        for f in file_list:
            rv += get_files(filename + "/" + f)

    elif filename.endswith(extension):
        # base case
        rv.append(filename)

    return rv

def run_example(filepath):
    'run a hylaa example at the given path'

    path_parts = os.path.split(filepath)

    mod_name, _ = os.path.splitext(path_parts[-1])
    loaded_module = imp.load_source(mod_name, filepath)

    run_hylaa_func = getattr(loaded_module, 'run_hylaa')
    #define_settings_func = getattr(loaded_module, 'define_settings')

    #settings = define_settings_func()

    working_dir = os.getcwd()
    mod_directory = "/".join(path_parts[:-1])
    os.chdir(mod_directory)

    run_hylaa_func()

    os.chdir(working_dir)

if __name__ == "__main__":
    main()
