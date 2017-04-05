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

    python_files = get_files('.', '.py')

    start = time.time()

    for filepath in python_files:
        print "Running example: {}".format(filepath)

        run_example(filepath)

    diff = time.time() - start
    print "Done! Ran all examples in {:.1f} seconds".format(diff)

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

    mod_name, _ = os.path.splitext(os.path.split(filepath)[-1])
    loaded_module = imp.load_source(mod_name, filepath)

    run_hylaa_func = getattr(loaded_module, 'run_hylaa')
    define_settings_func = getattr(loaded_module, 'define_settings')

    settings = define_settings_func()
    run_hylaa_func(settings)

if __name__ == "__main__":
    main()
