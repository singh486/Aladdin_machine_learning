import json
import os
import sys
from os.path import isfile, join

# check if a rootdir name was specified
if len(sys.argv) > 1:
    rootdir = sys.argv[1]
else:
    # otherwise use current directory
    rootdir = os.getcwd()

# array to hold invalid and valid files
invalid_json_files = []
read_json_files = []


def parse():
    # loop through the rootdir
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # check if the combined path and filename is a json file
            if isfile(os.path.join(subdir, file)) and os.path.join(subdir, file)[-5:] == '.json':
                # open the file
                with open(os.path.join(subdir, file)) as json_file:
                    # try reading the json file using the json interpreter
                    try:
                        json.load(json_file)
                        read_json_files.append(os.path.join(subdir, file))
                    except ValueError as e:
                        # if the file is not valid, print the error
                        #  and add the file to the list of invalid files
                        invalid_json_files.append(os.path.join(
                            subdir, file) + "\nJSON object issue: %s\n" % e)

    # sort the invalid json files alphabetically
    invalid_json_files.sort()
    for invalid in invalid_json_files:
        print(invalid)

    # print all read json files
    print("All read json files:")
    for read in read_json_files:
        print(read)

parse()
