import json
import os
import sys
from os.path import isfile,join

# check if a folder name was specified
if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = os.getcwd()

# array to hold invalid and valid files
invalid_json_files = []
read_json_files = []

def parse():
    # loop through the folder
    for files in os.listdir(folder):
        # check if the combined path and filename is a file
        if isfile(join(folder,files)):
            # open the file
            with open(join(folder,files)) as json_file:
                # try reading the json file using the json interpreter
                try:
                    json.load(json_file)
                    read_json_files.append(files)
                except ValueError as e:
                    # if the file is not valid, print the error 
                    #  and add the file to the list of invalid files
                    print("JSON object issue: %s" % e)
                    invalid_json_files.append(files)
    print(invalid_json_files)
    print(len(read_json_files))
    
parse()