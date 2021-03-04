import json
import glob
import os
import re
import pandas as pd


def concat_files():
    read_files = glob.glob("FD_5/*.json")
    with open("merged_file.json", "wb") as outfile:
        outfile.write('[{}]'.format(
            ','.join([open(f, "r").read() for f in read_files])).encode())


def count_dict():
    res = {}
    with open("action_config.txt") as f:
        for line in f:
            split = line.rsplit(' ', 1)
            res[split[0]] = int(split[1])
    return res

def merged_dict(cDictionary):
    res = {}
    with open('merged_config.json') as f:
        data = json.load(f)
        for key in data:
            res[key] = 0
            for val in data[key]:
                res[key] += cDictionary[val]
    return res

def parse_dataframe():
    # Dictionary to store total counts
    total = []

    # Dictionary to store action counts
    cDictionary = count_dict()

    for file in glob.glob("FD_5/*.json"):
        with open(file) as f:

            try:
                data = json.load(f)

                # For each JSON file count particular activities
                for item in data["Activities"]:
                    for subitem in item:
                        if subitem in cDictionary:
                            cDictionary[subitem] = cDictionary[subitem] + 1
                        if subitem == 'EnergyAnnualAnalysis':
                            total.append(
                                item['EnergyAnnualAnalysis']['Solar Panels']['Total'])

            except Exception as e:
                print("Error with ", file)
                print(str(e))

    # New merged dictionary
    mDictionary = merged_dict(cDictionary)
    action_df = pd.DataFrame(list(mDictionary.items()),
                             columns=['Actions', 'Counts'])
    total_df = pd.DataFrame(total, columns=['Energy Totals'])
    print(action_df)
    print(total_df)


if __name__ == "__main__":
    # concat_files()
    parse_dataframe()
