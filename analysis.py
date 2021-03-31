import json
import glob
import os
import re
import pandas as pd
from pathlib import Path

import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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


def remove_empty_files_and_folders(dir_path) -> None:
    for root, dirnames, files in os.walk(dir_path, topdown=False):
        for f in files:
            full_name = os.path.join(root, f)
            if os.path.getsize(full_name) == 0:
                os.remove(full_name)

        for dirname in dirnames:
            full_path = os.path.realpath(os.path.join(root, dirname))
            if not os.listdir(full_path):
                os.rmdir(full_path)


def parse_dataframe():
    total = {}
    cDictionary = count_dict()
    mDictionary = merged_dict(cDictionary)
    action_df = pd.DataFrame(mDictionary.keys(),
                                 columns=['Actions'])
    # total_df = pd.DataFrame()

    subfolders = [ f.path for f in os.scandir("Data") if f.is_dir() ]
    filepaths = []
    for sub in subfolders:
        f = [f.path for f in os.scandir(sub) if f.is_dir()]
        filepaths.append(f[0])

    for filepath in filepaths:

        # Reset Dictionary to store action counts
        cDictionary = count_dict()

        total[filepath] = []
        for file in glob.glob(filepath + "/*.json"):
            with open(file) as f:
    
                try:
                    data = json.load(f)
    
                    # For each JSON file count particular activities
                    for item in data["Activities"]:
                        for subitem in item:
                            if subitem in cDictionary:
                                cDictionary[subitem] = cDictionary[subitem] + 1
                            if subitem == 'EnergyAnnualAnalysis':
                                total[filepath].append(
                                    item['EnergyAnnualAnalysis']['Solar Panels']['Total'])

                
    
                except Exception as e:
                    print("Error with ", file)
                    print(str(e))

        # New merged dictionary
        mDictionary = merged_dict(cDictionary)
        action_df[filepath] = mDictionary.values()


    total_df = pd.concat({k: pd.Series(v, dtype='float64') for k, v in total.items()}, axis=1)
    total_df = total_df.fillna(0)

    action_df = action_df.drop('Actions', 1)
    action_df = action_df.transpose()
    action_df.columns = mDictionary.keys()

    last_energy_total = []

    for key in total:
        temp_list = total[key]
        if len(temp_list) == 0:
            # Remove energy totals with 0
            action_df = action_df.drop(key)
            # last_energy_total.append(-1)
        else:
            last_energy_total.append(temp_list[-1 + len(temp_list)])
    
    action_df['Last Energy Total'] = last_energy_total

    total_df = total_df.transpose()

    return action_df, total_df
     

def linear_regression(action_df, total_df):
    cDictionary = count_dict()
    mDictionary = merged_dict(cDictionary)

    X = action_df[mDictionary.keys()].values
    y = action_df['Last Energy Total'].values
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    seabornInstance.distplot(action_df['Last Energy Total'])
    plt.show()


    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, mDictionary.keys(), columns=['Coefficient'])  
    print(coeff_df)

    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)

    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


if __name__ == "__main__":
    # concat_files()
    remove_empty_files_and_folders("Data")
    action_df, total_df = parse_dataframe()
    print(action_df)
    print(total_df)
    linear_regression(action_df, total_df)