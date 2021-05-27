import json
import glob
import os
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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

def merged_key():
    with open('merged_config.json') as f:
        data = json.load(f)
    return data


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

def find_action(data_dict, target):
    index = 0
    for key in data_dict:
        values = data_dict[key]
        index += 1
        for v in values:
            if v == target:
                return index
    return None

def parse_sequence_dataframe():
    total = {}
    action_sequence = defaultdict(list)
    cDictionary = count_dict()
    action_key = merged_key()

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
                                action_sequence[filepath].append(find_action(action_key, subitem))
                            if subitem == 'EnergyAnnualAnalysis':
                                # total[filepath].append(
                                #     item['EnergyAnnualAnalysis']['Solar Panels']['Total'])

                                # Update to the correct total energy value
                                total[filepath].append(
                                    item['EnergyAnnualAnalysis']['Net']["Total"])

                
    
                except Exception as e:
                    print("Error with ", file)
                    print(str(e))
        
    action_df = pd.concat({k: pd.Series(v, dtype='float64') for k, v in action_sequence.items()}, axis=1)
        # print(action_df)

    print(action_df.keys())

    total_df = pd.concat({k: pd.Series(v, dtype='float64') for k, v in total.items()}, axis=1)
    total_df = total_df.fillna(0)

    action_df = action_df.transpose()

    final_net_energy = []

    print(len(total))
    # pd.set_option("display.max_rows", None, "display.max_columns", None)

    # print((action_df))

    for key in action_sequence:
        temp_list = total[key]
        # print(temp_list)
        # print(type(key))
        # print(action_df.isin([key]).any())
        if len(temp_list) == 0:
            # Remove energy totals with 0
            # if key in action_df:
            # print("YES")
            action_df = action_df.drop(key)
            # final_net_energy.append(-1)
        else:
            final_net_energy.append(temp_list[-1 + len(temp_list)])
            # TODO: Change to last index of temp_list
            # final_net_energy.append(temp_list[0])
    
    # print(len(final_net_energy))
    action_df['Final Net Energy'] = final_net_energy

    # action_df['Final Net Energy'] = pd.Series(final_net_energy)

    # OPTIONAL: remove an action cateory
    # action_df = action_df.drop(['Window'], 1)

    total_df = total_df.transpose()
    print(action_df)
    return action_df, total_df

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
                                # total[filepath].append(
                                #     item['EnergyAnnualAnalysis']['Solar Panels']['Total'])

                                # Update to the correct total energy value
                                total[filepath].append(
                                    item['EnergyAnnualAnalysis']['Net']["Total"])

                
    
                except Exception as e:
                    print("Error with ", file)
                    print(str(e))

        # New merged dictionary
        mDictionary = merged_dict(cDictionary)
        action_df[filepath] = mDictionary.values()

    print(action_df.keys())

    total_df = pd.concat({k: pd.Series(v, dtype='float64') for k, v in total.items()}, axis=1)
    total_df = total_df.fillna(0)

    action_df = action_df.drop('Actions', 1)

    action_df = action_df.transpose()
    action_df.columns = mDictionary.keys()

    final_net_energy = []    
    for key in total:
        temp_list = total[key]
        if len(temp_list) == 0:
            # Remove energy totals with 0
            action_df = action_df.drop(key)
            # final_net_energy.append(-1)
        else:
            # final_net_energy.append(temp_list[-1 + len(temp_list)])
            final_net_energy.append(temp_list[0])
    
    action_df['Final Net Energy'] = final_net_energy

    # OPTIONAL: remove an action cateory
    # action_df = action_df.drop(['Window'], 1)

    total_df = total_df.transpose()

    return action_df, total_df
     

def linear_regression(action_df, total_df):
    cDictionary = count_dict()
    cols = list(merged_dict(cDictionary).keys())

    # Cuztomize font size for all plots
    plt.rcParams.update({'font.size': 18})

    # OPTIONAL: remove an action cateory
    # cols.remove('Window')

    X = action_df[cols].values
    y = action_df['Final Net Energy'].values
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    seabornInstance.distplot(action_df['Final Net Energy'])
    plt.ylabel("Density of Values", fontsize=18)
    plt.xlabel("Final Net Energy (kWh)", fontsize=18)
    plt.show()

    print(X.shape)
    print(y.shape)

    # Cuztomize font size for second plot
    plt.rcParams.update({'font.size': 15})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, cols, columns=['Coefficient'])  
    print(coeff_df)

    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)

    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    custom_axis = list(range(1, len(df['Actual']) + 1))
    old_axis = list(range(0, len(df['Actual']-1)))
    plt.xticks(old_axis, custom_axis)

    plt.xlabel("Randomly Selected Students", fontsize=15)
    plt.ylabel("Final Net Energy (kWh)", fontsize=15)
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def logistic_regression(action_df, total_df):
    cDictionary = count_dict()
    cols = list(merged_dict(cDictionary).keys())

    # Cuztomize font size for all plots
    plt.rcParams.update({'font.size': 18})

    # OPTIONAL: remove an action cateory
    # cols.remove('Window')

    X = action_df[cols].values
    y = action_df['Final Net Energy'].values
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    seabornInstance.distplot(action_df['Final Net Energy'])
    plt.ylabel("Density of Values", fontsize=18)
    plt.xlabel("Final Net Energy (kWh)", fontsize=18)
    plt.show()

    print(X.shape)
    print(y.shape)

    # Cuztomize font size for second plot
    plt.rcParams.update({'font.size': 15})

    print(action_df.describe())

    # Categorize final net energy for logistic regression
    for idx, row in action_df.iterrows():
        if  action_df.loc[idx,'Final Net Energy'] >= -5000 and action_df.loc[idx,'Final Net Energy'] <= 5000:
            action_df.loc[idx,'Final Net Energy'] = 1
        else:
            action_df.loc[idx,'Final Net Energy'] = 0

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LogisticRegression(solver='lbfgs', max_iter=10000)  
    y_train = y_train.astype('int')
    regressor.fit(X_train, y_train)
    # coeff_df = pd.DataFrame(regressor.coef_, cols, columns=['Coefficient'])  
    # print(coeff_df)

    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)

    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    custom_axis = list(range(1, len(df['Actual']) + 1))
    old_axis = list(range(0, len(df['Actual']-1)))
    plt.xticks(old_axis, custom_axis)

    plt.xlabel("Randomly Selected Students", fontsize=15)
    plt.ylabel("Final Net Energy (kWh)", fontsize=15)
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


if __name__ == "__main__":
    remove_empty_files_and_folders("Data")
    # action_df, total_df = parse_dataframe()
    # print(action_df)
    # linear_regression(action_df, total_df)
    # logistic_regression(action_df, total_df)
    parse_sequence_dataframe()