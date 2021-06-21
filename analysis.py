import json
import glob
import os
import re
import math
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict

import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics

"""
Process the action count dictionary based on the initial counts
(See action_config.txt)
"""
def count_dict():
    res = {}
    # Initialize the action count dictionary with values
    with open("action_config.txt") as f:
        for line in f:
            split = line.rsplit(' ', 1)
            res[split[0]] = int(split[1])
    return res

"""
Creates a merged dictionary by matching counts to actions and their respective categories

Parameters
----------
cDictionary : dictionary
    Dictionary of actions with their respective counts
"""
def merged_dict(cDictionary):
    res = {}
    with open('merged_config.json') as f:
        data = json.load(f)
        for key in data:
            res[key] = 0
            for val in data[key]:
                # Add counts from the count dictionary to the merged dictionary
                res[key] += cDictionary[val]
    return res

"""
Load JSON from merged_config.json with actions and their respective categories
"""
def merged_key():
    with open('merged_config.json') as f:
        data = json.load(f)
    return data

"""
Removes empty files and folders
Resets End_Plots, Linear_Plots, and Logistic_Plots for every run of code

Parameters
----------
dir_path : string
    Parent directory path to begin removing empty files and folders
"""
def remove_empty_files_and_folders(dir_path) -> None:
    # Remove and recreate End_Plots, Linear_Plots, and Logistic_Plots
    shutil.rmtree('Logistic_Plots', ignore_errors=True)
    os.mkdir('Logistic_Plots')

    shutil.rmtree('Linear_Plots', ignore_errors=True)
    os.mkdir('Linear_Plots')

    shutil.rmtree('End_Plots', ignore_errors=True)
    os.mkdir('End_Plots')

    # Walk down parent directory and delete any empty files or folders
    for root, dirnames, files in os.walk(dir_path, topdown=False):
        for f in files:
            full_name = os.path.join(root, f)
            if os.path.getsize(full_name) == 0:
                os.remove(full_name)

        for dirname in dirnames:
            full_path = os.path.realpath(os.path.join(root, dirname))
            if not os.listdir(full_path):
                os.rmdir(full_path)

"""
Finds the index of an action in merged dictionary 
(see merged_config.json)

Parameters
----------
data_dict : dataframe
    Dataframe of actions and their respective categories
target : string
    Name of action to be found
"""
def find_action(data_dict, target):
    index = 0
    for key in data_dict:
        values = data_dict[key]
        index += 1
        for v in values:
            if v == target:
                # If found return index of action
                return index
    # Else return None
    return None

"""
Process the action_df (action dataframe) and total_df (final net energy dataframe) with a percentage of the action sequence
"""
def parse_sequence_dataframe(percent):
     # Initialize types
    total = {}
    action_sequence = defaultdict(list)
    cDictionary = count_dict()
    action_key = merged_key()

    # Scan all subfolders in /Data and add paths to filepaths
    subfolders = [ f.path for f in os.scandir("Data") if f.is_dir() ]
    filepaths = []
    for sub in subfolders:
        f = [f.path for f in os.scandir(sub) if f.is_dir()]
        filepaths.append(f[0])

    # Load the JSON from every folder and fill in actions and final net energies
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
                                # Append the action number to sequence if it's a subitem of cDictionary
                                action_sequence[filepath].append(find_action(action_key, subitem))
                            if subitem == 'EnergyAnnualAnalysis':
                                # Update to the correct total energy value
                                total[filepath].append(
                                    item['EnergyAnnualAnalysis']['Net']["Total"])

                except Exception as e:
                    print("Error with ", file)
                    print(str(e))

    # Truncate the action sequence based on percentage parameter
    for k in action_sequence:
        new_size = math.ceil(len(action_sequence[k])*percent)
        del action_sequence[k][new_size:]

    # Translate actions dictionary into a dataframe
    action_df = pd.concat({k: pd.Series(v, dtype='float64') for k, v in action_sequence.items()}, axis=1)
    action_df = action_df.fillna(-1)

    # Translate final net energies dictionary into a dataframe
    total_df = pd.concat({k: pd.Series(v, dtype='float64') for k, v in total.items()}, axis=1)
    total_df = total_df.fillna(0)

    action_df = action_df.transpose()

    final_net_energy = []
    for key in action_sequence:
        temp_list = total[key]
        if len(temp_list) == 0:
            # Remove energy totals with 0
            action_df = action_df.drop(key)
        else:
            # Append the last energy total in the list
            final_net_energy.append(temp_list[-1 + len(temp_list)])
    
    action_df['Final Net Energy'] = final_net_energy

    total_df = total_df.transpose()
    
    return action_df, total_df

"""
Process the action_df (action dataframe) and total_df (final net energy dataframe)
"""
def parse_dataframe():
    # Initialize types
    total = {}
    cDictionary = count_dict()
    mDictionary = merged_dict(cDictionary)
    action_df = pd.DataFrame(mDictionary.keys(),
                                 columns=['Actions'])

    # Scan all subfolders in /Data and add paths to filepaths
    subfolders = [ f.path for f in os.scandir("Data") if f.is_dir() ]
    filepaths = []
    for sub in subfolders:
        f = [f.path for f in os.scandir(sub) if f.is_dir()]
        filepaths.append(f[0])

    # Load the JSON from every folder and fill in actions and final net energies
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
                                # Count item if it is a subitem of cDictionary
                                cDictionary[subitem] = cDictionary[subitem] + 1
                            if subitem == 'EnergyAnnualAnalysis':
                                # Update with every final net energy value
                                total[filepath].append(
                                    item['EnergyAnnualAnalysis']['Net']["Total"])

                except Exception as e:
                    print("Error with ", file)
                    print(str(e))

        # New merged dictionary
        mDictionary = merged_dict(cDictionary)
        action_df[filepath] = mDictionary.values()

    # Translate final net energies dictionary into a dataframe
    total_df = pd.concat({k: pd.Series(v, dtype='float64') for k, v in total.items()}, axis=1)
    total_df = total_df.fillna(0)

    # Drop the 'Actions' column
    action_df = action_df.drop('Actions', 1)

    # Transpose dataframe and use keys with action categories
    action_df = action_df.transpose()
    action_df.columns = mDictionary.keys()

    # Remove students with no final net enery values
    final_net_energy = []    
    for key in total:
        temp_list = total[key]
        if len(temp_list) == 0:
            # Remove energy totals with 0
            action_df = action_df.drop(key)
        else:
            # Append the last energy total in the list
            final_net_energy.append(temp_list[-1 + len(temp_list)])
    
    action_df['Final Net Energy'] = final_net_energy

    # OPTIONAL: remove an action cateory
    # action_df = action_df.drop(['Window'], 1)

    # Transpose dataframe with final net energy values
    total_df = total_df.transpose()

    return action_df, total_df
     
"""
Performs linear regression given the parameters

Parameters
----------
action_df : dataframe
    Dataframe of actions
total_df : dataframe
    Dataframe of final net energy totals
is_seq : int, 0 or 1
    1 if linear regression predicting sequential data, else 0
index : int
    Numbered index to save unique graphs to each folder
"""
def linear_regression(action_df, total_df, is_seq, index):
    # Get the count dictionary to access action categories for columns
    cDictionary = count_dict()
    # If the prediction is sequential, use numbers as columns, else use action categories as columns
    if is_seq == 1:
        cols = list(total_df.keys())
    else:
        cols = list(merged_dict(cDictionary).keys())

    # Cuztomize font size for all plots
    plt.rcParams.update({'font.size': 18})

    # OPTIONAL: remove an action cateory
    # cols.remove('Window')

    # Create and save plot of density of values fed in for linear regression
    X = action_df[cols].values
    y = action_df['Final Net Energy'].values
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    seabornInstance.distplot(action_df['Final Net Energy'])
    plt.ylabel("Density of Values", fontsize=18)
    plt.xlabel("Final Net Energy (kWh)", fontsize=18)
    # Save figure to correct folder
    saved_name = '%s%d' % ('Linear_Plots/Density of Values x Final Net Energy (kWh)', index)
    plt.savefig(saved_name)

    # Create and save plot for histogram of final net energies fed in for linear regression
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    n_bins = 30
    plt.figure(figsize=(15,10))
    # We can set the number of bins with the `bins` kwarg
    plt.hist(y, bins=n_bins)
    plt.ylabel("Number of Values", fontsize=18)
    plt.xlabel("Final Net Energy (kWh)", fontsize=18)
    plt.title("Histogram of Values x Final Net Energy (kWh)")
    saved_name = '%s%d' % ('Linear_Plots/Histogram of Values x Final Net Energy (kWh)', index)
    plt.savefig(saved_name)

    # Print column numbers to see if dataframes align for prediction
    print(X.shape)
    print(y.shape)

    # Cuztomize font size for second plot
    plt.rcParams.update({'font.size': 15})

    # Split training set, 80% to train, 20% to predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    # Print regressor's coefficients
    coeff_df = pd.DataFrame(regressor.coef_, cols, columns=['Coefficient'])  
    print(coeff_df)

    # Predict the value
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)

    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    # Plot graph of actual and predicted values
    custom_axis = list(range(1, len(df['Actual']) + 1))
    old_axis = list(range(0, len(df['Actual']-1)))
    plt.xticks(old_axis, custom_axis)

    plt.xlabel("Randomly Selected Students", fontsize=15)
    plt.ylabel("Final Net Energy (kWh)", fontsize=15)
    # Save figure to correct folder
    saved_name = '%s%d' % ('Linear_Plots/Randomly Selected Students x Final Net Energy (kWh)', index)
    plt.savefig(saved_name)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

"""
Performs logistic regression given the parameters

Parameters
----------
action_df : dataframe
    Dataframe of actions
total:df : dataframe
    Dataframe of final net energy totals
is_seq : int, 0 or 1
    1 if logistic regression predicting sequential data, else 0
index : int
    Numbered index to save unique graphs to each folder
lower_lim : int
    Lower limit for logistic interval that will be categorized as 1
upper_lim : int
    Upper limit for logistic interval that will be categorized as 1
"""
def logistic_regression(action_df, total_df, is_seq, index, lower_lim, upper_lim):
    # Get the count dictionary to access action categories for columns
    cDictionary = count_dict()
    # If the prediction is sequential, use numbers as columns, else use action categories as columns
    if is_seq == 1:
        cols = list(total_df.keys())
    else:
        cols = list(merged_dict(cDictionary).keys())

    # Cuztomize font size for all plots
    plt.rcParams.update({'font.size': 18})

    # OPTIONAL: remove an action cateory
    # cols.remove('Window')

    # Create and save plot of density of values fed in for logistic regression
    X = action_df[cols].values
    y = action_df['Final Net Energy'].values
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    seabornInstance.distplot(action_df['Final Net Energy'])
    plt.ylabel("Density of Values", fontsize=18)
    plt.xlabel("Final Net Energy (kWh)", fontsize=18)
    # Save figure to correct folder
    saved_name = '%s%d' % ('Logistic_Plots/Density of Values x Final Net Energy (kWh)', index)
    plt.savefig(saved_name)

    # Print column numbers to see if dataframes align for prediction
    print(X.shape)
    print(y.shape)

    # Cuztomize font size for second plot
    plt.rcParams.update({'font.size': 15})

    # Print statistics for the actions dataframe
    # print(action_df.describe())

    # Categorize final net energy for logistic regression
    for idx, row in action_df.iterrows():
        if  action_df.loc[idx,'Final Net Energy'] >= lower_lim and action_df.loc[idx,'Final Net Energy'] <= upper_lim:
            action_df.loc[idx,'Final Net Energy'] = 1
        else:
            action_df.loc[idx,'Final Net Energy'] = 0
    

    # Split training set, 80% to train, 20% to predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LogisticRegression(solver='lbfgs', max_iter=10000)  
    y_train = y_train.astype('int')
    regressor.fit(X_train, y_train)
    # Print regressor's coefficients
    # coeff_df = pd.DataFrame(regressor.coef_, cols, columns=['Coefficient'])  
    # print(coeff_df)

    # Predict the value
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print(df1)

    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    # Plot graph of actual and predicted values
    custom_axis = list(range(1, len(df['Actual']) + 1))
    old_axis = list(range(0, len(df['Actual']-1)))
    plt.xticks(old_axis, custom_axis)

    plt.xlabel("Randomly Selected Students", fontsize=15)
    plt.ylabel("Final Net Energy in Range (0 = false, 1 = true)", fontsize=15)
    # Save figure to correct folder
    saved_name = '%s%d' % ('Logistic_Plots/Randomly Selected Students x Final Net Energy in Range (0 = false, 1 = true)', index)
    plt.savefig(saved_name)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    plt.clf()

    # TODO: Check this -> Still having constant accuracies
    correct = 0
    # Count how many accurate predictions
    for test, pred in zip(y_test, y_pred):
        if test == pred:
            correct = correct + 1

    # Create bar graph of correct and incorrect predictions
    accuracies = [correct, len(y_test) - correct]
    objects = ('Correct', 'Incorrect')
    y_pos = np.arange(len(objects))
    plt.barh(y_pos, accuracies, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Number of Predictions')
    plt.title('Prediction Accuracy')
    saved_name = '%s%d' % ('Logistic_Plots/Prediction Accuracy Bar Graph', index)
    plt.savefig(saved_name)
    plt.clf()
    
    # Calculate percentage of accurate predictions
    score = correct/len(y_test)
    return score

if __name__ == "__main__":
    # Remove and clean up files
    remove_empty_files_and_folders("Data")
    # Create graphs without percentages, fix logistic interval at -5000 to 5000
    action_df, total_df = parse_dataframe()
    print(action_df)
    linear_regression(action_df, total_df, 0, 0)
    logistic_regression(action_df, total_df, 0, 0, -5000, 5000)

    # Create scatterplot of accuracy based on percentage of data
    percentages = [.1, .2, .3, .35, .4, .45, .5, .6, .7, .8, .9, 1]
    predictions = []
    index = 0
    # Loop through percentages perform regression
    for percent in percentages:
        action_df_seq, total_df_seq = parse_sequence_dataframe(percent)
        linear_regression(action_df_seq, total_df_seq, 1, index)
        index = index + 1
        # Fix logistic interval from -5000 to 5000
        predictions.append(logistic_regression(action_df_seq, total_df_seq, 1, index, -5000, 5000))
    
    # Print predictions and create labels for axis
    print(predictions)
    np_percentages = np.array(percentages)
    np_predictions = np.array(predictions)
    predictions = np_predictions * 100
    percentages = np_percentages * 100
    
    # Plot graph
    plt.scatter(percentages, predictions, c='r')
    plt.title("Accuracy of Model for Percentage of Action Sequence")
    plt.xlabel("Percentage of Action Sequence (%)", fontsize=15)
    plt.ylabel("Accuracy of Model (%)", fontsize=15)
    plt.savefig('End_Plots/Accuracy of Model for Percentage of Action Sequence')

    # Create scatterplot with different ranges of final net energy
    predictions = []
    ranges = []
    index = 0
    lower_lim = 0

    # Loop through ranges, fix the percentage of data at 40%
    for x in range(0, 10):
        action_df_seq, total_df_seq = parse_sequence_dataframe(0.4)
        linear_regression(action_df_seq, total_df_seq, 1, index)
        # Create unique index for graph name
        index = index + 1
        lower_lim = lower_lim - 1000
        upper_lim = -1 * lower_lim
        ranges.append(str(lower_lim) + " - " + str(upper_lim))
        # Add accuracy to final list
        predictions.append(logistic_regression(action_df_seq, total_df_seq, 1, index, lower_lim, upper_lim))
    
    # Plot graph
    plt.scatter(ranges, predictions, c='r')
    plt.xticks(rotation = 45, fontsize = 10)
    plt.title("Accuracy of Model for Changed Logistic Range")
    plt.xlabel("Range of Action Sequence", fontsize=15)
    plt.ylabel("Accuracy of Model (%)", fontsize=15)
    plt.savefig('End_Plots/Accuracy of Model for Changed Logistic Range')