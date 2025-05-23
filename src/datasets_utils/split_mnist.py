import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import argparse
import joblib
from sklearn.utils import resample

#Provide the classes of MNIST to consider
parser = argparse.ArgumentParser()
parser.add_argument('label0', type=str)
parser.add_argument('label1', type=str)
args = parser.parse_args()
label0 = args.label0
label1 = args.label1

folder_path = "../../datasets/mnist{}{}".format(args.label0, args.label1)

#create folders
os.system("mkdir ../../datasets")
os.system("mkdir " + folder_path)
os.system("mkdir " + folder_path + "/dataset")
os.system("mkdir " + folder_path + "/models")
os.system("mkdir " + folder_path + "/models/gbdt")
os.system("mkdir " + folder_path + "/models/gbdt_lse")

#load dataset
X,y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.to_numpy().astype(np.float64)
y = y.to_numpy().astype(str)

#filter instances x labels
X = X[np.isin(y, [label0, label1])]
y = y[np.isin(y, [label0, label1])]
y[y==label0] = 0
y[y==label1] = 1

#Scale in [0-1]
y = y.reshape((y.shape[0], 1))
X = np.nan_to_num(X)
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#Save splittings
dataset = np.concatenate((y, X), axis=1)
dataset_df = pd.DataFrame(dataset)
dataset_df.to_csv(folder_path + "/dataset/dataset_normalized.csv", index=False, header=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=7, stratify=y)
test = np.concatenate((y_test, X_test), axis=1)
test_df = pd.DataFrame(test)
test_df.to_csv(folder_path + "/dataset/test_set_normalized.csv", index=False, header=False)
f = open(folder_path + "/dataset/test_set_normalized.csv", "r")
lines = f.read()
f.close()
f = open(folder_path + "/dataset/test_set_normalized.csv", "w")
f.write("# {} {}\n".format(test_df.shape[0], test_df.shape[1]-1))
f.write(lines)
f.close()

#data for gridsearch
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=7, stratify=y_train)
train = np.concatenate((y_train, X_train), axis=1)
val = np.concatenate((y_val, X_val), axis=1)
train_df = pd.DataFrame(train)
val_df = pd.DataFrame(val)
train_df.to_csv(folder_path + "/dataset/training_set_normalized.csv", index=False, header=False)
val_df.to_csv(folder_path + "/dataset/validation_set_normalized.csv", index=False, header=False)

#save subset of instances for expensive verification
test_df = resample(test_df, replace=False, n_samples = 500, random_state=7, stratify=test_df.iloc[:, 0])
test_df.to_csv(folder_path + "/dataset/test_set_normalized_500.csv", index=False, header=False)
f = open(folder_path + "/dataset/test_set_normalized_500.csv", "r")
lines = f.read()
f.close()
f = open(folder_path + "/dataset/test_set_normalized_500.csv", "w")
f.write("# {} {}\n".format(test_df.shape[0], test_df.shape[1]-1))
f.write(lines)
f.close()
