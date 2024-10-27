import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os

folder_path = "../../datasets/twitter_spam_urls"

#create folders
os.system("mkdir " + folder_path + "/dataset")
os.system("mkdir " + folder_path + "/models")
os.system("mkdir " + folder_path + "/models/gbdt")
os.system("mkdir " + folder_path + "/models/gbdt_lse")

data = pd.read_csv("../../datasets/twitter_spam_urls/dataset/unnormalized_twitter_spam.train.csv").to_numpy()
X = data[:, 1:].astype(float)
y = data[:, 0].astype(int).astype(str)
y[y=='0'] = 0
y[y=='1'] = 1
y = y.reshape((y.shape[0], 1))

#split and save training set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=7, stratify=y)
train = np.concatenate((y_train, X_train), axis=1)
val = np.concatenate((y_val, X_val), axis=1)
train_df = pd.DataFrame(train)
val_df = pd.DataFrame(val)
train_df.to_csv(folder_path + "/dataset/training_set_normalized.csv", index=False, header=False)
val_df.to_csv(folder_path + "/dataset/validation_set_normalized.csv", index=False, header=False)

#load test dataset
data = pd.read_csv("../../datasets/twitter_spam_urls/dataset/unnormalized_twitter_spam.test.csv").to_numpy()
X = data[:, 1:].astype(float)
y = data[:, 0].astype(str)
y[y=='0'] = 0
y[y=='1'] = 1
y = y.reshape((y.shape[0], 1))

#split and save training set
test = np.concatenate((y, X), axis=1)
test_df = pd.DataFrame(test)
test_df.to_csv(folder_path + "/dataset/test_set_normalized.csv", index=False, header=False)
f = open(folder_path + "/dataset/test_set_normalized.csv", "r")
lines = f.read()
f.close()
f = open(folder_path + "/dataset/test_set_normalized.csv", "w")
f.write("# {} {}\n".format(test_df.shape[0], test_df.shape[1]-1))
f.write(lines)
f.close()