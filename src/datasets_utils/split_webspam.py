from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.utils import resample

folder_path = "../../datasets/webspam/dataset"

#create folders
os.system("mkdir ../../datasets")
os.system("mkdir " + folder_path)
os.system("mkdir " + folder_path + "/dataset")
os.system("mkdir " + folder_path + "/models")
os.system("mkdir " + folder_path + "/models/gbdt")
os.system("mkdir " + folder_path + "/models/gbdt_lse")
os.system("wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.xz")
os.system("unxz webspam_wc_normalized_unigram.svm.xz")
os.system("rm webspam_wc_normalized_unigram.svm.xz")
os.system("mv webspam_wc_normalized_unigram.svm ../../datasets/webspam/dataset/")
os.system("python prepare_webspam.py")

#load dataset
data = pd.read_csv(folder_path + "/dataset/" + "webspam.csv", delimiter=",", header=None)

data.iloc[:, 0] = data.iloc[:, 0].astype(int).astype(str).replace("-1", "0")
data.iloc[:, 1:] = data.iloc[:, 1:].astype(float).fillna(0)

#Scale in [0-1]
data.iloc[:, 1:] = (data.iloc[:, 1:] - data.iloc[:, 1:].min())/(data.iloc[:, 1:].max()-data.iloc[:, 1:].min())
data = data.fillna(0)

#Save splittings
data_train, data_test = train_test_split(data, test_size = 0.3, random_state=7, stratify=data.iloc[:, 0])
data_test.to_csv(folder_path + "/dataset/test_set_normalized.csv", index=False, header=False)
f = open(folder_path + "/dataset/test_set_normalized.csv", "r")
lines = f.read()
f.close()
f = open(folder_path + "/dataset/test_set_normalized.csv", "w")
f.write("# {} {}\n".format(data_test.shape[0], data_test.shape[1]-1))
f.write(lines)
f.close()

#data for gridsearch
train, val = train_test_split(data_train, test_size = 0.2, random_state=7, stratify=data_train.iloc[:, 0])
train.to_csv(folder_path + "/dataset/training_set_normalized.csv", index=False, header=False)
val.to_csv(folder_path + "/dataset/validation_set_normalized.csv", index=False, header=False)

#save subset of instances for expensive verification
data_test = resample(data_test, replace=False, n_samples = 500, random_state=7, stratify=data_test.iloc[:, 0])
data_test.to_csv(folder_path + "/dataset/test_set_normalized_500.csv", index=False, header=False)
f = open(folder_path + "/dataset/test_set_normalized_500.csv", "r")
lines = f.read()
f.close()
f = open(folder_path + "/dataset/test_set_normalized_500.csv", "w")
f.write("# {} {}\n".format(data_test.shape[0], data_test.shape[1]-1))
f.write(lines)
f.close()



