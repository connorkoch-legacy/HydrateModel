import pandas as pd
import requests
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10, mnist
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.svm import SVC # This is the SVM classifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



# Read flowloop data from csv and convert values to float
data = pd.read_csv("flowloopData.csv")
data = data.applymap(float)

#//////////////////////////////////////////////

#Do we ever fit data to just the features output by the feature selection methods?
#Did you standardize the data in the field matrices?

#Create X and Y from data frame
X = np.asarray(data)
Y = X[:, 8]
X = X[:, :8]
Y = Y.astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

print(f"All Data:        {len(X)} points")
print(f"Training data:   {len(X_train)} points")
print(f"Testing data:    {len(X_test)} points")

#standardize data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

"""############## This section checks neural network, simple sequetial model ##############"""
print("############## Sequential Model ##############")
def build_model(X_train):
  model = Sequential([
    #Flatten(input_dim=8),
    Dense(100, input_dim=8, activation='relu'),
    Dropout(0.25),
    Dense(100, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(50, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(1,activation='sigmoid')
  ])

  optimizer = tf.train.AdamOptimizer()

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

simple_nn_model = build_model(X_train)
simple_nn_model.summary()

simple_nn_model.fit(X_train, Y_train, epochs=50)
simple_nn_model.summary()

print("Model evaluated on the test data: ")
print(simple_nn_model.metrics_names)
print(simple_nn_model.evaluate(X_test, Y_test))
print()
print()


#additional test vals for the model
# X_field=np.array([[0.1,	4.18,	750,	41,	0.772,	15, 0.03, 135],
# [0.1,	4.18,	750,	41,	0.772,	15, 0.03, 153],
# [0.23,	3.93,	1000,	41,	0.772,	15, 0.15, 216],
# [0.38,	9.92,	250,	41,	0.772,	15, 0.17, 307],
# [0.54,	9.92,	250,	41,	0.772,	15, 0.16, 211],
# [0.45,	3.92,	250,	41,	0.772,	15, 0.11, 208],
# [0.45,	3.92,	1021,	41,	0.772,	15, 0.01, 73.5],
# [0.45,	3.92,	1021,	41,	0.772,	15, 0.09, 97.8],
# [0.45,	3.92,	1077,	41,	0.772,	15, 0.09, 115],
# [0.45,	3.92,	1706,	41,	0.772,	15, 0.14, 107],
# [0.45,	3.92,	1706,	41,	0.772,	15, 0.14, 115],
# [0.6,	3.93,	1800,	41,	0.772,	15, 0.3, 210],
# [0.42,	4.18,	1100,	41,	0.772,	15, 0.19, 178]])
#
# X_field = (X_field - mean) / std
# Y_field = simple_nn_model.predict(X_field)
# #print(Y_field)
#
# X_field=np.array([[0.1,	4.18,	750,	41,	0.772,	15, 0.050, 135],
# [0.1,	4.18,	750,	41,	0.772,	15, 0.050, 153],
# [0.23,	3.93,	1000,	41,	0.772,	15, 0.084, 216],
# [0.38,	9.92,	250,	41,	0.772,	15, 0.12, 307],
# [0.54,	9.92,	250,	41,	0.772,	15, 0.132, 211],
# [0.45,	3.92,	250,	41,	0.772,	15, 0.077, 208],
# [0.45,	3.92,	1021,	41,	0.772,	15, 0.079, 73.5],
# [0.45,	3.92,	1021,	41,	0.772,	15, 0.079, 97.8],
# [0.45,	3.92,	1077,	41,	0.772,	15, 0.087, 115],
# [0.45,	3.92,	1706,	41,	0.772,	15, 0.094, 107],
# [0.45,	3.92,	1706,	41,	0.772,	15, 0.094, 115],
# [0.6,	3.93,	1800,	41,	0.772,	15, 0.211, 210],
# [0.42,	4.18,	1100,	41,	0.772,	15, 0.136, 178]])
#
# X_field = (X_field - mean) / std
# Y_field= simple_nn_model.predict(X_field)
# print(Y_field)

"""This section checks all the SVC models, linear, polynomial and rbf"""

print("############## SVC Model ##############")

params = {
    "C": [1e3, 5e3, 1e4, 2e4, 5e4],
    "random_state": [0]
}

svc = SVC(gamma="scale")
clf = GridSearchCV(svc, params, scoring="accuracy", cv=5)

clf.fit(X_train, Y_train)

GridSearchCVScore = clf.score(X_test, Y_test)
print(GridSearchCVScore)

clf_linear=SVC(kernel='linear')
clf_linear.fit(X_train, Y_train)

linearRegressScore=clf_linear.score(X_test,Y_test)
print("SVC linear", linearRegressScore)

clf_poly=SVC(kernel='poly',degree=3)
clf_poly.fit(X_train, Y_train)
polyRegressScore=clf_poly.score(X_test,Y_test)
print("SVC polynomial ", polyRegressScore)

clf_rbf=SVC(kernel='rbf',gamma=0.7)
clf_rbf.fit(X_train, Y_train)
rbfRegressScore=clf_rbf.score(X_test,Y_test)
print("SVC rbf", rbfRegressScore)

"""This section checks feature selection"""

mi_transformer=SelectKBest(mutual_info_regression,k=4)
#mi_X_train=mi_transformer.fit_transform(X_train,Y_train)
mi_X=mi_transformer.fit_transform(X,Y)
#print(mi_X_train.shape)

features=["water cut", "liquid velocity", "GOR", "Viscosity","Sp. Gravity", "IFT", "HVF", "Fa"]
print("KBest Selection")
for feature, importance in zip(features, mi_transformer.scores_):
  print(f"The MI score for {feature} is {importance}")
print()

tree=ExtraTreesClassifier(n_estimators=50)
tree.fit(X, Y)
print("ExtraTrees Selection")
for feature, importance in zip(features,tree.feature_importances_):
  print(f"The MI score for {feature} is {importance}")
print()
print()


"""############## This section does regression for Dp ##############"""
print("############## Regression for pressure drop ##############")

X = np.asarray(data)
Y = X[:, 10]
XX = X[:, 1:8]

X_train, X_test, Y_train, Y_test = train_test_split(XX, Y, random_state=0, test_size=0.2)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

def build_regression_model(X_train):
  model = Sequential([
    #Flatten(input_dim=8),
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1)
  ])

  #add dropout layer
  #optimizer = tf.train.RMSPropOptimizer(0.001)
  optimizer='adam'
  model.compile(optimizer=optimizer, loss='mae', metrics=['mae', "mse"])
  return model

simple_nn_model = build_regression_model(X_train)
simple_nn_model.summary()

simple_nn_model.fit(X_train, Y_train, epochs=60)
simple_nn_model.summary()

print("Model evaluated on the test data: ")
print(simple_nn_model.metrics_names)
print(simple_nn_model.evaluate(X_test, Y_test))

Dp_predictions = simple_nn_model.predict(X_test).flatten()

R2 = r2_score(Y_test, Dp_predictions)
print("r^2: ", R2)
print()
print()


# plt.scatter(Y_test, Dp_predictions)
# plt.xlabel('Measured Dp', fontsize='large')
# plt.ylabel('Predictions', fontsize='large')
# #plt.axis('equal')
# #plt.xlim(plt.xlim())
# #plt.ylim(plt.ylim())
# #_ = plt.plot([0, 200], [0, 200])
# plt.show()

"""############## This section investigates regression of hydrate growth ##############"""
print("############## Regression for hydrate growth ##############")

Features=data.iloc[:,[0,1,2,3,4,5,9]]
HVF=data.iloc[:,6]
X = np.asarray(Features)
Y= np.asarray(HVF)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

simple_nn_model = build_regression_model(X_train)
simple_nn_model.summary()

simple_nn_model.fit(X_train, Y_train, epochs=60)
simple_nn_model.summary()

print("Model evaluated on the test data: ")
print(simple_nn_model.metrics_names)
print(simple_nn_model.evaluate(X_test, Y_test))

HVF_predictions = simple_nn_model.predict(X_test).flatten()

R2 = r2_score(Y_test, HVF_predictions)
print("r^2: ", R2)
print()
print()

#print("HVF prediction: ", HVF_predictions)

# plt.scatter(Y_test, HVF_predictions)
# plt.xlabel('Exp HVF')
# plt.ylabel('Predictions')
# plt.axis('equal')
# plt.xlim(plt.xlim())
# plt.ylim(plt.ylim())
# _ = plt.plot([0, 1], [0, 1])
#
# X_field=np.array([[0.1,	4.18,	750,	41,	0.772,	15, 15],
# [0.1,	4.18,	750,	41,	0.772,	15, 15],
# [0.23,	3.93,	1000,	41,	0.772,	15, 30],
# [0.38,	9.92,	250,	41,	0.772,	15, 30],
# [0.54,	9.92,	250,	41,	0.772,	15, 30],
# [0.45,	3.92,	250,	41,	0.772,	15, 30],
# [0.45,	3.92,	1021,	41,	0.772,	15, 20],
# [0.45,	3.92,	1021,	41,	0.772,	15, 20],
# [0.45,	3.92,	1077,	41,	0.772,	15, 25],
# [0.45,	3.92,	1706,	41,	0.772,	15, 25],
# [0.45,	3.92,	1706,	41,	0.772,	15, 25],
# [0.6,	3.93,	1800,	41,	0.772,	15, 70],
# [0.42,	4.18,	1100,	41,	0.772,	15, 60]])
#
# X_field.shape
# X_field = (X_field - mean) / std
# HVF_field= simple_nn_model.predict(X_field)
# print(HVF_field)
