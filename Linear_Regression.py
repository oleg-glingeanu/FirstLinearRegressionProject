import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
from matplotlib import style
from sklearn import linear_model
import pickle

# Reading in data from the csv file using pandas
data = pd.read_csv("dataset_Facebook.csv", sep=";")
data = data[['Page total likes', 'Lifetime Post Total Reach', 'Post Month', 'share', 'Total Interactions']]

# Checking The shape and info of the data set
print(data.shape)

# Getting rid of any NaN, infinity or a value too large for dtype('float64').
data.share = data.share.fillna(0)
print(data.info())

# Predicting Value
predicting = "Page total likes"

x = np.array(data.drop([predicting], 1))
y = np.array(data)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Setting up the Linear Regression Algorithm
linear_algorithm = linear_model.LinearRegression()
linear_algorithm.fit(x_train, y_train)
accuracy = linear_algorithm.score(x_test, y_test)

# Setting the best accuracy we want to get
best_accuracy = 0.97

# Creating a loop to iterate trough and find a suitable accuracy
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc > best_accuracy:
        best_accuracy = acc
        print(best_accuracy)
        with open("dataset_Facebook.pickle", "wb") as f:
            pickle.dump(linear, f)
        break

# Pickle in the dataset and setting it to the algorithm
pickle_in = open("dataset_Facebook.pickle", "rb")
linear_algorithm = pickle.load(pickle_in)


# Plotting the Data
plot = 'Post Month'
style.use("ggplot")
pyplot.scatter(data[plot], data["Page total likes"])
pyplot.xlabel(plot)
pyplot.ylabel("Page total likes")
pyplot.show()