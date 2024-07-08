import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

'''
Data getting and Selecting the open - close and the high - low data
'''
data = quandl.get("NSE/TATAGLOBAL")
data['Open - Close'] = data['Open'] - data['Close']
data['High - Low'] = data['High'] - data['Low']
data = data.dropna()
X = data[['Open - Close', 'High - Low']]
# print(X.head())

#### BEGIN KNN Classification

'''
Storing the data where the next closing value was grater than the last closing value,  as +1,
and store the data where it wasnt as -1
'''
# Y = np.where(data['Close'].shift(-1)>data['Close'],1,-1)
# # print(Y)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=43)


# # Find the bast params for de knn classifier using gridsearch(hypermarameter optimization)
# params = {'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11,  12, 13, 14, 15]}
# knn = neighbors.KNeighborsClassifier()
# model = GridSearchCV(knn, params, cv=5)

# #fitting the model
# model.fit(X_train, Y_train)

# #Getting accuracy
# accuracy_train = accuracy_score(Y_train, model.predict(X_train))
# accuracy_test = accuracy_score(Y_test, model.predict(X_test))


# # Printing the ActualClass(Correct buy/sell action) and the PredictedClass(Predicted bu/sell by the model)
# predictions_classification = model.predict(X_test)
# actual_predicted_data = pd.DataFrame({'Actual Class':Y_test, 'Predicted Class':predictions_classification})
# # print(actual_predicted_data.head(10))

#### END KNN Classification


#### BEGIN KNN Regression

'''
Storing the data of the Closing value of the stocks
'''

Y = data['Close']
# print(Y)

X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X, Y, test_size=0.25, random_state=43)

params_reg = {'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11,  12, 13, 14, 15]}
knn_reg = neighbors.KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params_reg, cv=5)

model_reg.fit(X_train_reg, Y_train_reg)

predictions_reg = model_reg.predict(X_test_reg)

# print(predictions_reg)

# rmse = root mean-square error (acurracy mesurement)
rmse = np.sqrt(np.mean(np.power((np.array(Y_test_reg)-np.array(predictions_reg)), 2)))
# print(rmse)

actual_predicted_data_reg = pd.DataFrame({'Actual Class':Y_test_reg, 'Predicted Class':predictions_reg})
# print(actual_predicted_data_reg.head(10))
#### END KNN Regression

'''
Plotting the results
'''

# print(f"Train_data Accuracy: {accuracy_train}")
# print(f"Test_data Accuracy: {accuracy_test}")

# plt.figure(figsize=(16,8))
# plt.plot()
# plt.show()


'''
Regression X Classification

    Regression: Regression Predictive Modeling is the task to aproximate a value(int, float) 
    in quantity mesurement, exemple estimate the price of something looking how the price of it
    changes over time, the result prediction would be a price value. The error is the RMSE(root
    mean-squared error)


    Classification: Classification Predictive Modeling is the task to categorize a class of objects
    in quality mesurement, exemple classificate a email if it is spam or not-spam, the result
    prediction would be a label value(wich category the class belongs). The error is the accurancy
    percentage, how many times the model lebel the data correct x the times that the model label 
    incorrectly.

'''