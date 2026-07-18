import sklearn.datasets as D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

iris_data = D.load_iris(return_X_y=True)

print(type(iris_data))

X, y = iris_data
print('X type: {}, shape: {}'.format(type(X), X.shape))
print('y type: {}, shape: {}'.format(type(y), y.shape))

X_train, X_test, y_train, y_test = train_test_split(*iris_data, test_size=0.25)

print('X_train shape: {}, X_test shape: {}'.format(X_train.shape, X_test.shape))
print('y_train shape: {}, y_test shape: {}'.format(y_train.shape, y_test.shape))

lr = LinearRegression()
lr.fit(X_train, y_train)
coef = lr.coef_
inte = lr.intercept_

print('coef: {}'.format(coef))
print('inte: {}'.format(inte))

# coef: [-0.09679056 -0.08714976  0.21955846  0.61844216]
# inte: 0.277739727533875

y_pred = lr.predict(X_test)  # float64
y_pred = np.abs(y_pred.reshape(-1, 1) - np.array([0, 1, 2]).reshape(1, 3)).argmin(axis=1)

cls_rep = classification_report(y_test, y_pred)
print(cls_rep)

#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        11
#            1       1.00      0.94      0.97        18
#            2       0.90      1.00      0.95         9

#     accuracy                           0.97        38
#    macro avg       0.97      0.98      0.97        38
# weighted avg       0.98      0.97      0.97        38
