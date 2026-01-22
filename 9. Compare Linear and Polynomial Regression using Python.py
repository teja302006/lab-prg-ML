import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

X = np.array([1,2,3,4,5,6]).reshape(-1,1)
y = np.array([1,4,9,16,25,36])

lr = LinearRegression()
lr.fit(X, y)
y_pred_lr = lr.predict(X)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
pr = LinearRegression()
pr.fit(X_poly, y)
y_pred_pr = pr.predict(X_poly)

print("Linear Regression MSE:", mean_squared_error(y, y_pred_lr))
print("Polynomial Regression MSE:", mean_squared_error(y, y_pred_pr))
