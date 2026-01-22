import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data = {
    "year":[2015,2016,2017,2018,2019,2020,2021,2022],
    "km_driven":[60000,50000,45000,30000,25000,20000,15000,10000],
    "fuel_type":[1,1,0,1,0,1,0,1],
    "price":[400000,450000,500000,650000,700000,750000,820000,900000]
}

df = pd.DataFrame(data)

X = df.drop("price",axis=1)
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.25,random_state=1)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("MSE:",mean_squared_error(y_test,pred))
print("R2 Score:",r2_score(y_test,pred))
print("Predicted Prices:",pred)
