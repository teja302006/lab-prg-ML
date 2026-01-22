import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data = {
    "area":[800,1000,1200,1500,1800,2000,2200,2500],
    "bedrooms":[1,2,2,3,3,3,4,4],
    "bathrooms":[1,1,2,2,2,3,3,3],
    "price":[3000000,4000000,5200000,6500000,7500000,9000000,10500000,12000000]
}

df = pd.DataFrame(data)

X = df.drop("price",axis=1)
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.25,random_state=2)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("MSE:",mean_squared_error(y_test,pred))
print("R2 Score:",r2_score(y_test,pred))
print("Predicted House Prices:",pred)
