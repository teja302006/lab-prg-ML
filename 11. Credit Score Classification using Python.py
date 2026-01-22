import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data = {
    "income":[15000,30000,45000,60000,75000,90000,105000,120000],
    "credit_history":[0,0,1,1,1,1,0,1],
    "loan_amount":[50000,40000,30000,20000,25000,15000,60000,10000],
    "score":[0,0,0,1,1,1,0,1]
}

df = pd.DataFrame(data)

X = df.drop("score",axis=1)
y = df["score"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.25,random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,pred))
print("Classification Report:")
print(classification_report(y_test,pred))
