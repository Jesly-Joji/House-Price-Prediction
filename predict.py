#IMPORT STATEMENTS
import numpy as np
import pandas as pd
import seaborn as sns

#READ DATASET
df=pd.read_csv(r"C:\Users\jesly\OneDrive\Desktop\Projects\House Price Prediction\USA_Housing.csv")

#DEFINING X AND Y
X=df.iloc[:,:5]
Y=df["Price"]

#SPLITTING
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

#MODEL
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_train,Y_train)

#PREDICTION
pred=model.predict(X_test)

#EVALUATION
from sklearn.metrics import r2_score
r2_score(Y_test,pred)


a=[[80196.242251,6.675697,7.275193,3.17,48694.864144 ]]
print(model.predict(a))
#SAVE MODEL
import joblib
#joblib.dump(model,"saved_model.pkl")

