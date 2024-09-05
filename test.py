import joblib
# Load the pre-trained model
model = joblib.load("saved_model.pkl")

a=[[80196.242251,6.675697,7.275193,3.17,48694.864144 ]]
print(model.predict(a))