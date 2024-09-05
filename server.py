from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load("saved_model.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

        avg_area_income = float(request.form['avg_area_income'])
        area_house_age = float(request.form['area_house_age'])
        avg_area_rooms = float(request.form['avg_area_rooms'])
        avg_area_bedrooms = float(request.form['avg_area_bedrooms'])
        area_population = float(request.form['area_population'])
        
        
        # Prepare input for prediction
        input_data = np.array([[avg_area_income, area_house_age, avg_area_rooms, avg_area_bedrooms, area_population]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display prediction 
        return render_template('index.html', prediction=prediction)
  



if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
