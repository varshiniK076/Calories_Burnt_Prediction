from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('calories.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Age = float(request.form['Age'])
        Height = float(request.form['Height'])
        Weight = float(request.form['Weight'])
        Duration = float(request.form['Duration'])
        Heart_Rate = float(request.form['Heart_Rate'])
        Body_Temp = float(request.form['Body_Temp'])

        # HTML sends Male/Female → convert to Gender_male
        gender_value = request.form['Gender']
        Gender_male = 1 if gender_value.lower() == 'male' else 0

        input_df = pd.DataFrame([[
            Age,
            Height,
            Weight,
            Duration,
            Heart_Rate,
            Body_Temp,
            Gender_male
        ]], columns=[
            'Age',
            'Height',
            'Weight',
            'Duration',
            'Heart_Rate',
            'Body_Temp',
            'Gender_male'
        ])

        numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        prediction = model.predict(input_df)

        return render_template(
            'index.html',
            prediction_text=f"Predicted Calories Burned: {prediction[0]:.2f}"
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
