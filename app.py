# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("Studentmarkpredictor.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
    input_features =  [int(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    #validate input hours
    if input_features[0]<0 or input_features[0]>24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 hours if you live on the earth')

    output = model.predict([features_value])[0][0].round(2)   
    
    df= pd.concat([df,pd.DataFrame({'Study hours':input_features,'Predicted Output':[output]})],ignore_index=True)
    print(df)
    df.to_csv('C:\\Users\\Ishika Chandwadkar\\Downloads\\student_info.csv')
    
    return render_template('index.html', prediction_text='You will get [{}%] marks, when you do [{}] hours per day'.format(output, int(features_value[0])))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
    