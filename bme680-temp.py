#!/usr/bin/env python
# coding: utf-8
# ! pip install uvicorn


# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# # 2. Create the app object
app = FastAPI()

# #. Load trained Pipeline
model = load_model('BME680-temp')

# # Define predict function
@app.post('/predict')
def predict(time , value):
    data = pd.DataFrame([[time, value]])
    data.columns = ['time' , 'value']
    data['value'] = data.value.str.replace('[^\d.]', '').astype(float)
    data['time']= pd.to_datetime(data['time'])
    data.set_index('time', drop=True, inplace=True)
    data['day'] = [i.day for i in data.index]
    data['day_name'] = [i.day_name() for i in data.index]
    data['day_of_year'] = [i.dayofyear for i in data.index]
    data['week_of_year'] = [i.weekofyear for i in data.index]
    data['hour'] = [i.hour for i in data.index]
    data['is_weekday'] = [i.isoweekday() for i in data.index]
    data['minute'] = [i.minute for i in data.index]
    data['second'] = [i.second for i in data.index]
    
    predictions = predict_model(model, data=data)
    
    return {int(predictions['Label'][0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

