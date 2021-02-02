import tensorflow as tf
import numpy as np
import matplotlib
import pandas_datareader as pdr
import pandas as pd
import datetime as dt
import webbrowser
import time
import matplotlib.pyplot as plt
import os
import math
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def wholefunction(share_code):
    def data():
 
        #define parameters for input
        days_of_share_values = 3000
        end_of_share = 200
        number_input_days = 25

        #define path        
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'data/'+share_code+'.csv')

        #get the data from yahoo
        endtime = dt.date.today() - dt.timedelta(days = 1)
        starttime =  dt.date.today() - dt.timedelta(days = days_of_share_values)
        share = pdr.get_data_yahoo(share_code, start = starttime, end = endtime )
        share_fitted = share.head(int(days_of_share_values*0.7-end_of_share*0.7))
        share_fitted.to_csv(filename)
        close = share["Close"]
    
    
        #return data
        return close, number_input_days, share_code,days_of_share_values,filename
    
    #get data from function
    close,number_input_days, share_code, days_of_share_values,filename= data()

    #load close data from file
    CSV_FILE  = (filename)
    DAYS_BEFORE = number_input_days
    initial_stock_data = np.loadtxt(CSV_FILE,delimiter=",",skiprows=9,usecols=(4),comments="#",dtype=float)
    initial_stock_data = np.array(initial_stock_data,dtype="float").reshape(-1,1) 

    # scale data with minmaxscaler from sklearn
    min_max_scaler = MinMaxScaler(feature_range=(0,1))
    stock_data = min_max_scaler.fit_transform(initial_stock_data)


    # reshape data 

    def arrange_data(data, days):
        days_before_values = [] # T- days
        days_values = []  # T
        for i in range(len(data) - days -1):
            days_before_values.append(data[i:(i+days)]) 
            days_values.append(data[i + days]) 
        return np.array(days_before_values),np.array(days_values)
    
    days_before_values, days_values =  arrange_data(stock_data,DAYS_BEFORE)

    # split data for learning and testing

    def split_to_percentage(data,percentage):
        return  data[0: int(len(data)*percentage)] , data[int(len(data)*percentage):]

    percentage_training =0.95 
    X_train, X_test = split_to_percentage(days_before_values,percentage_training) 
    Y_train, Y_test = split_to_percentage(days_values,percentage_training)  


    # define the neural network
    number_of_neurons = 10
    number_of_activation_neurons =5
    learning_rate = 0.01
    
    stock_model = Sequential()
    stock_model.add(LSTM(number_of_neurons,input_shape=(DAYS_BEFORE,1),return_sequences=True))
    stock_model.add(LSTM(number_of_activation_neurons,activation="relu"))
    return_sequences=True

    stock_model.add(Dense(1))

    #train network and safe model
    sgd = SGD(lr=learning_rate)
    stock_model.summary()
    stock_model.compile(loss="mean_squared_error", optimizer=sgd, metrics=[tf.keras.metrics.mse])
    stock_model.fit(X_train, Y_train, epochs=100, verbose=1)

    stock_model.save("models/"+share_code+"stock.h5")

    # evaluate with function evaluate
    score, _ = stock_model.evaluate(X_test,Y_test)
    rmse = math.sqrt(score)
    print("RMSE {}".format(rmse))

    #alternative evaluate
    def fitAndEvaluate():
        stock_model.fit(X_train,Y_train,epochs=	100, validation_data=(X_test,Y_train))

    #predict one day in future and add this to input for more predictions
    def current_prediction():
        #define input arrays
        previous_prediction = np.array([0])
        data_prediction = (np.array(close.tail(number_input_days))).reshape(number_input_days,1)
        #scale the data between 0,1
        data_prediction = min_max_scaler.fit_transform(data_prediction)
        
        
        whole_current_data = data_prediction

        #last data is needed to calculate change in pct 
        data_last = (np.array(close))[-1:]
        data_prediction = data_prediction.reshape(1,data_prediction.shape[0],1)

        #define how many days should get predicted
        days_predicting = 1 

        for i in range(days_predicting):
            #shape data 
            data_prediction = data_prediction.reshape(1,number_input_days,1)
            #predict next day
            prediction_current = stock_model.predict(data_prediction)
            
            #refresh input with predicted data
            prediction_current = prediction_current+(((prediction_current[0]) -(previous_prediction[0])))
            prediction_current = prediction_current.reshape(1,prediction_current.shape[0],1)
            data_prediction = np.append(data_prediction,prediction_current)
            whole_current_data = np.append(whole_current_data,prediction_current)
            previous_prediction = prediction_current

            #correct size for input
            if len(data_prediction) > number_input_days:
                data_prediction = np.delete(data_prediction,[0])
            
        #inverse 0,1 scale 
        prediction_current = prediction_current.reshape(prediction_current.shape[0],1)
        prediction_current = min_max_scaler.inverse_transform(prediction_current)
        data_prediction = data_prediction.reshape(data_prediction.shape[0],1)
        data_prediction = min_max_scaler.inverse_transform(data_prediction)
        
        return prediction_current, data_prediction, days_predicting, data_last

    #get data from function
    current_prediction, last_input,days_predicting,data_last=current_prediction()
    pct_current_prediction= current_prediction/data_last
    #return prediction
    return current_prediction,pct_current_prediction


#predict for a few shares 
share_codes =["dax","msci","djia","spgi","dte","bayry","basfy","crzby","bmwyy","vwagy","vlvly","gold","tsla","amzn"]
results=[[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range(len(share_codes)):
    results[i]= wholefunction(share_codes[i])

    print(results)


#create html file
dirname = os.path.dirname(__file__)

filename_html = os.path.join(dirname, 'HTML/displayResults.html')

#absolut path of apache html folder
#filename_html = "/var/www/html/HTML/displayResults.html"


#define content
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results</title>
    <link rel="stylesheet" href="../CSS/unofficial.css">
</head>
<body>
    <p id="msg">prediction dax: {results[0]} </p>
    <p id="msg">prediction msci: {results[1]} </p>
    <p id="msg">prediction dow jones: {results[2]} </p>
    <p id="msg">prediction standart & poors: {results[3]} </p>
    <p id="msg">prediction telekom: {results[4]} </p>
    <p id="msg">prediction bayer: {results[5]} </p>
    <p id="msg">prediction basf: {results[6]} </p>
    <p id="msg">prediction commerzbank: {results[7]} </p>
    <p id="msg">prediction bmw: {results[8]} </p>
    <p id="msg">prediction vw: {results[9]} </p>
    <p id="msg">prediction volvo: {results[10]} </p>
    <p id="msg">prediction gold: {results[11]} </p>
    <p id="msg">prediction tesla: {results[12]} </p>
    <p id="msg">prediction amazon: {results[13]} </p>
</body>
</html>"""

#save doc
with open(filename_html, "w" ) as html_file: 
    html_file.write(html_content)
    
