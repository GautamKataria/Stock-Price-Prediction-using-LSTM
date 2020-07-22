import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas_datareader as pdr
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error



st.write("""

# Stock Prediction Using Stacked LSTMs


""")
import pandas_datareader as pdr
key="eab6a12a7d6f90aaab8d1f8008c8f2c24b0f48ec"

TickerSymbol = st.sidebar.text_input("TickerSymbol", "TSLA")
options = st.sidebar.selectbox('What to predict', ('close', 'high', 'low', 'open', 'volume', 'adjClose',
                                           'adjHigh', 'adjLow', 'adjOpen', 'adjVolume' ))

ins123 = st.sidebar.text_input("No. of Days to predict", "30")
df = pdr.get_data_tiingo(TickerSymbol, api_key=key)
df.to_csv(f'{TickerSymbol}.csv')
df=pd.read_csv(f'{TickerSymbol}.csv')


df_notime = pd.DataFrame(df.date.str.split(' ',1).tolist(),
                                 columns = ['date','Time'])

df = df.drop(columns =['date'])
df_notime = df_notime.drop(columns = ['Time'])

df = pd.concat([df_notime,df], axis =1)

df = df[['symbol', 'date', 'close', 'high', 'low', 'open', 'volume', 'adjClose','adjHigh', 'adjLow',
        'adjOpen', 'adjVolume', 'divCash', 'splitFactor']]

st.write(f"""

### The dataframe below is the first 10 data entries of {TickerSymbol}. 

""")
st.write(df.head(10))

st.write("""
### predictions will be done after the last data entry which is 
""")
st.write(df.tail(1))

df2 = df[['date', f'{options}']]

st.write("""
### Feature selected with corresponding dates 
""")
st.write(df2.head(10))
df2 = df2.set_index('date')
df_low = df[1000:]
df_low = df_low.set_index('date')
st.write("""
### Graph below for the last year
##### y-axis = Variable 
##### x-axis  = Dates

""")
st.line_chart(df_low[[f'{options}']], width=1000, use_container_width=False)

df1 = df.reset_index()[f'{options}']
#st.write(df2.head(10))

if st.button("predict"):


    #st.write(df1)
    st.write("""
    
    ### The Model is training on the given data
    ### It may take a while so sit back and relax!
    ### The end result would be displayed below
    
    """)


    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)


    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    math.sqrt(mean_squared_error(y_train,train_predict))
    math.sqrt(mean_squared_error(ytest,test_predict))

    look_back=100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

    st.write("""
    ###                  
    ### Model performance check:-
    ##### The blue line is our data
    ##### The orange line is how our model performed on the train data
    ##### The green line is how our model performed on the test data 
    
    """)
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    st.pyplot()

    x_input = test_data[340:].reshape(1, -1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while (i < int(ins123)):

        if (len(temp_input) > 100):

            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i + 1

    print(lst_output)

    changes = 101 + int(ins123)

    day_new = np.arange(1, 101)
    day_pred = np.arange(101, changes)

    plt.plot(day_new, scaler.inverse_transform(df1[1157:]))
    plt.plot(day_pred, scaler.inverse_transform(lst_output))
    st.write("""
        ###                
        ### separated prediction:-
        ##### x-axis = indexes 
        ##### y-axis = variable value
        """)
    st.pyplot()


    df3 = df1.tolist()
    df3.extend(lst_output)

    df3 = scaler.inverse_transform(df3).tolist()
    df3 = df3[1000:]

    st.write("""
    ### Connected overall graph including the values for the previous year:-
    ##### x-axis = indexes 
    ##### y-axis = variable value
    """)

    st.line_chart(df3,width = 1000 ,use_container_width = False)

else:

    st.write("""
    
    ## you can change the variables in the left hand sidebar and then press predict to run the predictions
    
    
    """)
