

#import pandas_datareader as pdr
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import streamlit as st
st.title('Sentiment Analysis of')

# In[3]:


df=pd.read_csv("file_name.csv")
#df=data.drop(['High', 'Low','Open','Adj Close','Volume'], axis = 1)
#df=df.reset_index()
#df.rename({'Close':'p_stnet'},inplace=True,axis=1)



# In[4]:


df["Date_"] = pd.to_datetime(df.Date)
df["Month"] = df.Date_.dt.strftime("%b") # month extraction
df["year"] = df.Date_.dt.strftime("%Y") # year extraction
df=df.iloc[:,1:]


# In[5]:


df.info()


# In[6]:


df['Close']=df['Close'].astype('int64')
df['t']=[i for i in range(1,len(df)+1)]
df['t_square']=[i**2 for i in df['t']]



# In[7]:


plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=df,values='Close',index="year",columns="Month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# In[8]:


# Boxplot for ever
plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="Month",y="Close",data=df)
plt.subplot(212)
sns.boxplot(x="year",y="Close",data=df)


# In[9]:


plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Close",data=df)


# In[10]:


df.info()


# In[11]:


#one hot encoding
df1= pd.get_dummies(df.Month)
df = pd.concat([df, df1], axis=1)



# In[12]:


Train = df.head(len(df)-365)
Test = df.tail(365)


# 
# # Forecasting using model base method

# In[13]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Close~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Close'])-np.array(pred_linear))**2))
#rmse_linear


# In[14]:


#Quadratic 

Quad = smf.ols('Close~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Close'])-np.array(pred_Quad))**2))
#rmse_Quad


# In[15]:


#Additive seasonality 

add_sea = smf.ols('Close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Close'])-np.array(pred_add_sea))**2))
#rmse_add_sea


# In[16]:


#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Close~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Close'])-np.array(pred_add_sea_quad))**2))
#rmse_add_sea_quad


# In[17]:


#Compare the results 

data = {"MODEL":pd.Series(["linear","Quad","add_sea","add_sea_quad"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Quad,rmse_add_sea,rmse_add_sea_quad])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[18]:




# Creating predicting Dataset

# ##  Creating Forcasting   dataset
# 

# In[19]:

#print(df)
x=df.iloc[-1,0]
#print("xxxxxxxxxxxxxx",x)
rng = pd.date_range(start=x, periods=365, freq='D',tz=None)
pr = pd.DataFrame(rng)
pr["Date_"] = pd.to_datetime(pr.iloc[:,0],format="%b-%y")
pr["Month"] = pr.Date_.dt.strftime("%b") # month extraction
pr["year"] = pr.Date_.dt.strftime("%Y") # year extraction
pr=pr.iloc[:,1:]
#pr


# In[20]:


#one hot encoding
pr1= pd.get_dummies(pr.Month)
#pr1


# In[21]:


predict_data= pd.concat([pr, pr1], axis=1)
#predict_data


# In[22]:


predict_data['t']=[i for i in range(len(df)+1,len(df)+366)]
predict_data['t_square']=[i**2 for i in predict_data['t']]
predict_data_1=predict_data
#predict_data_1


# In[ ]:





# In[23]:


#combining Data Row wise (adding for one year after 2006)
vertical_concat = pd.concat([df, predict_data_1], axis=0)
vertical_concat=vertical_concat.reset_index()
vertical_concat.drop(['index'],axis=1,inplace=True)
#vertical_concat


# In[ ]:





# In[24]:


#Data Splitting 
Train = vertical_concat.head(len(vertical_concat)-365)
Test = vertical_concat.tail(365)


# In[25]:


#Build the model on entire data set
add_sea_Quad = smf.ols('Close~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()


# In[26]:


#predicting for next year 
pred_new  = pd.Series(add_sea_Quad.predict(vertical_concat))
print(pred_new)
vertical_concat["forecasted_Close"] = pd.Series(pred_new)
#vertical_concat


# #Predicting for entire Dataset
# pred_entire  = pd.Series(add_sea_Quad.predict(vertical_concat))
# len(pred_entire)

# In[27]:


#vertical_concat


# In[28]:


#Graph Show forecasted value by model base method Additive Seasonality Quadratic 
fig2,ax=plt.subplots(1,1,figsize=(18,6))
ax.plot(vertical_concat.Date_,vertical_concat.Close,label="org")
ax.plot(vertical_concat.Date_,vertical_concat.forecasted_Close,label="forecasted")


# # Forecasting using Data Driven Models

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing 


# In[30]:


#Data Splitting
Train = df.head(len(df)-365)
Test = df.tail(365)


# Moving Average

# In[31]:
#Forecasting using Data Driven Method

plt.figure(figsize=(18,6))
df.Close.plot(label="org")
for i in range(2,24,6):
    df["Close"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


# Time series decomposition plot 

# In[32]:


decompose_ts_add = seasonal_decompose(df.Close,period=12)
decompose_ts_add.plot()
plt.show()


#  ACF plots and PACF plots

# In[33]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(df.Close,lags=12)
tsa_plots.plot_pacf(df.Close,lags=12)
plt.show()


# Evaluation Metric MAPE

# In[34]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# In[35]:


#Data Splitting
Train = df.head(len(df)-365)
Test = df.tail(365)


# Simple Exponential Method

# In[36]:


ses_model = SimpleExpSmoothing(Train["Close"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Close) 


# Holt method 

# In[37]:


# Holt method 
hw_model = Holt(Train["Close"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Close) 


# Holts winter exponential smoothing with additive seasonality and additive trend
# 

# In[38]:


hwe_model_add_add = ExponentialSmoothing(Train["Close"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Close) 


# In[39]:


#Compare the results 

data = {"MODEL":pd.Series(["ses_model","hw_model","hwe_model_add_add"]),"RMSE_Values":pd.Series([MAPE(pred_ses,Test.Close),MAPE(pred_hw,Test.Close),MAPE(pred_hwe_add_add,Test.Close)])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[40]:


#Data Splitting
Train = vertical_concat.head(len(vertical_concat)-365)
Test = vertical_concat.tail(365)
print(len(Train))


# In[41]:


#Build the model on entire data set
ses_model = SimpleExpSmoothing(Train["Close"]).fit(smoothing_level=0.2)


# In[42]:


#Forecasting for next 365 time periods
ses_model.forecast(365)


# In[43]:


#Result is not good so build second best model
hwe_model_add_add = ExponentialSmoothing(df["Close"],seasonal="add",trend="add",seasonal_periods=4).fit()
#Forecasting for next 24 time periods 
result=hwe_model_add_add.forecast(365)
#result


# In[44]:


#vertical_concat


# In[45]:


vertical_concat["forecasted_Close_hwe_model_add_add"] = result
#vertical_concat


# In[57]:


#shaded region show forecsated value by Data Driven Models (Holts winter exponential smoothing with additive seasonality and additive trend)
st.title('Forecasting Using Linear And Data Driven Methods')
fig3,ax=plt.subplots(1,1,figsize=(18,6))
ax.plot(vertical_concat.Date_,vertical_concat.Close,label="org")
ax.plot(vertical_concat.Date_,vertical_concat.forecasted_Close_hwe_model_add_add,label="hwe_model_add_add_datadriven")
ax.plot(vertical_concat.Date_,vertical_concat.forecasted_Close,label="forecasted_linear")
plt.legend()
ax.set(xlabel='Date', ylabel='Tax_Revenue')
#plt.axvspan('2022-06-01','2023-05-31', color='red', alpha=0.5)
st.pyplot(fig3)

# # Forecasting using ARIMA

# In[65]:
#ARIMA
df=pd.read_csv("file_name.csv")
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
#data_org = read_csv('CocaCola_Sales_Rawdata_org.csv', index_col=0, parse_dates=True, squeeze=True)
X=df['Close'].values

#Data Splitting
Train_size = int(len(df))
Train,Test=X[0:Train_size],X[Train_size:]
 
print(len(Train))
X = X.astype('float32')
#train_size=int(len(X)*0.50)
#train,test=X[0:train_size],X[train_size:]
print(len(X))
#Train


# In[66]:


   # from statsmodels.tsa.arima_model import ARIMA
    #model = ARIMA(Train, order=(3,1,0))
    #model_fit = model.fit()
    #forecast=model_fit.forecast(steps=365)[0]
import statsmodels.api as sm
model = sm.tsa.arima.ARIMA(Train, order=(1,1,2))
model_fit= model.fit()
forecast=model_fit.forecast(steps=365)[0]
#forecast


# In[69]:
st.title('Forecasting Using ARIMA')
from statsmodels.graphics.tsaplots import plot_predict
fig4,ax=plt.subplots(1,1,figsize=(18,6))
plt.xlabel("Date")
#model_fit.plot_predict(1, len(X)+365,ax=ax)
plot_predict(model_fit,start=1,end=1700,ax=ax)
st.pyplot(fig4)


# In[71]:





# In[ ]:




