#!/usr/bin/env python
# coding: utf-8

# ### Reading the market data of BAJAJFINSV stock and preparing a training dataset and validation dataset.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('E:\End-2-end Projects\Time_Series/BAJFINANCE.csv')
df.head()


# In[3]:


df.set_index('Date',inplace=True)


# In[ ]:





# #### Plotting the target variable VWAP over time

# In[5]:


df['VWAP'].plot()


# ### so u can observe here some kind of Seasonality
Feature Engineering
Almost every time series problem will have some external features or some internal feature engineering to help the model.

Let's add some basic features like lag values of available numeric features that are widely used for time series problems. Since we need to predict the price of the stock for a day, we cannot use the feature values of the same day since they will be unavailable at actual inference time. We need to use statistics like mean, standard deviation of their lagged values.

We will use three sets of lagged values, one previous day, one looking back 7 days and another looking back 30 days as a proxy for last week and last month metrics.
# ### Data Pre-Processing

# In[6]:


df.shape


# In[7]:


df.isna().sum()


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.isna().sum()


# In[10]:


df.shape


# In[11]:


data=df.copy()


# In[12]:


data.dtypes


# In[13]:


data.columns


# In[15]:


lag_features=['High','Low','Volume','Turnover','Trades']
window1=3
window2=7


# In[16]:


for feature in lag_features:
    data[feature+'rolling_mean_3']=data[feature].rolling(window=window1).mean()
    data[feature+'rolling_mean_7']=data[feature].rolling(window=window2).mean()


# In[17]:


for feature in lag_features:
    data[feature+'rolling_std_3']=data[feature].rolling(window=window1).std()
    data[feature+'rolling_std_7']=data[feature].rolling(window=window2).std()


# In[18]:


data.head()


# In[19]:


data.columns


# In[20]:


data.shape


# In[21]:


data.isna().sum()


# In[22]:


data.dropna(inplace=True)


# In[23]:


data.columns


# In[24]:


ind_features=['Highrolling_mean_3', 'Highrolling_mean_7',
       'Lowrolling_mean_3', 'Lowrolling_mean_7', 'Volumerolling_mean_3',
       'Volumerolling_mean_7', 'Turnoverrolling_mean_3',
       'Turnoverrolling_mean_7', 'Tradesrolling_mean_3',
       'Tradesrolling_mean_7', 'Highrolling_std_3', 'Highrolling_std_7',
       'Lowrolling_std_3', 'Lowrolling_std_7', 'Volumerolling_std_3',
       'Volumerolling_std_7', 'Turnoverrolling_std_3', 'Turnoverrolling_std_7',
       'Tradesrolling_std_3', 'Tradesrolling_std_7']


# In[25]:


training_data=data[0:1800]
test_data=data[1800:]


# In[26]:


training_data


# In[ ]:





# In[27]:


get_ipython().system('pip install pmdarima')


# In[28]:


from pmdarima import auto_arima


# In[30]:


import warnings
warnings.filterwarnings('ignore')


# In[31]:


model=auto_arima(y=training_data['VWAP'],exogenous=training_data[ind_features],trace=True)


# In[32]:


model.fit(training_data['VWAP'],training_data[ind_features])


# In[34]:


forecast=model.predict(n_periods=len(test_data), exogenous=test_data[ind_features])


# In[35]:


test_data['Forecast_ARIMA']=forecast


# In[37]:


test_data[['VWAP','Forecast_ARIMA']].plot(figsize=(14,7))


# #### The Auto ARIMA model seems to do a fairly good job in predicting the stock price

# In[ ]:





# #### Checking Accuracy of our model

# In[38]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[40]:


np.sqrt(mean_squared_error(test_data['VWAP'],test_data['Forecast_ARIMA']))


# In[41]:


mean_absolute_error(test_data['VWAP'],test_data['Forecast_ARIMA'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




