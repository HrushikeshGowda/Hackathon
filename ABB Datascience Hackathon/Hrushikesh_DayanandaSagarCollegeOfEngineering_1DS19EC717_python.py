#!/usr/bin/env python
# coding: utf-8

# ### Packages

# In[88]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as sts 
from sklearn.metrics import mean_squared_log_error as rmsle
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf , plot_predict
from pmdarima.arima import auto_arima
import joblib

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (20,15)
plt.rcParams['font.size'] = 20


# ### Loading the data

# In[2]:


data = pd.read_excel('Hackathon_Information_Data.xlsx',sheet_name = 'Data')
data.columns = ['Country','Date','Value']
data.Date = pd.to_datetime(data.Date)
data.head()


# In[3]:


countries = ['Canada','China','Germany','USA','Italy']
dates = pd.date_range(start = '2022-05-01' , periods = 15,freq = 'MS')

forecast_df = pd.DataFrame(
    columns=countries,
    index=dates)
forecast_df.head()


# # Canada Country Analysis

# ### Data Preparation

# In[4]:


df_canada=data[data.Country == 'Canada'].drop('Country',axis = 1).set_index('Date').astype(int)
df_canada.head()


# In[5]:


df_canada.plot()


# In[6]:


df_canada['Value'].plot(kind = 'box')


# In[6]:


df_canada['Value'].sort_values(ascending = False).head(4)


# In[7]:


df_canada['Value'].quantile(q= [0.1 , 0.25  , 0.5 , 0.75 , 1])


# **We can see there is a hude difference between max and the 2nd max value which can also be seen as outlier in the boxplot**
# **Now we will replace that outlier with the 2nd max value in the column**

# In[8]:


df_canada['New_Value'] = df_canada['Value'].replace(6633213 , 3220239)


# ### Seasonal Decomposition Of Data

# In[9]:


# Since there is no seasonality in the data we use additive model
decomposition = seasonal_decompose(df_canada['New_Value'] , model = 'additive')

# Lets plot the trend
decomposition.trend.plot()
plt.show()


# ### Check For Stationarity :-  A D-Fuller Test

# In[10]:


result = sts.adfuller(df_canada['New_Value'])
print(f'ADF Statistic : {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p_value: {result[1]}')

for key , value in result[4].items():
    print('Critical Values :')
    print(f'{key} , {value}')    


# **Since p_value < 0.05 We reject the Null Hypotheses that Data is Non-Stationary** \
# **Therefore Data is Stationary**

# ### Auto Correlation Function
# #### ACF is a statistical representation used to analyze the degree of similarity between a time series and a lagged version of itself.

# In[11]:


plot_pacf(df_canada['New_Value'] , lags = 18)
plt.show()


# **We can see that there is no possible degree of similarity.....Lets get the Return values**

# ### Creating Returns and Plotting ACF

# #### Formula to calculate returns
# 
# #### $ Return_{t} = \frac{value_{t} - value_{t-1}}{value_{t-1}} * 100 $

# In[12]:


df_canada['Return'] = df_canada['New_Value'].pct_change(1).mul(100)
df_canada.head()


# In[13]:


plot_pacf(df_canada['Return'][1:] , lags = 18)
plot_acf(df_canada['Return'][1:] , lags = 18)
plt.show()


# **By see the pacf and acf plots we can have these pairs of orders for ARIMA model :- \ 
# (1,0,0) , (1,0,1) , (3,0,0) , (3,0,1) , (4,0,0) , (4,0,1)**

# ### Finding The Suitable Model

# In[259]:


orders = [(1,0,0) , (1,0,1) , (3,0,0) , (3,0,1) , (4,0,0) , (4,0,1)]

for order in orders:
    ar,ma = order[0] , order[-1]
    arima = ARIMA(df_canada['Return'][1:] , order = (ar,0,ma))
    results = arima.fit()
    print(f'ARIMA{order} has LLF = {results.llf} and AIC = {results.aic}')


# **We can choose the model that has high LLF and low AIC value** \
# **After some analysis lets choose ARIMA(4,0,0)**

# In[14]:


arima_400 = ARIMA(df_canada['Return'][1:] , order =(4,0,0))
results_400 = arima_400.fit()


# In[16]:


# Plot the residual to check is there any degree of similarities in it

plot_pacf(results_400.resid , lags = 15)
plt.show()

# We can proceed since there are no lags


# ### Forecasting Future Values

# In[17]:


# Lets forecast for next 15 months
returns = results_400.forecast(steps = 15).values.tolist()

# Store the data values for calculating values from forecast values
values = df_canada['New_Value'].values.tolist()
len(values)


# **Formula to calculate original values from returns**
# 
# **$ value_{t} = (value_{t-1}*\frac{return}{100}) + value_{t-1} $**

# In[18]:


# Lets convert all the returns into values and add it to the data values list

for ret in returns:
    ret = ret/100
    last = values[-1]
    res = (last*ret)+last
    values.append(round(res))
    
# Plot the data values
plt.plot(values, label = 'Forecasted' , color = 'red')
plt.plot(df_canada['New_Value'].values, label = 'Previous values' ,color = 'blue')
plt.show()


# In[19]:


forecast_df['Canada'] = values[40:]


# In[282]:


# joblib.dump(results_400 ,  'Model_Canada.pkl')


# # China Country Analysis

# ### Data Preparation

# In[20]:


df_china=data[data.Country == 'China'].drop('Country',axis = 1).set_index('Date').astype(int)
df_china.head()


# In[21]:


df_china.plot()
plt.show()


# ### Seasonal Decomposition Of Data

# In[22]:


decomposition = seasonal_decompose(df_china['Value'] , model = 'additive')
decomposition.trend.plot()

plt.show()


# ### Check For Stationarity :-  A D-Fuller Test

# In[23]:


result = sts.adfuller(df_china['Value'])
print(f'ADF Statistic : {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p_value: {result[1]}')

for key , value in result[4].items():
    print('Critical Values :')
    print(f'{key} , {value}')    


# **Since p_value > 0.05 We can't reject the Null Hypotheses that Data is Non-Stationary** \
# **Therefore Data is Non-Stationary**
# **Let's take the Difference of the values**

# **Formula to calculate difference from original values**
# 
# **$ diff_{t} = value_{t} - value_{t-1} $**

# In[24]:


df_china['Diff_Value'] = df_china['Value'].diff()
df_china.head()


# In[25]:


result = sts.adfuller(df_china['Diff_Value'][1:])
print(f'ADF Statistic : {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p_value: {result[1]}')

for key , value in result[4].items():
    print('Critical Values :')
    print(f'{key} , {value}')    


# **Since p_value < 0.05 We reject the Null Hypotheses that Data is Non-Stationary** \
# **Therefore Data is Stationary**

# ### Auto Correlation Function

# In[26]:


plot_pacf(df_china['Diff_Value'][1:] ,  lags = 15)
plot_acf(df_china['Diff_Value'][1:] ,  lags = 15)
plt.show()


# **By see the pacf and acf plots we can have these pairs of orders for ARIMA model :-
# (1,0,1) , (1,0,3) , (2,0,1) , (2,0,3)**

# ### Finding The Suitable Model

# In[27]:


orders = [(1,0,1) , (1,0,3) , (2,0,1) , (2,0,3)]

for order in orders:
    ar,ma = order[0] , order[-1]
    arima = ARIMA(df_china['Diff_Value'][1:] , order = (ar,0,ma))
    results = arima.fit()
    print(f'ARIMA{order} has LLF = {results.llf} and AIC = {results.aic}')


# **We can choose ARIMA(2,0,3)**

# In[28]:


arima_203 = ARIMA(df_china['Diff_Value'][1:] , order = (2,0,3))
results_203 = arima_203.fit()


# In[29]:


plot_pacf(results_203.resid , lags = 15)
plt.show()

#we can proceed since there are no lags


# ### Forecasting Future Values

# In[31]:


# Lets forecast for next 15 months
differences = results_203.forecast(steps = 15).values.tolist()

# Store the data values for calculating values from forecast values
values = df_china['Value'].values.tolist()
len(values)


# **Formula to calculate original values from differences**
# 
# **$ value_{t} = diff_{t} + value_{t-1} $**

# In[32]:


for diff in differences:
    val = diff + values[-1]
    values.append(round(val))
    
plt.plot(values , label = 'Forecasted' , color = 'red')
plt.plot(df_china['Value'].values , label = 'Previous values' ,color = 'blue')
plt.legend()
plt.show()


# In[33]:


forecast_df['China'] = values[40:]


# In[32]:


# joblib.dump(results_203 , 'Model_China.pkl')


# # Germany Country Analysis

# ### Data Preparation

# In[34]:


df_germany=data[data.Country == 'Germany'].drop('Country',axis = 1).set_index('Date').astype(int)
df_germany.head()


# In[35]:


df_germany.plot()


# ### Seasonal Decomposition Of Data

# In[36]:


decomposition = seasonal_decompose(df_germany['Value'] , model = 'additive')
decomposition.trend.plot()

plt.show()


# ### Check For Stationarity :-  A D-Fuller Test

# In[37]:


result = sts.adfuller(df_germany['Value'])
print(f'ADF Statistic : {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p_value: {result[1]}')

for key , value in result[4].items():
    print('Critical Values :')
    print(f'{key} , {value}')    


# **Since p_value > 0.05 We can't reject the Null Hypotheses that Data is Non-Stationary** \
# **Therefore Data is Non-Stationary**
# **Let's take the Returns of the values**

# ### Creating Returns and Plotting ACF

# In[38]:


# Since Taking Difference didn't worked well on forecasting i choose to use Returns


# In[39]:


df_germany['Return'] = df_germany['Value'].pct_change(1).mul(100)
df_germany.head()


# In[40]:


result = sts.adfuller(df_germany['Return'][1:])
print(f'ADF Statistic : {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p_value: {result[1]}')

for key , value in result[4].items():
    print('Critical Values :')
    print(f'{key} , {value}')    


# **Since p_value < 0.05 We reject the Null Hypotheses that Data is Non-Stationary** \
# **Therefore Data is Stationary**

# ### Auto Correlation Function

# In[41]:


plot_pacf(df_germany['Return'][1:] ,  lags = 15)
plot_acf(df_germany['Return'][1:] ,  lags = 15)
plt.show()


# **The possible pairs of orders for ARIMA model can be :- \
# (1,0,0) , (0,0,1) , (1,0,1) , (4,0,0) , (4,0,1) , (5,0,0) , (5,0,1) , (6,0,0) , (6,0,1)** 

# ### Finding The Suitable Model

# In[42]:


orders= [(1,0,0) , (0,0,1) , (1,0,1) , (4,0,0) , (4,0,1) , (5,0,0) , (5,0,1) , (6,0,0) , (6,0,1)]

for order in orders:
    ar,ma = order[0] , order[-1]
    arima = ARIMA(df_germany['Return'][1:] , order = (ar,0,ma))
    results = arima.fit()
    print(f'ARIMA{order} has LLF = {results.llf} and AIC = {results.aic}')


# **After making some analysis let's choose ARIMA(5,0,0)**

# In[43]:


arima_500 = ARIMA(df_germany['Return'][1:] , order = (5,0,0))
results_500 = arima_500.fit()


# In[44]:


plot_pacf(results_500.resid , lags = 15)
plt.show()

#we can ignore that 6 because even if we take ARIMA(6,0,0) it will be present


# ### Forecasting Future Values

# In[45]:


# Lets forecast for next 15 months
returns = results_500.forecast(steps = 15).values.tolist()

# Store the data values for calculating values from forecasted values
values = df_germany['Value'].values.tolist()
len(values)


# In[46]:


for ret in returns:
    ret = ret/100
    last = values[-1]
    res = (last*ret)+last
    values.append(round(res))
    
plt.plot(values , label = 'Forecasted' , color = 'red')
plt.plot(df_germany['Value'].values , label = 'Previous values' ,color = 'blue')
plt.legend()
plt.show()


# In[47]:


forecast_df['Germany'] = values[40:]


# In[256]:


# joblib.dump(results_500 , 'Model_Germany.pkl')


# # USA Country Analysis

# ### Data Preparation

# In[48]:


df_usa=data[data.Country == 'USA'].drop('Country',axis = 1).set_index('Date').astype(int)
df_usa.head()


# In[49]:


df_usa.plot()
plt.show()


# ### Seasonal Decomposition Of Data

# In[50]:


decomposition = seasonal_decompose(df_usa['Value'] , model = 'additive')
decomposition.trend.plot()

plt.show()


# ### Check For Stationarity :-  A D-Fuller Test

# In[51]:


result = sts.adfuller(df_usa['Value'])
print(f'ADF Statistic : {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p_value: {result[1]}')

for key , value in result[4].items():
    print('Critical Values :')
    print(f'{key} , {value}')    


# **Since p_value > 0.05 We can't reject the Null Hypotheses that Data is Non-Stationary \
# Therefore Data is Non-Stationary  \
# Let's take the Difference of the values. \
# Since single level difference don't have any degree of similarity \
# we will make two level difference**

# In[52]:


df_usa['Diff_Value'] = df_usa['Value'].diff(2)
df_usa.head()


# In[53]:


result = sts.adfuller(df_usa['Diff_Value'][2:])
print(f'ADF Statistic : {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p_value: {result[1]}')

for key , value in result[4].items():
    print('Critical Values :')
    print(f'{key} , {value}')    


# **Since p_value < 0.05 We reject the Null Hypotheses that Data is Non-Stationary** \
# **Therefore Data is Stationary**

# ### Auto Correlation Function

# In[54]:


plot_pacf(df_usa['Diff_Value'][2:] ,  lags = 15)
plot_acf(df_usa['Diff_Value'][2:] ,  lags = 15)
plt.show()


# **The possible pairs of orders for ARIMA model can be :- (1,0,0) , (0,0,1) , (1,0,1)** 

# ### Finding The Suitable Model

# In[55]:


orders= [(1,0,0) , (0,0,1) , (1,0,1)]

for order in orders:
    ar,ma = order[0] , order[-1]
    arima = ARIMA(df_usa['Diff_Value'][2:] , order = (ar,0,ma))
    results = arima.fit()
    print(f'ARIMA{order} has LLF = {results.llf} and AIC = {results.aic}')


# **Let's choose ARIMA(1,0,1)**

# In[56]:


arima_101 = ARIMA(df_usa['Diff_Value'][2:] , order = (1,0,1))
results_101 = arima_101.fit()


# In[57]:


plot_pacf(results_101.resid , lags = 15)
plt.show()

#we can proceed since there are no lags


# ### Forecasting Future Values

# In[65]:


# Lets forecast for next 15 months
differences = results_101.forecast(steps = 15).values.tolist()

# Store the data values for calculating values from forecast values
values = df_usa['Value'].values.tolist()
len(values)


# In[66]:


for diff in differences:
    val = diff + values[-2]
    values.append(round(val))
    
plt.plot(values , label = 'Forecasted' , color = 'red')
plt.plot(df_usa['Value'].values , label = 'Previous values' ,color = 'blue')
plt.legend()
plt.show()


# In[67]:


forecast_df['USA'] = values[40:]


# In[168]:


# joblib.dump(results_101 , 'Model_USA.pkl')


# # Italy Country Analysis

# ### Data Preparation

# In[68]:


df_italy=data[data.Country == 'Italy'].drop('Country',axis = 1).set_index('Date').astype(int)
df_italy.head()


# In[69]:


df_italy.plot()
plt.show()


# ### Seasonal Decomposition Of Data

# In[70]:


decomposition = seasonal_decompose(df_italy['Value'] , model = 'additive')
decomposition.trend.plot()

plt.show()


# ### Check For Stationarity :-  A D-Fuller Test

# In[71]:


result = sts.adfuller(df_italy['Value'])
print(f'ADF Statistic : {result[0]}')
print(f'n_lags: {result[2]}')
print(f'p_value: {result[1]}')

for key , value in result[4].items():
    print('Critical Values :')
    print(f'{key} , {value}')    


# **Since p_value < 0.05 We reject the Null Hypotheses that Data is Non-Stationary** \
# **Therefore Data is Stationary**

# ### Auto Correlation Function

# In[72]:


plot_pacf(df_italy['Value'] , lags = 18)
plot_acf(df_italy['Value'] , lags = 18)
plt.show()


# **We can see that there is no possible degree of similarity.....Lets get the Difference values**

# ### Differencing and Plotting ACF

# In[73]:


df_italy['Diff_Value'] = df_italy['Value'].diff()[1:]
df_italy.head()


# In[74]:


plot_pacf(df_italy['Diff_Value'][1:] , lags = 18)
plot_acf(df_italy['Diff_Value'][1:] , lags = 18)
plt.show()


# **By see the pacf and acf plots we can have 3 pairs of orders for ARIMA model :- \
# (2,0,0) , (2,0,2) , (0,0,2)**

# ### Finding The Suitable Model

# In[75]:


#Let's try auto arima o find some insights

auto_model = auto_arima(df_italy['Diff_Value'][1:])
auto_model.summary()


# In[76]:


# Let's take some pair of orders along with one we got from auto arima
orders= [(1, 0, 2) , (2,0,0) , (2,0,2) , (0,0,2)]

for order in orders:
    ar,ma = order[0] , order[-1]
    arima = ARIMA(df_italy['Diff_Value'][1:] , order = (ar,0,ma))
    results = arima.fit()
    print(f'ARIMA{order} has LLF = {results.llf} and AIC = {results.aic}')


# **Let's choose ARIMA(1,0,2)**

# In[77]:


arima_102 = ARIMA(df_italy['Diff_Value'][1:] , order = (1,0,2))
results_102 = arima_102.fit()


# In[78]:


plot_pacf(results_102.resid , lags = 15)
plt.show()

#we can proceed since there are no lags below 10


# In[79]:


fig, ax = plt.subplots()
df_italy['Diff_Value'].plot(ax=ax)
plot_predict(results_102, start = '2022-05-01',end = '2023-10-01', alpha=0.05, ax=ax)
plt.show()


# ### Forecasting Future Values

# In[80]:


# Lets forecast for next 15 months
differences = results_102.forecast(steps = 15).values.tolist()

# Store the data values for calculating values from forecasted values
values = df_italy['Value'].values.tolist()
len(values)


# In[81]:


for diff in differences:
    val = diff + values[-1]
    values.append(round(val))
    
plt.plot(values , label = 'Forecasted' , color = 'red')
plt.plot(df_italy['Value'].values , label = 'Previous values' ,color = 'blue')
plt.legend()
plt.show()


# In[366]:


# forecast_df['Italy'] = values[40:]


# In[212]:


# joblib.dump(results_102 , 'Model_Italy.pkl')


# # Saving The Forecasted DataFrame

# In[379]:


# forecast_df = forecast_df.reset_index().rename(columns = {'index':'Date'})
# forecast_df.head()


# In[382]:


# forecast_df.to_csv('Hrushikesh_DayanandaSagarCollegeOfEngineering_1DS19EC717_submission.csv' , index = False)


# In[87]:


# forecast_data = pd.read_csv('Hrushikesh_DayanandaSagarCollegeOfEngineering_1DS19EC717_submission.csv')
# forecast_data.tail()


# In[ ]:




