#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
# plt.style.use('fivethirtyeight') # For plots
sns.set_style("darkgrid")
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
import os


# In[2]:


#reading the excel file
sales = pd.read_excel("0120-AI___Data_Science_Bootcamp_Time-Series_Dataset.xlsx")
sales.head()


# In[3]:


#indexing the data set
sales.set_index('Date',inplace=True)


# In[4]:


sales.head()


# In[5]:


#plotting the dataset
sales.plot()
plt.show()


# In[6]:


#find the NaNs locations
missIndex = sales['Total_Sales'].index[sales['Total_Sales'].apply(np.isnan)]
print(missIndex)


# In[7]:


#get the average to replace NaN
sales.loc['2019-04-01'] = (sales.loc['2019-03-01'] + sales.loc['2019-05-01'])/2

print(sales.loc['2019-04-01'])


# In[8]:


sales.plot()
plt.show()


# In[9]:


sales.info()


# In[10]:


#Autocorrelation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(sales)
plt.show()


# In[11]:


from statsmodels.tsa.arima_model import ARIMA

# fit model
model = ARIMA(sales, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title('ARMA Fit Residual Error Line Plot')
plt.show()

residuals.plot(kind='kde')
plt.title('ARMA Fit Residual Error Density Plot')
plt.show()
print(residuals.describe())


# In[12]:


from sklearn.metrics import mean_squared_error

X = sales.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=-1)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.legend(['test','prediction'])
plt.show()


# In[13]:


predictions_ARIMA = pd.Series(model_fit.fittedvalues, copy=True)
print(predictions_ARIMA.head())


# In[23]:


model_fit.plot_predict(1,50)
plt.show()


# In[ ]:


or

