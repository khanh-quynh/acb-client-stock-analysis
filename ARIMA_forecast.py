# general setup
import pandas as pd
from pandas.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import signal

import warnings
warnings.filterwarnings('ignore')

# load dataset
df_draft = pd.read_excel('Data_HPG.xlsx')
df_draft.head(5)

df_draft['Ngày'] = pd.to_datetime(df_draft['Ngày'])


# Change to RATE_OF_CHANGE
''' USE RATE OF CHANGE FOR PREDICTION '''
rate = [0]
for i in range(1,len(df_draft)):
    rate.append((df_draft['GIA_DIEUCHINH'][i] - df_draft['GIA_DIEUCHINH'][i-1])*100/df_draft['GIA_DIEUCHINH'][i-1])

df_draft['ROC'] = rate
df_draft.head()


# TRAIN and TEST dataframe
df = pd.DataFrame({
    'Date':df_draft['Ngày'],
    'ROC':df_draft['ROC']})

df.set_index('Date', inplace=True)
df.index

df_train = df[:690]
df_test = df[690:]


# check for STATIONARY
plt.xticks(rotation=45)
plt.plot(df_train, color = 'green')

plt.xticks(rotation=45)
plt.plot(df_test, color = 'orange')

plt.plot(df_train, color = 'green')
plt.plot(df_test, color = 'orange')
plt.ylabel('Rate of Change')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.show()


# ADF Test
def adfuller_test(df, target):
    adf_result = adfuller(df[target].values,
                         autolag='AIC')
    if adf_result[1] <= 0.05:
        result = f'p_value: {adf_result[1]}. Reject null hypothesis. Series is stationary'
    else:
        result = f'p_value: {adf_result[1]}. Fail to reject null hypothesis. Series is non-stationary'
    return result

adfuller_test(df_train, 'ROC')


# KPSS Test
def kpss_test(df, target):
    kpss_result = kpss(df[target].values,
                         regression='ct')
    if kpss_result[1] <= 0.05:
        result = f'p_value: {kpss_result[1]}. Reject null hypothesis. Series is non-stationary'
    else:
        result = f'p_value: {kpss_result[1]}. Fail to reject null hypothesis. Series is stationary'
    return result

kpss_test(df_train, 'ROC')

# plot autocorrelation
fig, (ax1, ax2) = plt.subplots(2)
plot_acf(df.ROC, ax=ax1)
plot_acf(df.ROC.diff().dropna(), ax=ax2)
plt.show()

# plot partial autocorrelation
fig, (ax1, ax2) = plt.subplots(2)
plot_pacf(df.ROC, ax=ax1)
plot_pacf(df.ROC.diff().dropna(), ax=ax2)
plt.show()


# Transform to stationary using differencing
df_train_diff = df_train.diff()
df_train_diff.fillna(0, inplace=True)

kpss_test(df_train_diff, 'ROC') #previously non-stationary

adfuller_test(df_train_diff, 'ROC')

df_train_diff.plot() # plot adjusted dataset to be stationary



# Determine AR(p) and MA(q)
acf_diff = plot_acf(df_train_diff)
pacf_diff = plot_pacf(df_train_diff)

y_test = df_test['ROC']
mselist= []
for q in range(0,3+1):
    for p in range(1,7+1):
        model = ARIMA(df_train_diff.ROC, order=(p,1,q))
        model_fit = model.fit()

        predict = model_fit.forecast(len(df_test))
        mselist.append(mean_squared_error(y_test, predict))
        
        print(f'ARIMA {p}1{q} -- MSE:{mean_squared_error(y_test, predict)}')



# Fit ARIMA(0,1,2)
model = ARIMA(df_train_diff.ROC, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

predict = model_fit.forecast(len(df_test))
df['forecast'] = [None]*len(df_train) + list(predict)
df[680:].plot()

plt.plot(df_train, color = 'blue')
plt.xticks(rotation=45)
plt.plot(model_fit.fittedvalues, color='red')

residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)

residuals.plot(title='residuals', ax=ax[0])
residuals.plot(title='density', ax=ax[1], kind='kde')
plt.show()

plot_acf(residuals)
plot_pacf(residuals)


# metrics
mean_squared_error(y_test, predict), r2_score(y_test, predict), mean_absolute_error(y_test, predict)
plt.show()



