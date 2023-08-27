# general setup
import pandas as pd
from pandas.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import calendar

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
pd.options.display.max_rows = 1000



# load dataset
df_draft = pd.read_excel('Data_HPG.xlsx')
df_draft['Ngày'] = pd.to_datetime(df_draft['Ngày'])
df_draft.head(5)

df = pd.DataFrame({
    'Date':df_draft['Ngày'],
    'Price':df_draft['GIA_DIEUCHINH']})
df.head()



# Add columns: week order in a year, total weeks in df
''' create dataframe with weekday and week "order" in a year '''
weekday_list = []
week_list = []

for i in range(len(df)):
    weekday_list.append(calendar.day_name[df['Date'][i].weekday()])
    week_list.append(df['Date'][i].isocalendar().week)
    
df['weekday'] = weekday_list
df['week'] = week_list
df.set_index('Date', inplace=True)
df.head(5)

'''    count number of week in THIS dataframe     '''
weeknum = [1]
count = 1

    # loop through each date
for i in range(1, len(df)):
    
    # if this week order is same as last week, count doesn't increase
    if df['week'][i] == df['week'][i-1]:
        weeknum.append(count)
    # if this week order of the year is not the same between two dates, they are 2 different weeks -> increase count += 1 
    else:
        count+=1
        weeknum.append(count) 
df['weekInt'] = weeknum
df.head(5)
df_new = df.copy()

'''  get Friday values'''
df_new.drop_duplicates(subset = 'weekInt', keep = 'last', inplace=True)

df_new.drop(columns=['weekday', 'week', 'weekInt'], inplace=True)
df_train = df_new[:139]
df_test = df_new[139:]


# Check for stationary
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

    # test stationary using adfuller
def adfuller_test(df, target):
    adf_result = adfuller(df[target].values,
                         autolag='AIC')
    if adf_result[1] <= 0.05:
        result = f'p_value: {adf_result[1]}. Reject null hypothesis. Series is stationary'
    else:
        result = f'p_value: {adf_result[1]}. Fail to reject null hypothesis. Series is non-stationary'
    return result

adfuller_test(df_train, 'Price')

    # test stationary usig kpss
def kpss_test(df, target):
    kpss_result = kpss(df[target].values,
                         regression='ct')
    if kpss_result[1] <= 0.05:
        result = f'p_value: {kpss_result[1]}. Reject null hypothesis. Series is non-stationary'
    else:
        result = f'p_value: {kpss_result[1]}. Fail to reject null hypothesis. Series is stationary'
    return result

kpss_test(df_train, 'Price')

    # take difference one time to convert to stationary series
df_train_diff = df_train.diff()
df_train_diff.fillna(0, inplace=True)

print(kpss_test(df_train_diff, 'Price')) #previously non-stationary
print(adfuller_test(df_train_diff, 'Price'))


# Use ACF/ PACF to decide ARIMA(p,d,q)
df_train_diff.plot()
acf_diff = plot_acf(df_train_diff)
pacf_diff = plot_pacf(df_train_diff)


# Fit ARIMA with significant lags from ACF and PACF
model = ARIMA(df_train_diff.Price, order=(14,1,21))
model_fit = model.fit()
print(model_fit.summary())


# take difference for test dataframe
df_test_diff = df_test.diff()
df_test_diff.fillna(0, inplace=True)
df_test_diff.plot()


# predict values 
predict_val = model_fit.forecast(len(df_test_diff)).to_list()
decision=[]


# decision buy or not: buy if stock is predicted to increase, otherwise, deicison = 0 =  do not buy 
for i in predict_val:
    if i > 0.0 :
        decision.append(1)
    else:
        decision.append(0)

df_test_diff['forecast'] = list(predict_val) 
df_test_diff['decision'] = decision
df_test_diff


# add decision column to original dataset (3 years value)
df_new['decision'] = ['None']*len(df_train) + decision
df_new.tail()


''' ===================== TRADING ===================== '''
# trade on the last 10 weeks
trade_df = df_new[139:]
trade_df

# trading algorithm

# initial investment
money =10000

# loop through 10 weeks of trading
for i in range(len(trade_df)):
    
    # buy if decision=1
    if trade_df['decision'][i] == 1:
        stock = np.round(money/trade_df['Price'][i],0)
        print(f'At week {i+1}, stocks brought: {stock} ')
        if i == 10:
            money = np.round(stock*trade_df['Price'][i],4) # code for last week of trading: money earned = whatever remaining
        else:
            money = np.round(stock*trade_df['Price'][i+1],4)
        print(f'Money earned: ${money:,}')
        print()
    
    # do not buy if decision=0
    else:
        print(f'At week {i+1}, stocks brought: 0')
        money=money # money does not change if no transaction happened
        print(f'Money earned: ${money:,}')
        print()

profit = np.round((13048.85-10000)*100/10000, 2)

print(f'Using this model for 10 weeks for trading, profit earned: {profit}%')

