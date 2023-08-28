# general setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import calendar

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, explained_variance_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

import warnings
warnings.filterwarnings('ignore')


# load dataset
df = pd.read_csv('df_to_train_final.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.columns.to_list()




""" ElasticNet (no cross-validation)  """
'''================== TRAIN TEST SPLIT =================='''
data_train, data_test = train_test_split(df, random_state=89, train_size=0.8)

data_X_train = data_train.drop(columns=['target_price'])
data_X_test  = data_test.drop(columns=['target_price'])
data_y_train = data_train['target_price']
data_y_test  = data_test['target_price']

    # fit model and predict target-price
model_en = ElasticNet(alpha=0.5, l1_ratio=0.2, max_iter=800, random_state=89)

result = model_en.fit(data_X_train, data_y_train)
    
y_pred_train = model_en.predict(data_X_train)
y_pred_test = model_en.predict(data_X_test)
    
mse_train = mean_squared_error(data_y_train, y_pred_train)
mse_test = mean_squared_error(data_y_test, y_pred_test)
    
print(f'MSE score train for {model_en.__class__.__name__}: {mse_train}')
print(f'MSE score test for {model_en.__class__.__name__}: {mse_test}')




""" ElasticNet (with cross-validation) """
# create new dataframe for cross validation
data = data_train
X = data_X_train
y = data_y_train

# set parameters for kfold
random_state = 89 
n_splits = 5

kfold = KFold(n_splits = n_splits, random_state=random_state, shuffle=True)

params = {'alpha':0.5, 
          'l1_ratio':0.2, 
          'max_iter':800, 
          'random_state':89}

# store results of folds
model_list = []

# loop through each fold
for i, (train_index, test_index) in enumerate(kfold.split(data)):
    
    # model for each fold
    model = ElasticNet(**params)
    
    # new train_test for each fold
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy() 
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
    
    print(f'======================== Fold {i+1}/{n_splits} ========================')
    
    # fit model and predict
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    
    # calculate MSE
    mse_test = mean_squared_error(y_test, y_hat)
    print(f'MSE for fold {i+1}: {mse_test} ')
    
    # store folds model
    model_list.append(model)

def mean_y_hat(X, model_list):
    total_y_hat = np.zeros(X.shape[0])
    
    # loop through each model and predict
    for model in model_list:
        y_hat = model.predict(X)
        total_y_hat += y_hat
    
    # find avg of all y predicted from 5 models
    mean_y_hat = total_y_hat/len(model_list)
    return mean_y_hat

mean_y_hat_train = mean_y_hat(data_X_train, model_list)

# add this to df_valid 
mean_y_hat_test = mean_y_hat(data_X_test, model_list)

mse_test = mean_squared_error(data_y_test, mean_y_hat_test)
mse_test

df_valid = data_test.copy()
df_valid['y_pred'] = mean_y_hat_test

df_valid.head()



""" TRADING """
df_date = pd.read_csv('df_feed.csv')
df_date

df['date'] = df_date['Date']
df['date'] = pd.to_datetime(df['date'])
df.head()

weekday_list = []
week_list = []

for i in range(len(df)):
    weekday_list.append(calendar.day_name[df['date'][i].weekday()])
    week_list.append(df['date'][i].isocalendar().week)
    
df['weekday'] = weekday_list
df['week'] = week_list

df

# separate by week
weeknum = [1]
count = 1

for i in range(1, len(df)):
    #print(df['weekday'][i])
    
    if df['week'][i] == df['week'][i-1]:
        weeknum.append(count)
        #print(count)
        
    else:
        count+=1
        weeknum.append(count)
        #print(count)
df['weekInt'] = weeknum

df_new = df.copy()
df_new.drop_duplicates(subset = 'weekInt', keep = 'last', inplace=True)
df_new.head(5)
df.head(5)

monday = df.copy()
monday.drop_duplicates(subset = 'weekInt', keep = 'first', inplace=True)
monday.head(5)

df_new.drop(columns=['weekday', 'week', 'weekInt'], inplace=True)
df_train = df_new[:139]
df_test = df_new[139:]
df_test

df_testX = df_test.drop(columns=['target_price', 'date'])
df_testy = df_test['target_price']
df_testX.shape, df_testy.shape

trading_y_hat = model_en.predict(df_testX)
trading_df = pd.DataFrame({'real_price':monday['target_price'][139:],
                          'forecast':trading_y_hat})
trading_df

decision = []
for i in range(len(trading_df)):
    if trading_df['forecast'].to_list()[i] > trading_df['real_price'].to_list()[i]:
        decision.append(1)
    else:
        decision.append(0)
trading_df['decision'] = decision
trading_df


# start trading
money =10000
for i in range(len(trading_df)):
    if trading_df['decision'].to_list()[i] == 1:
        stock = np.round(money/trading_df['real_price'].to_list()[i],0)
        print(f'At week {i+1}, stocks brought: {stock} ')
        
        if i == 10:
            money = np.round(stock*trading_df['real_price'].to_list()[i],4)
        else:
            money = np.round(stock*trading_df['real_price'].to_list()[i+1],4)
        print(f'Money earned: ${money:,}')
        print()
    else:
        print(f'At week {i+1}, stocks brought: 0')
        money=money
        print(f'Money earned: ${money:,}')
        print()

