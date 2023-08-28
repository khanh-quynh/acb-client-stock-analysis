# general setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
import warnings
warnings.filterwarnings('ignore')

# load dataset
df = pd.read_csv('df_to_train.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.info()

# add column for previous_price
prev_list = df['target_price'].to_list()
prev_list.insert(0, 0)
prev_list=prev_list[:-1]

df['prev_price'] = prev_list



'''================== TRAIN TEST SPLIT =================='''
data_train, data_test = train_test_split(df, random_state=89, train_size=0.8)

data_X_train = data_train.drop(columns=['target_price'])
data_X_test  = data_test.drop(columns=['target_price'])
data_y_train = data_train['target_price']
data_y_test  = data_test['target_price']

compare = []



'''================== ElasticNet (dont use) =================='''
model_en = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=89)

model_en.fit(data_X_train, data_y_train)
    
y_pred_train = model_en.predict(data_X_train)
y_pred_test = model_en.predict(data_X_test)
    
mse_train = mean_squared_error(data_y_train, y_pred_train)
mse_test = mean_squared_error(data_y_test, y_pred_test)
    
compare.append(mse_test)
    
print(f'MSE score train for {model_en.__class__.__name__}: {mse_train}')
print(f'MSE score test for {model_en.__class__.__name__}: {mse_test}')



'''================== LinearRegression =================='''
model_lr = LinearRegression()
model_lr.fit(data_X_train, data_y_train)
    
y_pred_train = model_lr.predict(data_X_train)
y_pred_test = model_lr.predict(data_X_test)
    
mse_train = mean_squared_error(data_y_train, y_pred_train)
mse_test = mean_squared_error(data_y_test, y_pred_test)
    
compare.append(mse_test)    
    
print(f'MSE score train for {model_lr.__class__.__name__}: {mse_train}')
print(f'MSE score test for {model_lr.__class__.__name__}: {mse_test}')



'''================== Ridge (dont use) =================='''
model_ridge = Ridge(alpha=2, random_state=89)
model_ridge.fit(data_X_train, data_y_train)
    
y_pred_train = model_ridge.predict(data_X_train)
y_pred_test = model_ridge.predict(data_X_test)
    
mse_train = mean_squared_error(data_y_train, y_pred_train)
mse_test = mean_squared_error(data_y_test, y_pred_test)

compare.append(mse_test)

print(f'MSE score train for {model_ridge.__class__.__name__}: {mse_train}')
print(f'MSE score test for {model_ridge.__class__.__name__}: {mse_test}')




"""=============================== GradientBoostingRegressor ==============================="""
# no tuning
model_gb = GradientBoostingRegressor(random_state=89)

model_gb.fit(data_X_train, data_y_train)

y_pred_train = model_gb.predict(data_X_train)
y_pred_test = model_gb.predict(data_X_test)

mse_train = mean_squared_error(data_y_train, y_pred_train)
mse_test = mean_squared_error(data_y_test, y_pred_test)

print(f'MSE score train for {model_gb.__class__.__name__}: {mse_train}')
print(f'MSE score test for {model_gb.__class__.__name__}: {mse_test}')

param_grid = {
    'learning_rate':[0.01,0.05,0.1,0.2],
    'n_estimators':range(30,71,10),
    'min_samples_split':range(300,1100,100),
    'max_depth':range(4,11,1),
    'subsample':[0.4,0.5,0.6,0.7,0.8]
}

search = GridSearchCV(model_gb, param_grid, refit=True)
search.fit(data_X_train, data_y_train)

search.best_estimator_

# tuning
model_gb = GradientBoostingRegressor(max_depth=7, min_samples_split=300, 
                                     n_estimators=70,
                                     random_state=89, subsample=0.8)

model_gb.fit(data_X_train, data_y_train)

y_pred_train = model_gb.predict(data_X_train)
y_pred_test = model_gb.predict(data_X_test)

mse_train = mean_squared_error(data_y_train, y_pred_train)
mse_test = mean_squared_error(data_y_test, y_pred_test)

compare.append(mse_test)

print(f'MSE score train for {model_gb.__class__.__name__}: {mse_train}')
print(f'MSE score test for {model_gb.__class__.__name__}: {mse_test}')



"""=============================== SVR ==============================="""
model_svr = SVR(C=100, gamma=1)
model_svr.fit(data_X_train, data_y_train)
    
y_pred_train = model_svr.predict(data_X_train)
y_pred_test = model_svr.predict(data_X_test)
    
mse_train = mean_squared_error(data_y_train, y_pred_train)
mse_test = mean_squared_error(data_y_test, y_pred_test)

compare.append(mse_test)

print(f'MSE score train for {model_svr.__class__.__name__}: {mse_train}')
print(f'MSE score test for {model_svr.__class__.__name__}: {mse_test}')

param_grid = {
    'C':[0.01,0.1,1,10,100,1000],
    'gamma':[1,0.1,0.01,0.001, 0.0001]}

grid = GridSearchCV(SVR(), param_grid, refit=True)
grid.fit(data_X_train, data_y_train)

grid.best_estimator_




# wrap up:
print(compare)

df.to_csv('df_to_train_final.csv')





