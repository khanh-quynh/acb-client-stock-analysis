# general setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, Ridge

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
import warnings
warnings.filterwarnings('ignore')

# load dataset
df = pd.read_csv('df_feed.csv')
df.drop(columns=['Date'], inplace=True)
df.head(5)

df['GIA_DIEUCHINH'].plot()
plt.xticks(rotation=45)
plt.show()

# rescale except target column
df_to_scale = df.drop(columns=['GIA_DIEUCHINH'])
df_to_scale

    # set scaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_to_scale)

    # print scaled dataframe
df_scaled = pd.DataFrame(data = scaled,
                        index = df.index,
                        columns = df_to_scale.columns.to_list())

    # add "target" column back to dataframe
df_scaled['target_price'] = df['GIA_DIEUCHINH']

    # print result
df_scaled



''' ===================== Filter method: p_value ===================== '''
# set X and y for OLS
X_ols = df_scaled.drop(columns=['target_price'])
y_ols = df_scaled['target_price']

# add constant
X_ols = sm.add_constant(X_ols)

# fit model and result
model_ols = sm.OLS(y_ols, X_ols)
result = model_ols.fit()

# df of all p_values
pval_df = pd.DataFrame(result.pvalues)
pval_df.rename(columns={0: 'p_value'}, inplace=True)
pval_df = pval_df.iloc[1:,:]
pval_df.head()

result.summary()

feature_drop_pval = pval_df[pval_df['p_value'] > 0.25].index.to_list()
feature_drop_pval, len(feature_drop_pval)

df_scaled = df_scaled.drop(columns=feature_drop_pval)
df_scaled.shape



''' ===================== L1 regularization ===================== '''
# train test split for Lasso
X = df_scaled.drop(columns=['target_price'])
y = df_scaled['target_price']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=89)

# fit Lasso() model 
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# get coefficient
selected_feature = lasso.coef_
len(selected_feature.tolist()), X_train.shape

# create dataframe of coefficient after L1 regularization
lasso_feature_df = pd.DataFrame({'feature':X_train.columns.to_list(),
                                 'coefficient':selected_feature.tolist()})

# get list of feature with coefficient = 0
feature_0coef = lasso_feature_df[lasso_feature_df['coefficient'] == 0]['feature'].to_list()
feature_0coef

# remove feature with 0 coefficient from df_scaled
df_scaled.drop(columns=feature_0coef, inplace=True)
df_scaled



''' ===================== RFE ===================== '''
# independent and target variables
X = df_scaled.drop(columns=['target_price'])
y = df_scaled['target_price']

# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=89)

# model and fit
model_lr = LinearRegression()
model_lr.fit(X, y)

# RFE for feature selection
rfe_method = RFE(model_lr, n_features_to_select=1, step=1)
rfe_method.fit(X_train, y_train)

# predict values
y_pred = rfe_method.predict(X_test)

imp = rfe_method.ranking_.tolist()
feature= X.columns.to_list()

imp_df = pd.DataFrame({'feature':feature, 'importance':imp}).sort_values(by='importance')
imp_df

# Plot MSE
mse_list = []
no_feature_list = []

for i in range(2,34):
    #create list of features to be used in the model
    feature_list = imp_df['feature'].to_list()[0:i]
    
    # train test split
    X = df_scaled[feature_list]
    y = df_scaled['target_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    
    # fit model and predict target price
    model1 = LinearRegression()
    model1.fit(X_train, y_train)
    
    y_pred = model1.predict(X_test)
    
    # mse as metrics for model evaluation
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

    no_feature_list.append(len(feature_list))

plt.plot(no_feature_list, mse_list)
plt.ylabel('Mean Squared Error')
plt.xlabel('Number of Features')

imp_df['MSE'] = mse_list
imp_df

# choose features at the diminishing point of the curve
final_feature = imp_df['feature'][:11].to_list()
final_feature

# drop features that if we add more doesn't improve the model
drop = imp_df['feature'][11:].to_list()
drop

# use this dataframe to train
df_to_train = df_scaled.drop(columns=drop)
df_to_train

df_to_train.to_csv('df_to_train.csv')

