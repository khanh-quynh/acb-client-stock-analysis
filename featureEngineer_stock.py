# general setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# load dataset
df = pd.read_excel('Data_HPG.xlsx')
df.rename(columns={'Ngày':'Date'}, inplace=True)
df.set_index('Date', inplace=True)
df.drop(columns=['Data'], inplace=True)

df.shape

drop_col = []

for i in df.columns.to_list():
    if i[:6]=='Target':
        drop_col.append(i)

df.drop(columns=drop_col, inplace=True)

df.shape

# Missing values
null_df = pd.DataFrame(df.isna().sum())
null_df.rename(columns={0:'count'}, inplace=True)
null_df[null_df['count']!=0]

df['next_t5_price'].fillna(np.mean(df['next_t5_price']), inplace=True)
df['next_t10_price'].fillna(np.mean(df['next_t10_price']), inplace=True)
df['next_t15_price'].fillna(np.mean(df['next_t15_price']), inplace=True)

# Binary variables
unique_df = pd.DataFrame(df.nunique())
unique_df.rename(columns={0:'count'}, inplace=True)
binary_var = unique_df[unique_df['count']==2].index.to_list()
len(binary_var)

small_nunique_df = unique_df[(unique_df['count']<16) & (unique_df['count']>2)]
df.groupby('KLDK_MUA_BATDAU_CDNB').count()

zero_list = ['KLDK_MUA_BATDAU_CDNB', 
             'KLDK_MUA_KETTHUC_CDNB', 
             'KLDK_BAN_BATDAU_CDNB', 
             'KL_MUA_THUCHIEN_CDNB', 
             'KLDK_BAN_KETTHUC_CDNB']

for zero in zero_list:
    new = []
    for i in df[zero]:
        if i == 0:
            new.append(0) 
        else:
            new.append(1)
    df[zero]=new
df.groupby('KL_MUA_THUCHIEN_CDNB').count()

# Outlier
outlier_list = unique_df[unique_df['count']>=16]
outlier_list = outlier_list.iloc[1:,:]
outlier_list.head(5)

for i in outlier_list.index.to_list():
    
    # lower and upper limit
    lower = np.percentile(df[i],1)
    upper = np.percentile(df[df[i]>0][i],99)
    
    # adjust outliers
    adj_list = []
    
    for j in df[i]:
        if j <= lower:
            adj_list.append(lower)
        elif j >= upper:
            adj_list.append(upper)
        else:
            adj_list.append(j)
    
    df[i] = adj_list

# Filter method: correlation, p_value
corr_maxtrix = df.corr().abs()
target_corr_df = corr_maxtrix['GIA_DIEUCHINH'].to_frame().reset_index()
target_corr_df.rename(columns={'index':'feature', 'GIA_DIEUCHINH':'corr_y'}, inplace=True)
target_corr.sort_values(by=['corr_y'], ascending=False)
target_corr_df

target_corr_df2 = target_corr_df.copy()
target_corr_df2.rename(columns={'feature':'feature2'}, inplace=True)
target_corr_df2

x_list = outlier_list.index.to_list()

def corr_iv(df, list, thres):
    
    #create correlation df with 1.o's diagonally
    corr_df = df[list].corr().abs()
    
    #correlation df: 2 columns of variables
    corr_lst = []
    for c in corr_df.columns:
        for r in corr_df.index:
            if c != r and corr_df[c][r] >= thres:
                corr_lst.append({"feature":c, "feature2":r, "corr_xx": corr_df[c][r]})

    corr_df2 = pd.DataFrame(corr_lst)

    #merge corr_y into correlation df
    corr_merged = corr_df2.merge(target_corr_df, on='feature').merge(target_corr_df2, on='feature2')
    
    #add "drop" column
    corr_merged['drop'] = [0]*len(corr_merged)
    for i in range(len(corr_merged)):
        if corr_merged['corr_y_x'][i] < corr_merged['corr_y_y'][i]:
            corr_merged['drop'][i] = corr_merged['feature'][i]
        else:
            corr_merged['drop'][i] = corr_merged['feature2'][i]
            
    
    #list of variables to be dropped
    drop_list = corr_merged['drop'].to_list()
    drop_list=[*set(drop_list)]
    
    return drop_list

drop_list = corr_iv(df, x_list, 0.8)

new_df = df.drop(columns=drop_list)
new_df.shape

new_df.to_csv('df_feed.csv')
