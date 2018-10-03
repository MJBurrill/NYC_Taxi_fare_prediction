#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 21:02:08 2018

@author: matthewburrill
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import datetime as dt
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import os
import gc


df = pd.read_csv('../input/train.csv',nrows=21_000_000, usecols=[1,2,3,4,5,6])

print(df.head())

df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')

# Remove observations with missing values
# Since there are only a few of these, i'm not concerned with imputation
df.dropna(how='any', axis='rows', inplace=True)

# Removing observations with erroneous values
mask = df['pickup_longitude'].between(-75, -72)
mask &= df['dropoff_longitude'].between(-75, -72)
mask &= df['pickup_latitude'].between(40, 42)
mask &= df['dropoff_latitude'].between(40, 42)
#mask &= df['passenger_count'].between(0, 8)
mask &= df['fare_amount'].between(0, 300)

df = df[mask]

def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):  
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    
    return distance
    
def h_dist(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude):
    pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude = map(np.radians, [pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude])
    dlon = dropoff_longitude - pickup_longitude
    dlat = dropoff_latitude - pickup_latitude
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6367 * c
    return distance
    
def transform(data):
    # Extract date attributes and then drop the pickup_datetime column
    data['hour'] = data['pickup_datetime'].dt.hour
    data['day'] = data['pickup_datetime'].dt.day
    data['weekday'] = data['pickup_datetime'].dt.weekday
    data['month'] = data['pickup_datetime'].dt.month
    data['year'] = data['pickup_datetime'].dt.year
    data = data.drop('pickup_datetime', axis=1)

    # Distances to nearby airports, and city center
    
    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)
    tsq = (-73.9851,40.7589)
    sol = (-74.0445,40.6892)
    cbe = (-73.8603,40.8333)
 
    data['pickup_distance_to_center'] = h_dist(nyc[1], nyc[0],
                                      data['pickup_latitude'], data['pickup_longitude'])

  
    data['pickup_distance_to_jfk'] = h_dist(jfk[1], jfk[0],
                                         data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_jfk'] = h_dist(jfk[1], jfk[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_ewr'] = h_dist(ewr[1], ewr[0], 
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_ewr'] = h_dist(ewr[1], ewr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_lgr'] = h_dist(lgr[1], lgr[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_lgr'] = h_dist(lgr[1], lgr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_tsq'] = h_dist(tsq[1], tsq[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_tsq'] = h_dist(tsq[1], tsq[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_sol'] = h_dist(sol[1], sol[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_sol'] = h_dist(sol[1], sol[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_cbe'] = h_dist(cbe[1], cbe[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_cbe'] = h_dist(cbe[1], cbe[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    
    data['long_dist'] = data['pickup_longitude'] - data['dropoff_longitude']
    data['lat_dist'] = data['pickup_latitude'] - data['dropoff_latitude']
    
    data['dist'] = dist(data['pickup_latitude'], data['pickup_longitude'],
                        data['dropoff_latitude'], data['dropoff_longitude'])
                        
    data['h_dist'] = h_dist(data['pickup_longitude'], data['pickup_latitude'],
                        data['dropoff_longitude'], data['dropoff_latitude'])
                        
    
    
    return data


df = transform(df)


y = df['fare_amount']
df = df.drop(columns=['fare_amount'])


print(df.head())

x_train,x_test,y_train,y_test = train_test_split(df,y,random_state=123,test_size=0.10)

del df
del y
gc.collect()

params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': 4,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,
        'num_rounds':50000
    }
    
train_set = lgbm.Dataset(x_train, y_train, silent=False,categorical_feature=['year','month','weekday','day'])
valid_set = lgbm.Dataset(x_test, y_test, silent=False,categorical_feature=['year','month','weekday','day'])
model = lgbm.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=valid_set)
del x_train
del y_train
del x_test
del y_test
gc.collect()
                        
test_df =  pd.read_csv('../input/test.csv')

test_df['pickup_datetime'] = test_df['pickup_datetime'].str.slice(0, 16)
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
test_df = transform(test_df)
                                                                    


test_key = test_df['key']
test_df = test_df.drop(columns=['key','passenger_count'])

#Predict from test set
prediction = model.predict(test_df, num_iteration = model.best_iteration)      
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction
})

submission.to_csv('submission.csv',index=False)