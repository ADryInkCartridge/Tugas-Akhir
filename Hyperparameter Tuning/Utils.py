import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


def r_to_py():
    string = input("Enter a string: ")
    words = []
    for word in string.split():
        if word.startswith('"') and word.endswith('"'):
            words.append(word[1:-1].replace('.', ' '))
    return words

def py_to_r(arr):
    for i in range(len(arr)):
        arr[i] = arr[i].replace(' ', '.')
        print(arr[i])
    return arr
    
def flatten(l):
    return [item for sublist in l for item in sublist]


def error_rate(methods,preds,vals):
  rmse = math.sqrt(mean_squared_error(vals,preds))
  mae = mean_absolute_error(vals, preds)
  r2 = r2_score(vals, preds)
  # print(f"RMSE : ",  round(rmse, 3))
  # print(f"MAE : ",  round(mae, 3))
  # print(f"R^2 : ",  round(r2, 3))
  
  return methods,rmse,mae,r2

def evaluate_linreg(dfX,y,kf):
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import cross_val_score
  from statistics import mean
  reg = LinearRegression()
  vals = []
  preds = []
  error_rates_linreg = []
  for train_index, test_index in kf.split(dfX):
      # print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      reg.fit(X_train, y_train)
      reg.score(X_train, y_train)

    
      pred = reg.predict(X_test)
      # rate = error_rate(pred,y_test)
      # print(rate)
      # error_rates_linreg.append(rate)
      preds.append(pred)
      vals.append(y_test)
  linear_reg_df = pd.DataFrame({'Prediction':flatten(preds),'Actual Values':flatten(vals)})
  return error_rate("Linear Regression",linear_reg_df['Prediction'],linear_reg_df['Actual Values'])
  
  # linear_reg_df
# evaluate_linreg(dfX,y)

def evaluate_dt(dfX,y,kf):
  from sklearn.tree import DecisionTreeRegressor
  dtregressor = DecisionTreeRegressor()
  vals = []
  preds = []
  error_rates_dt = []
  for train_index, test_index in kf.split(dfX):
    # print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      dtregressor.fit(X_train,y_train)
      dt_pred = dtregressor.predict(X_test)

      # print(len(preds),len(ytest))
      # rate = error_rate(dt_pred,y_test)
      # print(rate)
      # error_rates_dt.append(rate)
      preds.append(dt_pred)
      vals.append(y_test)

  dt_df = pd.DataFrame({'Prediction':flatten(preds),'Actual Values':flatten(vals)})
  # print("Mean Error Rate of 10 Folds With 3 Repeats: " + str(mean(error_rates_dt)))
  # scores = cross_val_score(dtregressor, dfX, y, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1)
  # print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )
  return error_rate("Decision Tree",dt_df['Prediction'],dt_df['Actual Values'])
  
# evaluate_randomforest(dfX,y)


def evaluate_xgboost(dfX,y,kf):
  from xgboost import XGBRegressor

  vals = []
  preds = []
  error_rates_dt = []
  xgb = XGBRegressor()

  for train_index, test_index in kf.split(dfX):
    # print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      xgb.fit(X_train,y_train)
      xgb_pred = xgb.predict(X_test)

      # print(len(preds),len(ytest))
      # rate = error_rate(xgb_pred,y_test)
      # print(rate)
      # error_rates_dt.append(rate)
      preds.append(xgb_pred)
      vals.append(y_test)

  xgb_df = pd.DataFrame({'Prediction':flatten(preds),'Actual Values':flatten(vals)})
  return error_rate("XGBoost",xgb_df['Prediction'],xgb_df['Actual Values'])
# evaluate_xgboost(dfX,y)

def evaluate_catboost(dfX,y,kf):
  from catboost import CatBoostRegressor
  from catboost import Pool
  from sklearn.metrics import r2_score, mean_squared_error
  vals = []
  preds = []
  error_rates_dt = []

  cbr = CatBoostRegressor()
  for train_index, test_index in kf.split(dfX):
    # print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      train_pool = Pool(data=X_train, label=y_train)
      test_pool = Pool(data=X_test, label=y_test)

      cbr.fit(train_pool, verbose=False)
      cbr_pred = cbr.predict(test_pool)
      preds.append(cbr_pred)
      vals.append(y_test)

  cat_df = pd.DataFrame({'Prediction':flatten(preds),'Actual Values':flatten(vals)})
  return error_rate("Catboost",cat_df['Prediction'],cat_df['Actual Values'])
# evaluate_catboost(dfX,y)
def evaluate_rf(dfX,y,kf):
  from sklearn.ensemble import RandomForestRegressor
  vals = []
  preds = []
  error_rates_dt = []

  rf = RandomForestRegressor(random_state = 0)

  for train_index, test_index in kf.split(dfX):
      X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      rf.fit(X_train,y_train)
      rf_pred = rf.predict(X_test)
      preds.append(rf_pred)
      vals.append(y_test)

  rf_df = pd.DataFrame({'Prediction':flatten(preds),'Actual Values':flatten(vals)})
  return error_rate("Random Forest",rf_df['Prediction'],rf_df['Actual Values'])

# evaluate_rf(dfX,y)

def evaluate_lightgbm(dfX,y,kf):
  import lightgbm as lgb  
  params = {
      'task': 'train', 
      'boosting': 'gbdt',
      'objective': 'regression',
      'num_leaves': 31,
      'learning_rate': 0.05,
      'metric': {'l2','l1'},
      'verbose': -1
  }
  
  vals = []
  preds = []



  for train_index, test_index in kf.split(dfX):

      X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      lgb_train = lgb.Dataset(X_train, y_train)
    #   lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

      lgb_reg = lgb.train(params,
                  train_set=lgb_train)
      
      lgb_pred = lgb_reg.predict(X_test)
      preds.append(lgb_pred)
      vals.append(y_test)

  lgb_df = pd.DataFrame({'Prediction':flatten(preds),'Actual Values':flatten(vals)})
  return error_rate("LightGBM",lgb_df['Prediction'],lgb_df['Actual Values'])
# evaluate_lightgbm(dfX,y)

def evaluate_svr(dfX,y,kf):
  from sklearn.svm import SVR

  vals = []
  preds = []
  error_rates_dt = []

  svr_reg = SVR()
  for train_index, test_index in kf.split(dfX):
      X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      svr_reg.fit(X_train,y_train)
      svr_pred = svr_reg.predict(X_test)
      preds.append(svr_pred)
      vals.append(y_test)

  svr_df = pd.DataFrame({'Prediction':flatten(preds),'Actual Values':flatten(vals)})
  return error_rate("SVR",svr_df['Prediction'],svr_df['Actual Values'])

def eval_models(dfX,y,kf):
  evals = []
  evals.append(evaluate_linreg(dfX,y,kf))  
  evals.append(evaluate_svr(dfX,y,kf))  
  evals.append(evaluate_dt(dfX,y,kf))  
  evals.append(evaluate_rf(dfX,y,kf))  
  evals.append(evaluate_xgboost(dfX,y,kf))  
  evals.append(evaluate_catboost(dfX,y,kf))  
  evals.append(evaluate_lightgbm(dfX,y,kf))  
  dfEval = pd.DataFrame(evals,columns=["Method","RSME","MAE","R2"])
  return dfEval

def compare_models(x1,x2,y,kf):
    dfEval_1 = eval_models(x1,y,kf)
    dfEval_2 = eval_models(x2,y,kf)
    dfDiff = dfEval_1.copy()
    dfDiff["RSME Diff"] = dfEval_1["RSME"] - dfEval_2["RSME"]
    dfDiff["MAE Diff"] = dfEval_1["MAE"] - dfEval_2["MAE"]
    dfDiff["R2 Diff"] = dfEval_1["R2"] - dfEval_2["R2"]
    dfDiff.drop(["Method","RSME","MAE","R2"],axis=1,inplace=True)
    return pd.concat([dfEval_1,dfEval_2,dfDiff],axis=1)
