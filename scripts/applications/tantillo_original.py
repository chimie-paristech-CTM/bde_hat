import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

#define the scaling for descriptors
scaler = MinMaxScaler()

#load the data sets
df=pd.read_csv('data/input_tantillo.csv', index_col=0)
df_train=df[df.Set=='Training']
df_test=df[df.Set=='Test']

#define the response variables
y_train=df_train['DFT_Barrier']
y_test=df_test['DFT_Barrier']

#define the predictor variables
#features = ['BDE', 'BDEfr', 'forceconstant', 'bondorder', 'charge_C', 'alpha_C']
#features = ['BDEfr', 'forceconstant', 'alpha_C']
features = ['fr_BDE']
X_train=df_train[features]
X_test=df_test[features]

#scale the predictor variables according to the fit on the training set
X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])

#add a constant of 1 to every record for the regression in sm
X_train_lm = sm.add_constant(X_train, has_constant='add')
X_test_lm = sm.add_constant(X_test, has_constant='add')

#define and fit the model
lr_1 = sm.OLS(y_train, X_train_lm).fit()

#compute the training set MAE
y_pred=lr_1.predict(X_train_lm)
train_MAE=mean_absolute_error(y_train, y_pred)
del y_pred

#compute the test set MAE and R-squared
y_pred=lr_1.predict(X_test_lm)
test_MAE=mean_absolute_error(y_test, y_pred)
test_rsquared=r2_score(y_test, y_pred)

#compute all the predicted outcomes to record them
frames=[X_train_lm, X_test_lm]
saved_pred=lr_1.predict(sm.add_constant(pd.concat(frames)))

#print summary data
print("\n")
print(lr_1.summary())
print("\n")
print("Training Set R-squared: " +str(lr_1.rsquared_adj))
print("Training Set MAE: " +str(train_MAE))
print("Test Set R-squared: " +str(test_rsquared))
print("Test Set MAE: " +str(test_MAE))
print("\n")
#print(saved_pred.sort_index())

#if the number of features is one, exit as VIF will error out
#if (len(features) == 1):
#    exit()

#compute the Variance Inflation Factors
#vif = pd.DataFrame()
#vif['Features'] = X_train.columns
#vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
#vif['VIF'] = round(vif['VIF'], 2)
#vif = vif.sort_values(by = "VIF", ascending = False)
#print(vif)
#print("\n")

#cross validate on all 24 records
#setup the model
model = LinearRegression()

#use the whole data set
y=df['DFT_Barrier']
X=df[features]
X = scaler.fit_transform(X)

#set up metrics to examine
scoring = {'MAE': 'neg_mean_absolute_error', 'R2':'r2'}

#setup the cross validation
cv = RepeatedKFold(n_splits=3, n_repeats=10)
scores = cross_validate(model, X, y, scoring=scoring, cv=cv, return_train_score=True)

#return the metrics
print(-1*np.mean(scores['test_MAE']))
print(np.mean(scores['test_R2']))

