import numpy as np
import pandas as pd
import os
import sys
# sys.path.append("/kaggle/input/featureengine")
import feature_engine
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")


def multivariate_imputation_linear_regression(train, test):
    ej_train, ej_index_train = train['EJ'].tolist(), train.columns.get_loc('EJ')
    train = train.drop('EJ', axis=1)
    columns = []
    for column in train:
        columns.append(column)
    lr = LinearRegression()
    imp = IterativeImputer(estimator=lr, missing_values=np.nan, max_iter=10, verbose=0, imputation_order='roman',
                           random_state=0)

    imp.fit(train)
    train = imp.transform(train)
    train = pd.DataFrame(train, columns=columns)
    train.insert(ej_index_train, 'EJ', ej_train)

    ej_test, ej_index_test = test['EJ'].tolist(), test.columns.get_loc('EJ')
    test = test.drop('EJ', axis=1)
    columns = []
    for column in test:
        columns.append(column)

    test = imp.transform(test)
    test = pd.DataFrame(test, columns=columns)
    test.insert(ej_index_test, 'EJ', ej_test)

    return train, test


def cat_frequency_encode(train, test):
    count_enc = CountFrequencyEncoder(encoding_method='frequency', variables='EJ')
    count_enc.fit(train)
    train = count_enc.transform(train)
    test = count_enc.transform(test)
    return train, test


def standardize_features(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    final_train = pd.DataFrame(train_scaled, columns=train.columns)
    final_test = pd.DataFrame(test_scaled, columns=test.columns)
    return final_train, final_test


def pred_test(X_train, y_train, test):
    X_train, test = multivariate_imputation_linear_regression(X_train, test)
    X_train, test = cat_frequency_encode(X_train, test)
    X_train, test = standardize_features(X_train, test)
    clf = RandomForestClassifier(random_state=6)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(test)
    return y_pred


X_train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
y_train = X_train['Class']
del X_train['Class']
del X_train['Id']
del test['Id']
pred = pred_test(X_train, y_train, test)
sample_submission = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv")
sample_submission[['class_0', 'class_1']] = pred
sample_submission.to_csv('/kaggle/working/submission.csv', index=False)

