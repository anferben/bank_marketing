import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# Parameters

n_splits = 5
output_file = f'rf_model.bin'


# train and predict funcionts

def train(df_train, y_train):
    train_dict = df_train[numerical + categorical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)

    model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=3)
    model.fit(X_train, y_train)

    return dv, model   


def predict(df, dv, model):
    df_dict = df[numerical + categorical].to_dict(orient='records')

    X = dv.transform(df_dict)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Data preparation

df = pd.read_csv('../data/bank-full.csv', sep=';')

df.drop_duplicates(inplace=True)
df.drop(['day', 'month', 'contact'], axis=1, inplace=True)

df.rename(columns={'y': 'success'}, inplace=True)
df.success = (df.success == 'yes').astype('int')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=7)

numerical = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']


# Validating the model

print(f'Validating the model...')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train['success'].values
    y_val = df_val['success'].values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'AUC on fold {fold} is {auc}')

    fold = fold + 1

print()
print('Vaidation results:')
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


# Training the final model

print()
print('Training the final model...')

dv, model = train(df_full_train, df_full_train['success'].values)
y_pred = predict(df_test, dv, model)

y_test = df_test['success'].values
auc = roc_auc_score(y_test, y_pred)

print(f'AUC = {auc}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print()
print(f'The model was saved to {output_file}')

