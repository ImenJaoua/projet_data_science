import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())

import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.client.comet import add_metrics
from src.client import start_experiment
from src.visualizations.visualization import plots
from src.features import load_df_shots
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, precision_score, recall_score
import pickle

df_2016 = load_df_shots(2016)
df_2017 = load_df_shots(2017)
df_2018 = load_df_shots(2018)
df_2019 = load_df_shots(2019)
df_2020 = load_df_shots(2020)

df_po = pd.read_pickle('2020-playoffs.pkl')

# columns = ['Shot_distance']
# columns = ['Shot_angle']
columns = ['Shot_distance', 'Shot_angle']

# df_train = pd.concat([df_2016, df_2017, df_2018]).reset_index(drop=True)
# x_tval = df_train[columns].abs()
# y_tval = df_train.Goal

# x_val = df_2019[columns].abs()
# y_val = df_2019.Goal

df_train = pd.concat([df_2016, df_2017, df_2018, df_2019]).reset_index(drop=True)
x_train = df_train[columns].abs()
y_train = df_train.Goal

# x_test = df_2020[columns].abs()
# y_test = df_2020.Goal

x_test = df_po[columns].abs()
y_test = df_po.Goal

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)

y_pred = np.insert(y_pred, 0, [0,0], axis=0)
np.savetxt("LR_da_p.csv", y_pred, delimiter=",")
# pickle.dump(clf, open('model/lr-model.pkl', 'wb'))

# df_train = pd.concat([df_2016, df_2017, df_2018]).reset_index(drop=True)
# x_train = df_train[columns].abs()
# y_train = df_train.Goal

# clf = LogisticRegression()
# clf.fit(x_train, y_train)
# preds = clf.predict(x_val)
# probs = clf.predict_proba(x_val)[:,1]

# accuracy_score(y_val, preds)
# roc_auc_score(y_val, probs)
# precision_score(y_val, preds)
# recall_score(y_val, preds)

# plots(y_val, probs, 'xgb-all')

# exp = start_experiment()
# exp.set_name(f'lr-{exp.get_name()}')
# exp.add_tags(['LR', 'Distance', 'Angle', 'Final'])
# add_metrics(
#         exp,
#         accuracy_score(y_val, preds),
#         roc_auc_score(y_val, probs),
#         precision_score(y_val, preds),
#         recall_score(y_val, preds))

# exp.log_model("lr-da", "model/lr-model.pkl")
# plots(y_val, probs, 'LR Distance and Angle', exp)







