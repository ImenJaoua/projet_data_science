import os, sys

sys.path.insert(0, os.getcwd()) 
from src.client.comet import add_metrics
from src.visualizations.visualization import plots
from src.client import start_experiment
from src.features import load_df_shots
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


df_2016 = load_df_shots(2016)
df_2017 = load_df_shots(2017)
df_2018 = load_df_shots(2018)
df_2019 = load_df_shots(2019)

# len(pd.concat([df_2016, df_2017, df_2018]))

df = pd.concat([df_2016, df_2017, df_2018]).reset_index(drop=True)
df.Shot_angle = df.Shot_angle.abs()
df_2019.Shot_angle = df_2019.Shot_angle.abs()

# print(len(pd.concat([df_2016, df_2017, df_2018])) / len(df))

# X_train, X_val, y_train, y_val = train_test_split(df.Shot_distance, df.Goal, test_size=0.23, random_state=42)

X_train = df[['Shot_distance','Shot_angle']]
# X_train = df[['Shot_angle']]
y_train = df.Goal
X_val = df_2019[['Shot_distance','Shot_angle']]
# X_val = df_2019[['Shot_angle']]
y_val = df_2019.Goal
val_labels = y_val.values.reshape(-1,1)

bst = XGBClassifier()
bst.fit(X_train, y_train)

preds = bst.predict(X_val)
probs = bst.predict_proba(X_val)[:,1]

# accuracy_score(y_val, preds)
# roc_auc_score(y_val, probs)
# precision_score(y_val, preds)
# recall_score(y_val, preds)

df = pd.concat([df_2016, df_2017, df_2018, df_2019]).reset_index(drop=True)
X_train = df[['Shot_distance','Shot_angle']].abs()
y_train = df.Goal

bst = XGBClassifier()
bst.fit(X_train, y_train)
bst.save_model('model/xgb.model.json')

# print(y_val.values)

exp = start_experiment()
exp.set_name(f'xgd-distance-angle-{exp.get_name()}')
add_metrics(
        exp,
        accuracy_score(y_val, preds),
        roc_auc_score(y_val, probs),
        precision_score(y_val, preds),
        recall_score(y_val, preds))
exp.log_model("xgb", "model/xgb.model.json")
exp.add_tags(['XGB', 'Distance', 'Angle'])
plots(val_labels, probs, f'xgd-distance-angle', exp)

# plots(val_labels, probs, f'xgd-distance')