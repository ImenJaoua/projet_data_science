import os, sys

sys.path.insert(0, os.getcwd()) 
from src.client.comet import add_metrics
from src.visualizations.visualization import plots
from src.client import start_experiment
from src.features import load_df_shots, add_goalie_ratio, add_opponent_concedes, add_shooter_ratio, add_team_goals
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, minmax_scale
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, precision_score, recall_score
from keras import metrics


df_2016 = load_df_shots(2016)
df_2017 = load_df_shots(2017)
df_2018 = load_df_shots(2018)
df_2019 = load_df_shots(2019)
df_2020 = load_df_shots(2020)

df = pd.concat([df_2016, df_2017, df_2018, df_2019, df_2020]).reset_index(drop=True)
# df = pd.concat([df_2016, df_2017, df_2018, df_2019]).reset_index(drop=True)

df = add_shooter_ratio(df)
df = add_goalie_ratio(df)
df = add_team_goals(df)
df = add_opponent_concedes(df)

df.Shot_angle = df.Shot_angle.abs()
df['Powp'] = df.Players - df.Opp_players
df.loc[df.Powp < 0, 'Powp'] = 0

df_tot = df.drop(['Strength','Time'], axis=1).copy()

enc = OrdinalEncoder()

df_tot.X = minmax_scale(df_tot.X.values)
df_tot.Y = minmax_scale(df_tot.Y.values)
df_tot.Previous_x = minmax_scale(df_tot.Previous_x.values)
df_tot.Previous_y = minmax_scale(df_tot.Previous_y.values)
df_tot.Previous_time_since = minmax_scale(df_tot.Previous_time_since.values)
df_tot.Team_goals = minmax_scale(df_tot.Team_goals.values)
df_tot.Opp_concedes = minmax_scale(df_tot.Opp_concedes.values)
df_tot.Team = enc.fit_transform(df_tot.Team.values.reshape(-1,1)).reshape(-1)
df_tot.OppTeam = enc.fit_transform(df_tot.OppTeam.values.reshape(-1,1)).reshape(-1)
df_tot.Shooter = enc.fit_transform(df_tot.Shooter.values.reshape(-1,1)).reshape(-1)
df_tot.Goalie = enc.fit_transform(df_tot.Goalie.values.reshape(-1,1)).reshape(-1)

df_tot.Game_time = minmax_scale(df_tot.Game_time.values)
df_tot.Previous_distance = minmax_scale(df_tot.Previous_distance.values)
df_tot.Speed = minmax_scale(df_tot.Speed.values)
df_tot.Time_since_powp = minmax_scale(df_tot.Time_since_powp.values)
df_tot.Shot_distance = minmax_scale(df_tot.Shot_distance.values)
df_tot.Shooter_ratio = minmax_scale(df_tot.Shooter_ratio.values)
df_tot.Goalie_ratio = minmax_scale(df_tot.Goalie_ratio.values)
df_tot.Powp = minmax_scale(df_tot.Powp.values)
df_tot.Shot_angle = minmax_scale(df_tot.Shot_angle.values)
df_tot.Rebound_angle = minmax_scale(df_tot.Rebound_angle.values)

df_train = df_tot[df_tot.Year < 2019].drop('Year', axis=1)
df_val = df_tot[df_tot.Year == 2019].drop('Year', axis=1)
df_test = df_tot[df_tot.Year == 2020].drop('Year', axis=1)

type_enc = df_train.groupby('Type').Goal.mean().reset_index(name='Type_enc')
df_train = df_train.drop('Type_enc', axis=1, errors='ignore')
df_train = df_train.merge(type_enc[['Type', 'Type_enc']], how='left', on=['Type'])
df_train = df_train.drop('Type', axis=1, errors='ignore')
df_val = df_val.drop('Type_enc', axis=1, errors='ignore')
df_val = df_val.merge(type_enc[['Type', 'Type_enc']], how='left', on=['Type'])
df_val = df_val.drop('Type', axis=1, errors='ignore')
df_test = df_test.drop('Type_enc', axis=1, errors='ignore')
df_test = df_test.merge(type_enc[['Type', 'Type_enc']], how='left', on=['Type'])
df_test = df_test.drop('Type', axis=1, errors='ignore')

Pe_enc = df_train.groupby('Previous_event_type').Goal.mean().reset_index(name='Pe_enc')
df_train = df_train.drop('Pe_enc', axis=1, errors='ignore')
df_train = df_train.merge(Pe_enc[['Previous_event_type', 'Pe_enc']], how='left', on=['Previous_event_type'])
df_train = df_train.drop('Previous_event_type', axis=1, errors='ignore')

df_val = df_val.drop('Pe_enc', axis=1, errors='ignore')
df_val = df_val.merge(Pe_enc[['Previous_event_type', 'Pe_enc']], how='left', on=['Previous_event_type'])
df_val = df_val.drop('Previous_event_type', axis=1, errors='ignore')
df_val.Pe_enc.fillna(df_val.Pe_enc.mean(), inplace=True)

df_test = df_test.drop('Pe_enc', axis=1, errors='ignore')
df_test = df_test.merge(Pe_enc[['Previous_event_type', 'Pe_enc']], how='left', on=['Previous_event_type'])
df_test = df_test.drop('Previous_event_type', axis=1, errors='ignore')
df_test.Pe_enc.fillna(df_test.Pe_enc.mean(), inplace=True)

df_train.Empty_net = df_train.Empty_net.astype(int)
df_train.Is_rebound = df_train.Is_rebound.astype(int)
df_train.Goal = df_train.Goal.astype(float)

df_val.Empty_net = df_val.Empty_net.astype(int)
df_val.Is_rebound = df_val.Is_rebound.astype(int)
df_val.Goal = df_val.Goal.astype(float)

df_test.Empty_net = df_test.Empty_net.astype(int)
df_test.Is_rebound = df_test.Is_rebound.astype(int)
df_test.Goal = df_test.Goal.astype(float)

train_labels = df_train.Goal.values.reshape(-1,1)
val_labels = df_val.Goal.values.reshape(-1,1)
test_labels = df_test.Goal.values.reshape(-1,1)

X_train = df_train.drop('Goal', axis=1)
y_train = df_train.Goal
X_val = df_val.drop('Goal', axis=1)
y_val = df_val.Goal
X_test = df_test.drop('Goal', axis=1)
y_test = df_test.Goal

# hypers = {"scale_pos_weight":9, "eta":0.01, "max_depth":5}
hypers = {"scale_pos_weight":1.5, "eta":0.1, "n_estimators":400, "max_depth":4}

bst = XGBClassifier(**hypers)
# bst = XGBClassifier()
# fit model
bst.fit(X_train, y_train)
preds = bst.predict(X_val)
probs = bst.predict_proba(X_val)[:,1]


# print(f'accuracy: {accuracy_score(y_val, preds)}\nauc: {roc_auc_score(y_val, probs)}\nprecision: {precision_score(y_val, preds)}\nrecall: {recall_score(y_val, preds)}')
# print(f'auc: {roc_auc_score(y_val, probs)}')
# print(f'precision: {precision_score(y_val, preds)}')
# print(f'recall: {recall_score(y_val, preds)}')
# exp = start_experiment(workspace='dizga', project_name='test')
# bst.save_model('model/xgb.model.json')
exp = start_experiment()
exp.set_name(f'xgb-all-{exp.get_name()}')
exp.log_parameters(hypers)
exp.log_dataset_info('columns', df_tot.columns)
exp.add_tags(['XGB', 'All', 'Valid', 'Best'])
add_metrics(
        exp,
        accuracy_score(y_val, preds),
        roc_auc_score(y_val, probs),
        precision_score(y_val, preds),
        recall_score(y_val, preds))

exp.log_model("xgb", "model/xgb.model.json")
plots(val_labels, probs, 'xgb-all', exp)

# plots(val_labels, probs, f'xgd-all-params')

# booster = bst.get_booster()
# importance = booster.get_score(importance_type='weight')

# # Sorting the feature importances
# sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# # Print sorted feature importances
# for feature, score in sorted_importance:
#     print(f'Feature {feature}: Score: {score}')

# exp.log_model("test-model", "model/test.model.json")


# exp.log_metrics(mtrs_dir)