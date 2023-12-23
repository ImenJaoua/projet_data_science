import os, sys

sys.path.insert(0, os.getcwd()) 
from src.client.comet import add_metrics
from src.client import start_experiment
from src.visualizations.visualization import plots
from src.features import load_df_shots, add_goalie_ratio, add_opponent_concedes, add_shooter_ratio, add_team_goals
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, minmax_scale
import tensorflow_decision_forests as tfdf
from sklearn.metrics import roc_curve
from keras import metrics


df_2016 = load_df_shots(2016)
df_2017 = load_df_shots(2017)
df_2018 = load_df_shots(2018)
df_2019 = load_df_shots(2019)
df_2020 = load_df_shots(2020)

df = pd.concat([df_2016, df_2017, df_2018, df_2019, df_2020]).reset_index(drop=True)

df = add_shooter_ratio(df)
df = add_goalie_ratio(df)
df = add_team_goals(df)
df = add_opponent_concedes(df)

df.Shot_angle = df.Shot_angle.abs()
df['Powp'] = df.Players - df.Opp_players
df.loc[df.Powp < 0, 'Powp'] = 0

df_tot = df[[
        'Game_id',
        'Game_time',
        'Type',
        # 'Empty_net',
        'Previous_distance',
        'Speed',
        # 'Is_rebound',
        'Time_since_powp',
        'Powp',
        'Shot_distance',
        'Shot_angle',
        'Rebound_angle',
        'Year',
        'Shooter_ratio',
        'Goalie_ratio',
        'Team_goals',
        'Opp_concedes',
        'Previous_event_type',
        'Goal']].copy()

# df['P_diff'] = df.Players - df.Opp_players

enc = OrdinalEncoder()

df_tot.Game_time = minmax_scale(df_tot.Game_time.values)
df_tot.Previous_distance = minmax_scale(df_tot.Previous_distance.values)
df_tot.Speed = minmax_scale(df_tot.Speed.values)
df_tot.Time_since_powp = minmax_scale(df_tot.Time_since_powp.values)
df_tot.Shot_distance = minmax_scale(df_tot.Shot_distance.values)
df_tot.Shooter_ratio = minmax_scale(df_tot.Shooter_ratio.values)
df_tot.Goalie_ratio = minmax_scale(df_tot.Goalie_ratio.values)
df_tot.Team_goals = minmax_scale(df_tot.Team_goals.values)
df_tot.Opp_concedes = minmax_scale(df_tot.Opp_concedes.values)
df_tot.Shot_angle = minmax_scale(df_tot.Shot_angle.values)
df_tot.Powp = minmax_scale(df_tot.Powp.values)
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

type_enc = df_train.groupby('Previous_event_type').Goal.mean().reset_index(name='Pe_enc')
df_train = df_train.drop('Pe_enc', axis=1, errors='ignore')
df_train = df_train.merge(type_enc[['Previous_event_type', 'Pe_enc']], how='left', on=['Previous_event_type'])
df_train = df_train.drop('Previous_event_type', axis=1, errors='ignore')

df_val = df_val.drop('Pe_enc', axis=1, errors='ignore')
df_val = df_val.merge(type_enc[['Previous_event_type', 'Pe_enc']], how='left', on=['Previous_event_type'])
df_val = df_val.drop('Previous_event_type', axis=1, errors='ignore')
df_val.Pe_enc.fillna(df_val.Pe_enc.mean(), inplace=True)

df_test = df_test.drop('Pe_enc', axis=1, errors='ignore')
df_test = df_test.merge(type_enc[['Previous_event_type', 'Pe_enc']], how='left', on=['Previous_event_type'])
df_test = df_test.drop('Previous_event_type', axis=1, errors='ignore')
df_test.Pe_enc.fillna(df_test.Pe_enc.mean(), inplace=True)

# df_train.Empty_net = df_train.Empty_net.astype(int)
# df_train.Is_rebound = df_train.Is_rebound.astype(int)
df_train.Goal = df_train.Goal.astype(float)

# df_val.Empty_net = df_val.Empty_net.astype(int)
# df_val.Is_rebound = df_val.Is_rebound.astype(int)
df_val.Goal = df_val.Goal.astype(float)

# df_test.Empty_net = df_test.Empty_net.astype(int)
# df_test.Is_rebound = df_test.Is_rebound.astype(int)
df_test.Goal = df_test.Goal.astype(float)

train_labels = df_train.Goal.values.reshape(-1,1)
val_labels = df_val.Goal.values.reshape(-1,1)
test_labels = df_test.Goal.values.reshape(-1,1)

pos = df_train.Goal.sum()
neg = len(df_train) - df_train.Goal.sum()
tot = len(df_train)
class_weight = {0: (1 / neg) * (tot/2.0), 1: (1 / pos) * (tot/2.0)}

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="Goal")
val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_val, label="Goal")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_test, label="Goal")

# exp = start_experiment(workspace='dizga', project_name='test')
exp = start_experiment()
exp.set_name(f'tfdf-{exp.get_name()}')

hypers = {"num_trees":400, "max_depth":28}

model = tfdf.keras.RandomForestModel(**hypers)
model.fit(train_ds, class_weight=class_weight)

predictions = model.predict(val_ds).reshape(-1)

fpr, tpr, _ = roc_curve(val_labels, predictions)


mtrcs = ["accuracy", metrics.AUC(), metrics.Precision(), metrics.Recall()]
model.compile(metrics=mtrcs)
eval = model.evaluate(val_ds)[1:]

exp.log_curve(f'tfdf-{exp.get_name()}', fpr, tpr)
exp.log_parameters(hypers)

add_metrics(exp, eval[0], eval[1], eval[2], eval[3])

plots(val_labels, predictions, "Decision forest", exp)

exp.end()