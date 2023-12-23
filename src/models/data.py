import os, sys
sys.path.insert(0, os.getcwd()) 
from src.client import start_experiment
from src.features import load_df_shots, add_goalie_ratio, add_opponent_concedes, add_shooter_ratio, add_team_goals
import pandas as pd

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

subset_df = df[(df.Year == 2017) & (df.Game_id == 1065)]
subset_df.drop('Strength', axis=1, inplace=True)

exp = start_experiment()
exp.set_name("wpg_v_wsh-df")
exp.log_dataframe_profile(subset_df, name='wpg_v_wsh_2017021065', dataframe_format='csv')