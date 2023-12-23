import pandas as pd

def add_shooter_ratio(df):
    df = df.copy()
    goal_df = df[['Year','Game_id','Shooter', 'Goal']]
    goal_ratio_df = goal_df.groupby(['Year','Game_id', 'Shooter']).Goal.mean().reset_index(name='Goal_ratio')

    base = pd.DataFrame([[0,0,n,0.1] for n in goal_ratio_df.Shooter.unique()], columns=goal_ratio_df.columns)
    goal_ratio_df = pd.concat([base, goal_ratio_df]).reset_index(drop=True)
    goal_ratio_df['Ema'] = goal_ratio_df.groupby('Shooter')['Goal_ratio'].transform(
        lambda x: x.ewm(alpha=0.01, adjust=False).mean()
    )

    df = df.drop('Shooter_ratio', axis=1, errors='ignore')

    goal_ratio_df['Shooter_ratio'] = goal_ratio_df.groupby('Shooter')['Ema'].shift().round(4)
    df = df.merge(goal_ratio_df[['Year','Game_id', 'Shooter', 'Shooter_ratio']], how='left', on=['Year','Game_id', 'Shooter'])
    return df

def add_goalie_ratio(df):
    df = df.copy()
    save_df = df[['Year','Game_id','Goalie', 'Goal']]
    save_ratio_df = save_df.groupby(['Year','Game_id', 'Goalie']).Goal.mean().reset_index(name='Save_ratio')
    save_ratio_df.Save_ratio = 1 - save_ratio_df.Save_ratio

    base = pd.DataFrame([[0,0,n,0.9] for n in save_ratio_df.Goalie.unique()], columns=save_ratio_df.columns)
    save_ratio_df = pd.concat([base, save_ratio_df]).reset_index(drop=True)
    save_ratio_df['Ema'] = save_ratio_df.groupby('Goalie')['Save_ratio'].transform(
        lambda x: x.ewm(alpha=0.01, adjust=False).mean()
    )

    df = df.drop('Goalie_ratio', axis=1, errors='ignore')

    save_ratio_df['Goalie_ratio'] = save_ratio_df.groupby('Goalie')['Ema'].shift().round(4)
    df = df.merge(save_ratio_df[['Year','Game_id', 'Goalie', 'Goalie_ratio']], how='left', on=['Year','Game_id', 'Goalie'])
    df.loc[df.Goalie == '', 'Goalie_ratio'] = 0.5
    return df

def add_team_goals(df):
    df = df.copy()
    goal_df = df[['Year','Game_id','Team', 'Goal']]
    goal_ratio_df = goal_df.groupby(['Year','Game_id', 'Team']).apply(
        lambda x: x['Goal'].sum()
    ).reset_index(name='Goal_ratio')

    base = pd.DataFrame([[0,0,n,3] for n in goal_ratio_df.Team.unique()], columns=goal_ratio_df.columns)
    goal_ratio_df = pd.concat([base, goal_ratio_df]).reset_index(drop=True)
    goal_ratio_df['Ema'] = goal_ratio_df.groupby('Team')['Goal_ratio'].transform(
        lambda x: x.ewm(alpha=0.01, adjust=False).mean()
    )

    df = df.drop('Team_goals', axis=1, errors='ignore')

    goal_ratio_df['Team_goals'] = goal_ratio_df.groupby('Team')['Ema'].shift().round(4)
    df = df.merge(goal_ratio_df[['Year','Game_id', 'Team', 'Team_goals']], how='left', on=['Year','Game_id', 'Team'])
    return df

def add_opponent_concedes(df):
    df = df.copy()
    goal_df = df[['Year','Game_id', 'OppTeam', 'Goal']]
    goal_ratio_df = goal_df.groupby(['Year','Game_id', 'OppTeam']).apply(
        lambda x: x['Goal'].sum()
    ).reset_index(name='Goals')

    base = pd.DataFrame([[0,0,n,3] for n in goal_ratio_df.OppTeam.unique()], columns=goal_ratio_df.columns)
    goal_ratio_df = pd.concat([base, goal_ratio_df]).reset_index(drop=True)
    goal_ratio_df['Ema'] = goal_ratio_df.groupby('OppTeam')['Goals'].transform(
        lambda x: x.ewm(alpha=0.01, adjust=False).mean()
    )

    df = df.drop('Opp_concedes', axis=1, errors='ignore')

    goal_ratio_df['Opp_concedes'] = goal_ratio_df.groupby('OppTeam')['Ema'].shift().round(4)
    df = df.merge(goal_ratio_df[['Year','Game_id', 'OppTeam', 'Opp_concedes']], how='left', on=['Year','Game_id', 'OppTeam'])
    return df