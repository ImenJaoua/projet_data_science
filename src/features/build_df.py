import math
import os
from src.features.powerplay import get_powerplay_states
import numpy as np
import pandas as pd

from src.data.load import NHLDataDownloader


def load_df_shots(year, playoffs = False, filename: str = "", save = True) -> pd.DataFrame:
  """
  Load season data and transform to a DataFrame with shots and goals events.

  Args:
      year (int): The first year of the season to retrieve, i.e. for the 2016-17
                season you'd put in 2016
      filename (Optional[str]): Path + filename of the file to load or save data into.
  """

  version = 0.16
  path = f'data/df/{version}'
  os.makedirs(path, exist_ok=True)

  filename = filename or f'{path}/df_{year}{"-p" if playoffs else ""}_{version}.pkl'

  if os.path.isfile(filename):
    return pd.read_pickle(filename)
  
  season = NHLDataDownloader(year).load_processed_data()

  columns = [
    'Game_id',
    'Game_time',  # In seconds
    'Period',
    'Time',       # Time since the period started mm:ss 
    'Team',
    'OppTeam',
    'Goal',
    'X',
    'Y',
    'Shooter',
    'Goalie',
    'Type',
    'Empty_net',
    'Strength',
    'Previous_event_type',
    'Previous_x',
    'Previous_y',
    'Previous_time_since',
    'Previous_distance',
    'Speed',
    'Is_rebound',
    'Time_since_powp',
    'Players',
    'Opp_players']
  
  # add column: is_home_team, score so far, player goal ratio runing average, goalie stops ratio running average...
  
  data = []

  print("Creating Dataframe...")
  
  games = season.playoffs.values() if playoffs else season.regulars
  for game_id, game in enumerate(games):
    if game.plays:
      powp_states = iter(get_powerplay_states(game))
      powp = next(powp_states)
      next_powp = next(powp_states)
    last_play = None
    for play in game.plays:

      if play.coordinates and (play.result.event == 'Goal' or play.result.event == 'Shot'):

        tireur = ""
        gardien = ""
        for player_event in play.players:
          if player_event.playerType == 'Scorer' or player_event.playerType == 'Shooter':
            tireur = player_event.player.fullName
          if player_event.playerType == 'Goalie':
            gardien = player_event.player.fullName

        game_time = play.game_time
        periode = play.about.period
        time = play.about.periodTime.isoformat()[3:]
        id = game_id + 1       
        team = play.team.triCode
        opp_team = game.away_team.triCode if game.is_home_team(team) else game.home_team.triCode
        but = play.result.event == 'Goal'
        x = play.coordinates.x or 0
        y = play.coordinates.y or 0
        shot_type = play.result.secondaryType
        empty_net = play.result.emptyNet
        strength = play.result.strength
        last_play_event = last_play.result.event
        if (last_play.coordinates is None):
          last_play_x = 0
          last_play_y = 0
        else:
          last_play_x = last_play.coordinates.x or 0
          last_play_y = last_play.coordinates.y or 0
        time_since_last_play = game_time - last_play.game_time
        last_play_distance = math.dist([x, y],[last_play_x, last_play_y])
        is_rebound = True if last_play_event == 'Shot' else False
        delta = time_since_last_play or 1
        speed = last_play_distance / delta

        while game_time > next_powp.time:
          powp = next_powp
          next_powp = next(powp_states)

        powp_time = game_time - powp.teams[team].start_time if (powp.teams[team].start_time is not None) else 0
        team_players = powp.teams[team].players
        opp_team_players = powp.teams[opp_team].players
    
        data.append([
          id,
          game_time,
          periode,
          time,
          team,
          opp_team,
          but,
          x,
          y,
          tireur,
          gardien,
          shot_type,
          empty_net,
          strength,
          last_play_event,
          last_play_x,
          last_play_y,
          time_since_last_play,
          last_play_distance,
          speed,
          is_rebound,
          powp_time,
          team_players,
          opp_team_players])

      last_play = play
        
  df = pd.DataFrame(data, columns=columns)

  df.Empty_net = df.Empty_net.fillna(False)
  df.Strength = df.Strength.fillna('Even')

  # Get the opponant net position
  df['Avg'] = df.groupby(['Game_id', 'Period', 'Team'])['X'].transform('mean')
  def distance_x_from_net(row):
      x = row.X
      x_net = 89 if row.Avg > 0 else -89
      return abs(x_net-x)

  df['X_net'] = df.apply(distance_x_from_net, axis=1)
  df['Shot_distance'] = df.apply(lambda row: math.dist([row.X_net, row.Y],[0, 0]), axis=1)
  df.drop('Avg', axis=1, inplace = True)

  # Compute shot angle, goes from -90 (on the left) to 90 (on the right).
  df['Shot_angle'] = df.apply(lambda row: math.degrees(math.atan2(abs(row.Y), row.X_net)), axis=1)
  sign = np.sign(df.Y) * np.sign(df.X)
  df['Shot_angle'] *= sign

  # Compute rebound angle
  prev_angle = df.Shot_angle.shift(1, fill_value=0)
  df['Rebound_angle'] = abs(prev_angle - df['Shot_angle'])
  df.loc[~df.Is_rebound, 'Rebound_angle'] = 0

  
  df['Year'] = year

  df = df.infer_objects()

  if save:
    df.to_pickle(filename)

  print("Done!")

  return df