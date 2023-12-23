import math
import os
import pickle
import sys
import pandas as pd

from src.data import RequestNHL
from src import utils
from src.data.models import Season

class NHLDataDownloader:
    def __init__(self, year):
        self.nb_playoffs = 105
        self.season_year = year
        self.season_fullname = utils.season_full_name(year) 
        self.r_games = []
        self.p_games = {}
    
    def set_season(self, season_input_value):
        self.season_year = season_input_value
        self.season_fullname = utils.season_full_name(season_input_value)

    def playoff_code(self,id: int) -> int:
      """
      Return a playoff id {round}{matchup}{game} depending on an index
      https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
      i.e.    index 1 correspond to the playoff game 111
              index 2 correspond to the playoff game 112
              ...
              index 8 correspond to the playoff game 121
              etc.

      Args:
          id (int): index to convert
      """
      threshold = 0
      rest = id - 1
      for i in range(4):
        rest = rest - threshold
        threshold = 7 * 8 / 2 ** (i)
        if rest < threshold:
          matchup = int(rest / 7) + 1
          game = int(rest % 7) + 1
          round = i + 1
          return int(f"{round}{matchup}{game}")

    def load_data(self, filename: str = "", samples: bool = False) -> object:
      """
      Load season data from file. If the file does not exists, the data is download and saved.
      Data includes regular and playoff games.

      Args:
          year (int): The first year of the season to retrieve, i.e. for the 2016-17
                    season you'd put in 2016
          filename (Optional[str]): Path + filename of the file to load or save data into.

          samples (bool): if true, only a small portion of the data is downloaded, default false.
      """
      # Définir le chemin
      dir_data = 'data'

      # Créer le dossier data
      try:
        if not os.path.isdir(dir_data):
          os.mkdir(dir_data)
          print("Le dossier '% s' a été créé" % dir_data) 
      except FileExistsError as e:
          print("Erreur dans le temps créer de dessier '% s' avec : "+e % dir_data) 

      season_fullname = utils.season_full_name(self.season_year)

      default_filename = f'data/{season_fullname}.pkl'
      if samples:
        default_filename = f'data/{season_fullname}-samples.pkl'

      season_file = filename or default_filename

      if os.path.isfile(season_file):
        with open(season_file, 'rb') as file:
          data = pickle.load(file)
          print(f'Season {self.season_year} successfully loaded from file')
      else:

        data = {}

        nb_regular_games = RequestNHL.nb_regular_games(self.season_year)
        nb_playoffs = 105
        if samples:
          nb_regular_games = 20
          nb_playoffs = 5

        print('Downloading:')
        for id in range(1, nb_regular_games + 1):
          game = RequestNHL.get_game(self.season_year, id)
          self.r_games.append(game)
          sys.stdout.write(f'\r Regulars: {id}/{nb_regular_games}')
          sys.stdout.flush()
        print()

        for id in range(1, nb_playoffs + 1):
          code = self.playoff_code(id)
          game = RequestNHL.get_game(self.season_year, code, regular=False)

          # Some not played game have no data (except a message)
          if 'message' in game:
            game['gamePk'] = int(code)

          self.p_games[code] = game
          sys.stdout.write(f'\r Playoffs: {id}/{nb_playoffs}')
          sys.stdout.flush()
        print()

        data = {"regulars":self.r_games, "playoffs":self.p_games}

        # Save season data to file
        with open(season_file, 'wb') as file:
          pickle.dump(data, file)
          print(f'Season {self.season_year} successfully saved to file')

      return data

    def load_processed_data(self, filename: str = "", samples: bool = False) -> Season :
      """
      Load season data and parse the data into the Season model.

      Args:
          year (int): The first year of the season to retrieve, i.e. for the 2016-17
                    season you'd put in 2016
          filename (Optional[str]): Path + filename of the file to load or save data into.

          samples (bool): if true, only a small portion of the data is downloaded, default false.
      """
      data = self.load_data(filename, samples)
      print('Processing data... (1-2 minutes)')
      season = Season.model_validate(data)
      print('Done!')
      return season

    def load_df_shots(self, filename: str = "", season: Season = None) -> pd.DataFrame:
      """
      Load season data and transform to a DataFrame with shots and goals events.

      Args:
          year (int): The first year of the season to retrieve, i.e. for the 2016-17
                    season you'd put in 2016
          filename (Optional[str]): Path + filename of the file to load or save data into.

          season (Season): (Optional) Season to convert to DataFrame
      """

      version = 0.1

      filename = filename or f'data/shots_{self.season_year}-{version}.pkl'

      if not season:
        if os.path.isfile(filename):
          return pd.read_pickle(filename)
        season = self.load_processed_data()

      columns = ['Game_id',
              'Period',
              'Time',
              'Team',
              'Goal',
              'X',
              'Y',
              'Shooter',
              'Goalie',
              'Type',
              'Empty_net',
              'Strength']
      
      data = []

      print("Creating Dataframe...")
      
      for game_id, game in enumerate(season.regulars):
        for play in game.plays:
          if play.coordinates and (play.result.event == 'Goal' or play.result.event == 'Shot'):

            tireur = ""
            gardien = ""
            for player_event in play.players:
              if player_event.playerType == 'Scorer' or player_event.playerType == 'Shooter':
                tireur = player_event.player.fullName
              if player_event.playerType == 'Goalie':
                gardien = player_event.player.fullName

            periode = play.about.period
            time = play.about.periodTime.isoformat()[3:]
            id = game_id + 1
            equipe = play.team.triCode
            but = play.result.event == 'Goal'
            x = play.coordinates.x
            y = play.coordinates.y
            shot_type = play.result.secondaryType
            empty_net = play.result.emptyNet
            strength = play.result.strength
            data.append([id, periode, time, equipe, but, x, y, tireur, gardien, shot_type, empty_net, strength])
            
      df = pd.DataFrame(data, columns=columns)

      # Get the opponant net position
      df['Avg'] = df.groupby(['Game_id', 'Period', 'Team'])['X'].transform('mean')
      def distance_x_from_net(row):
          x = row.X
          x_net = 89 if row.Avg > 0 else -89
          return abs(x_net-x)

      df['X_dist'] = df.apply(distance_x_from_net, axis=1)
      df['Net_distance'] = df.apply(lambda row: math.dist([row.X_dist, row.Y],[0, 0]), axis=1)
      df.drop('Avg', axis=1, inplace = True)

      df = df.infer_objects()

      df.to_pickle(filename)

      print("Done!")

      return df
    
def get_seconds(time_str:str)-> int:
    # Diviser la chaîne de temps
    minutes, seconds = map(int, time_str.split(":"))
    # Calculer le temps en secondes
    total_seconds = minutes * 60 + seconds
    #print(total_seconds)
    return total_seconds


# Jeux de pénalité 
def play_penalty(season_data):
  penalty_plays = season_data['regulars'][0]['liveData']['plays']['penaltyPlays']
  #print('penalty_plays: ',penalty_plays)
  all_penalties = []

  for play in penalty_plays:
      penalty_info = season_data['regulars'][0]['liveData']['plays']['allPlays'][play]
      # Trouver des données  
      penalty_team = penalty_info['team']['triCode']
      penalty_time = penalty_info['about']['periodTime']
      penalty_period = penalty_info['about']['period']
      penalty_description = penalty_info['result']['description']
      
      # Ajouter dans List pour retourner
      penalty_details = {
          'penalty_period':penalty_period,
          'penalty_team':penalty_team,
          'penalty_time':get_seconds(penalty_time),
          'penalty_description':penalty_description
          }
      all_penalties.append(penalty_details)
      
  #print(all_penalties)
  return all_penalties

# Calculer des variables question 4.4
def get_jeu_puissance(season_data,all_penalties)->list:
    power_play_active = False  # Indique si un power-play est actif
    power_play_start_time = 0  # Temps de début du power-play en secondes
    friendly_skaters_on_ice = 5  # Nombre de patineurs non-gardiens amicaux sur la glace
    opposing_skaters_on_ice = 5  # Nombre de patineurs non-gardiens adverses sur la glace
    home = season_data['regulars'][0]['gameData']['teams']['home']['triCode']
    #print(home)
    away = season_data['regulars'][0]['gameData']['teams']['away']['triCode']
    #print(away)
    # Parcourir toutes les pénalités
    for penalty in all_penalties:
        penalty_team = penalty['penalty_team']
        penalty_time = penalty['penalty_time']

        # Vérifier si une nouvelle pénalité a commencé
        if not power_play_active:
            power_play_active = True
            power_play_start_time = penalty_time
            if penalty_team == home:
                friendly_skaters_on_ice -= 1
            else:
                opposing_skaters_on_ice -= 1
        else:
            # Vérifier si la pénalité actuelle a expiré
            if penalty_time - power_play_start_time >= 120:
                power_play_active = False
                power_play_start_time = 0
                if penalty_team == home:
                    friendly_skaters_on_ice += 1
                else:
                    opposing_skaters_on_ice += 1
    return {
            'time_elapsed_power_play': power_play_start_time,
            'friendly_skaters_on_ice': friendly_skaters_on_ice,
            'opposing_skaters_on_ice': opposing_skaters_on_ice
        }




