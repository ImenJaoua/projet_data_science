import requests
from src import utils

class RequestNHL():
    nb_games = {30: 1230, 31:1271, 32:1353}
    base_url = "https://statsapi.web.nhl.com/api/v1/"
    teams_url = base_url + 'teams'

    @classmethod
    def get_game(self, season: int, id = int, regular = True) -> object:
        """
        Get a game data

        Args:
            season (int): The first year of the season to retrieve, i.e. for the 2016-17
                season you'd put in 2016
            if (int): The id of the game to retrive
        """
        season = str(season)
        game_id = f"{id:04d}"
        url = self.base_url + 'game/' + season
        # if regular game we add to the url 02 and if it is playoff we add 03
        url += '02' if regular else '03'
        url += game_id
        url += '/feed/live'
        return requests.get(url).json()
    
    @classmethod
    def nb_regular_games(self, season: int) -> int:
        """
        Get the number of regulars games for the specified season

        Args:
            season (int): The first year of the season to retrieve, i.e. for the 2016-17
                season you'd put in 2016
        """
        # season_fullname = season + str(int(season)+1)
        season_fullname = utils.season_full_name(season)
        teams = requests.get(self.teams_url, params={"season": season_fullname})
        teams = teams.json()
        return self.nb_games[len(teams["teams"])]