from dataclasses import dataclass
from copy import deepcopy
import typing

from src.data.models import Game

_60_seconds = 60
_120_seconds = 120

@dataclass
class Penalty:
    type: str
    start_time: int
    end_time: int
    team: str

@dataclass
class PPTeam:
    players: int
    start_time: int

@dataclass
class PPState:
    time: int
    teams: typing.Dict[str, PPTeam]

# Not all penalties induce power-play
pp_minor_pens = ['Bench Minor', 'Minor']
pp_major_pens = ['Major', 'Match']
minor_pen = 'minor'
major_pen = 'major'

def get_first_minor_penalty(penalties, team):
  for penalty in penalties:
    if penalty.team == team and penalty.type == minor_pen:
      return penalty


def get_powerplay_states(game:Game) -> list[PPState]:
  home_team = game.home_team.triCode
  away_team = game.away_team.triCode

  get_team = lambda x: home_team if game.is_home_team(x) else away_team
  get_other_team = lambda x: away_team if game.is_home_team(x) else home_team
  state = PPState(0, {home_team:PPTeam(5, None), away_team:PPTeam(5, None)})
  states = [deepcopy(state)]
  penalties = []
  last_time = 0
  pens_and_goals = sorted(game.penalty_plays + game.scoring_plays + [len(game.plays) - 1])

  def set_powerplay(state: PPState, time):
    if state.teams[home_team].players > state.teams[away_team].players:
      state.teams[away_team].start_time = None
      state.teams[home_team].start_time = state.teams[home_team].start_time or time
    elif state.teams[home_team].players < state.teams[away_team].players:
      state.teams[home_team].start_time = None
      state.teams[away_team].start_time = state.teams[home_team].start_time or time   
    else:
      state.teams[home_team].start_time = None
      state.teams[away_team].start_time = None
    return state

  for play_id in pens_and_goals:
    play = game.plays[play_id]
    current_time = play.game_time

    team = get_team(play.team.triCode) if play.team else None
    other_team = get_other_team(play.team.triCode) if play.team else None
    
    penalties.sort(key=lambda x: x.end_time)

    # Update state with ended penalty
    penalties_handled = 0
    for penalty in penalties:
      if penalty.end_time < current_time:
        penalties_handled += 1
        state.teams[penalty.team].players += 1
        state = set_powerplay(state, penalty.end_time)
        state.time = penalty.end_time
        if last_time == penalty.end_time:
          states[-1] = deepcopy(state)
        else:
          states.append(deepcopy(state))
        last_time = penalty.end_time

    penalties = penalties[penalties_handled:]


    # End a minor penalty for the other team if a goal was made during powerplay
    if play.result.event == 'Goal' and state.teams[team].start_time is not None:
      minor_penalty = get_first_minor_penalty(penalties, other_team)
      if minor_penalty:
        if minor_penalty.end_time - current_time < _120_seconds:
          minor_penalty.end_time = current_time
        else:
          delta = _120_seconds + minor_penalty.start_time - current_time
          minor_penalty.end_time -= delta

    # Update state with new penalty
    if play.result.penaltySeverity in pp_minor_pens + pp_major_pens:
      end_time = current_time + _60_seconds * play.result.penaltyMinutes
      if play.result.penaltySeverity in pp_minor_pens:
        penalty = Penalty('minor', current_time, end_time, team)
      else:
        penalty = Penalty('major', current_time, end_time, team)
      penalties.append(penalty)
      state.teams[team].players -= 1
      state = set_powerplay(state, current_time)
      state.time = penalty.start_time
      if last_time == penalty.start_time:
        states[-1] = deepcopy(state)
      else:
        states.append(deepcopy(state))
      last_time = penalty.start_time

  states.append(PPState(game.plays[-1].game_time, {home_team:PPTeam(5, None), away_team:PPTeam(5, None)}))
  
  return states
