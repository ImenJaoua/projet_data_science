import numpy as np

def season_full_name(season: int or str):
  return season + str(int(season)+1) if type(season) == str else f'{season}{season+1}'

def normalize(arr, min_value=0, max_value=1):
    min_x = np.min(arr)
    max_x = np.max(arr)
    
    normalized = min_value + (arr - min_x) / (max_x - min_x) * (max_value - min_value)
    return normalized