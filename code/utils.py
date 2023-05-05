import pickle
import pandas as pd


def load_pikle(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d

def save_pikle(d, path):
    pickle.dump(d, open(path, 'wb'))
    
class ErrorFit(Exception):
    pass
    
    
def clean_data(df):
    df.drop(['round_number', 'game_id'], axis=1, inplace=True)
    # df = df.drop(["ct_equipment_value_sum", "t_equipment_value_sum"], axis=1)
    df = df[df['ct_players_alive'] <= 5]
    df = df[df['t_players_alive'] <= 5]
    df = df[df['round_time'] >= 0].reset_index(drop=True)
    df['round_time'] = pd.TimedeltaIndex(df['round_time'], unit='s')
    df = df.drop(df[df['round_time'] > pd.Timedelta(200, 's')].index)
    df['round_time'] = df['round_time'].dt.total_seconds()

    # TEMPORARY
    df.drop(["round_time"], axis=1, inplace=True)
    
    df = df[df['round_time_remaining'] >= 0].reset_index(drop=True)
    df['round_time_remaining'] = pd.TimedeltaIndex(df['round_time_remaining'], unit='s')
    df['round_time_remaining'] = df['round_time_remaining'].dt.total_seconds()
    return df    