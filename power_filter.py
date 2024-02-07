import pandas as pd
import argparse
from yaml import load
from yaml import FullLoader


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='')
    return parser.parse_args()

def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)


    
def main():
    opt = get_options()
    print(opt.config_path)
    
    config = load_config(f'{opt.config_path}/config_offline.yaml')
    POWER_LIMIT = config['POWER_LIMIT']
    POWER_INDEX = config['POWER_ID']
    power_data = pd.read_csv(config['TEST_FILE'])[POWER_INDEX]

    for k in range(config['NUM_GROUPS']):
        predict_file_path = f'/home/art/InControl/Offline_LSTM/csv_predict/csv_predict/predict_{k}.csv'
        df = pd.read_csv(predict_file_path)
        print(f'Processing file: {predict_file_path}')

        for i in range(len(df)):
            if power_data[i] < POWER_LIMIT:
                df.at[i, 'target_value'] = 0

        new_file_path = f'/home/art/InControl/Offline_LSTM/csv_predict/csv_predict_new/predict_{k}.csv'
        df.to_csv(new_file_path, index=False)

        print(f'Saved modified data to: {new_file_path}')

main()