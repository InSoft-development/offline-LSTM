import time
import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np
import argparse
import json
from loguru import logger
from tensorflow.keras.models import load_model
from scipy.special import softmax
import clickhouse_connect
from utils.smooth import exponential_smoothing, double_exponential_smoothing
from utils.utils import load_config, get_len_size, hist_threshold, get_anomaly_interval

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Use argparse to get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='')
parser.add_argument('--file_format', type=str, default='db')
parser.add_argument('--csv_kks', type=bool, default=False)
opt = parser.parse_args()

# Load configuration
config = load_config(f'{opt.config_path}/config_offline.yaml')
KKS = config['KKS']
WEIGHTS = config['WEIGHTS']
SCALER = config['SCALER']
NUM_GROUPS = config['NUM_GROUPS']
LAG = config['LAG']
ROLLING_MEAN = config['ROLLING_MEAN']
EXP_SMOOTH = config['EXP_SMOOTH']
DOUBLE_EXP_SMOOTH = config['DOUBLE_EXP_SMOOTH']
ROLLING_MEAN_WINDOW = config['ROLLING_MEAN_WINDOW']
POWER_ID = config['POWER_ID']
POWER_LIMIT = config['POWER_LIMIT']
MEAN_NAN = config['MEAN_NAN']
DROP_NAN = config['DROP_NAN']
AUTO_DROP_LIST = config['AUTO_DROP_LIST']
DROP_LIST = config['DROP_LIST']
CSV_SAVE_RESULT = config['CSV_SAVE_RESULT']
CSV_DATA = config['CSV_DATA']
JSON_DATA = config['JSON_DATA']
TEST_FILE = config['TEST_FILE']
COUNT_ANOMALY = config['COUNT_ANOMALY']
ROLLING_MEAN_LOSS = config['ROLLING_MEAN_LOSS']
ANOMALY_TRESHOLD = config['ANOMALY_TRESHOLD']
COUNT_TOP = config['COUNT_TOP']
POWER_FILL = config['POWER_FILL']

# Create directories if they don't exist
os.makedirs(CSV_SAVE_RESULT, exist_ok=True)
os.makedirs(CSV_DATA, exist_ok=True)
os.makedirs(JSON_DATA, exist_ok=True)


# Load dataset based on file format
if opt.file_format == 'db':
    client = clickhouse_connect.get_client(host='10.23.0.177', username='default', password='asdf')
    test_df = client.query_df(TEST_FILE)
elif opt.file_format == 'csv':
    test_df = pd.read_csv(TEST_FILE)
elif opt.file_format == 'sqlite':
    cnx = sqlite3.connect(TEST_FILE)
    test_df = pd.read_sql_query("SELECT * FROM 'data_train'", cnx)

# Preprocess data
time_ = test_df['timestamp']
# logger.warning(time_)
test_df['index'] = test_df.index
test_df = test_df[test_df[POWER_ID] > POWER_LIMIT]
logger.warning(test_df)
test_time = test_df['timestamp']
index = test_df['index']
test_df = test_df.drop(columns=['timestamp','index'])

if MEAN_NAN:
    test_df = test_df.fillna(test_df.mean())

if DROP_NAN:
    test_df = test_df.dropna()

if ROLLING_MEAN:
    rolling_mean = test_df.rolling(window=ROLLING_MEAN_WINDOW).mean()

if EXP_SMOOTH:
    test_df = test_df.apply(exponential_smoothing, alpha=0.2)

if DOUBLE_EXP_SMOOTH:
    test_df = test_df.apply(double_exponential_smoothing, alpha=0.02, beta=0.09)

# Load models and scalers
model_list = []
scaler_list = []
for i in range(NUM_GROUPS):
    # Use f-strings to interpolate the values
    model_file = f'{WEIGHTS}/lstm_group_{i}.h5'
    model = load_model(model_file)

    scaler_file = f'{SCALER}/scaler_{i}.pkl'
    scaler = joblib.load(scaler_file)
    scaler_list.append(scaler)
    model_list.append(model)

# Load groups
if not opt.csv_kks:
    groups = client.query_df("SELECT * FROM kks")
else:
    groups = pd.read_csv(KKS, sep=';')

g = groups[groups['group']==0]
zero_group = g['kks']
group_list = []
unscaled = []
for i in range(NUM_GROUPS):
    group = groups[groups['group'] == i]
    names = groups['name'][groups['group'] == i]

    if i != 0:
        group = group.append(groups[groups['group'] == 0])

    if len(group) == 0:
        continue

    group = test_df[group['kks']]
    scaler = scaler_list[i]
    scaled_data = pd.DataFrame(
        data=scaler.transform(group),
        columns=group.columns
    )

    group_list.append(scaled_data)
    unscaled.append(group)

if AUTO_DROP_LIST:
    DROP_LIST = list(group_list[0].columns)
    logger.debug(DROP_LIST)

dict_list = []  # Create an empty list to store dictionaries
for i, (data, scaled_data, model) in enumerate(zip(unscaled, group_list, model_list)):
    X = scaled_data.to_numpy()
    len_size = get_len_size(LAG, X.shape[0])
    X = X[:len_size].reshape(int(X.shape[0] / LAG), LAG, X.shape[1])
    preds = model.predict(X, verbose=1)
    preds = preds[:, 0, :]
    yhat = X[:, 0, :]
    loss = np.mean(np.abs(yhat - preds), axis=1)
    prob = softmax(loss)
    each_loss = np.abs(yhat - preds)
    df_lstm = pd.DataFrame(each_loss, columns=scaled_data.columns)
    df_lstm['target_value'] = loss
    df_lstm['softmax'] = prob
    # df_lstm['timestamp'] = np.array(test_time)
    df_lstm.index = test_time[:len_size][::LAG]
    print(str(df_lstm.index[0]))
    df_timestamps = pd.DataFrame()
    df_timestamps['timestamp'] = time_
    print(df_timestamps)
    df_lstm = pd.merge(df_lstm, df_timestamps, on='timestamp', how='right')
    # logger.info(df_lstm)
    # df_lstm['index_'] = np.array(index)
      # Можно указать 'zeroes', чтобы заполнять нулями
    if POWER_FILL == 'last_value':
        df_lstm.fillna(method='ffill', inplace=True)  # Заполняем пропущенные значения последними непустыми значениями
    elif POWER_FILL == 'zeroes':
        df_lstm.fillna(0, inplace=True) 
    logger.error(df_lstm)
    try:
        if i != 0:
            df_lstm = df_lstm.drop(columns=DROP_LIST)

    except:
        logger.info('No columns to drop')

    try:
        df_lstm = df_lstm.drop(columns=DROP_LIST)
        logger.info(f'Drop {DROP_LIST}')

    except:
        logger.info('No columns to drop')

    df_lstm.to_csv(f'{CSV_SAVE_RESULT}/lstm_group_{i}.csv')
    data.to_csv(f'{CSV_DATA}/group_{i}.csv')
    logger.info(df_lstm)
    rolling_loss = df_lstm.rolling(window=ROLLING_MEAN_LOSS, axis='rows', min_periods=1).mean()
    rolling_loss = rolling_loss.drop(columns=zero_group)
    rolling_loss = rolling_loss.drop(columns = ['softmax'])
    treshold = hist_threshold(rolling_loss['target_value'], ANOMALY_TRESHOLD)
    interval_list, idx_list = get_anomaly_interval(rolling_loss['target_value'], treshold, min_interval_len=COUNT_ANOMALY)
    time = df_lstm['timestamp']
    for j in idx_list:
        top_list = rolling_loss[j[0]:j[1]].drop(columns='target_value').mean().sort_values(ascending=False).index[:COUNT_TOP].to_list()
        # Create a dictionary for each anomaly
        report_dict = {
            "time": (str(time[j[0]]), str(time[j[1]])),
            "len": j[1] - j[0],
            "index": j,
            "top_sensors": top_list
        }
        dict_list.append(report_dict)

    # Save the dictionary to a json file
    with open(f"{JSON_DATA}/group{i}.json", "w") as outfile:
        json.dump(dict_list, outfile, indent=4)

    dict_list = []  # Reset the dict_list for the next group
