
import tensorflow as tf
import numpy as np
import pandas as pd
from yaml import load
from yaml import FullLoader
from loguru import logger


def set_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)
    
def get_len_size(LAG,x_size):
  return int(x_size/LAG) * LAG

def hist_threshold(data, percent):
    h,bins = np.histogram(pd.DataFrame(data),bins = 1000)
    max_p = sum(h)* percent
    s = 0
    N = -1
    for i in range(0,1000):
      s += h[i]
      if s > max_p:
        N = i 
        break
    return bins[N]


def get_interval(data,loss,df,time,count_anomaly,anomaly_treshold,left_space, right_space, left_history, right_history):
    anomaly_list = []
    time_list = []
    report_list = []
    data_list = []
    history_list = []
    i = 0
    logger.debug(f'porog {float(anomaly_treshold)}')
    for value in loss:
      if float(value)>anomaly_treshold:
        anomaly_list.append(value)
      else:
        if len(anomaly_list) > count_anomaly:

          report_list.append(df[i-len(anomaly_list):i])
          data_list.append(data[i-len(anomaly_list):i])

          if (i-len(anomaly_list)-left_history)>=0 and (i+right_history)<=len(df):
            history_list.append(data[i-len(anomaly_list)-left_history:i+right_history])
  
            logger.debug(f'all {len(data[i-len(anomaly_list)-left_space:i+right_space])}')
          elif (i-len(anomaly_list)-left_history)<0: 
            history_list.append(data[i-len(anomaly_list):i+right_history])
          
            logger.debug(f' left {len(data[i-len(anomaly_list):i+right_space])}')
            
          elif (i+right_history)>=len(df):
            history_list.append(data[i-len(anomaly_list)-left_history:i])
            
            logger.debug(f' right {len(data[i-len(anomaly_list)-left_space:i])}')
          else: 
            history_list.append(data[i-len(anomaly_list):i])
          time_list.clear()
          anomaly_list.clear()
        else:
          time_list.clear()
          anomaly_list.clear()
      
      i+=1
    return report_list, data_list, history_list
def get_anomaly_interval(loss, threshold_short, threshold_long, len_long, len_short,count_continue_short = 10, count_continue_long = 15):
  interval_list = []
  loss_interval = []
  count = 0
  i = 0
  idx_list = []
  sum_anomaly = 0
  for val in loss:
    i+=1
    if val>threshold_long:
      loss_interval.append(val)
    else:
       count+=1
       loss_interval.append(val)
       if count>count_continue_long:
         if len(loss_interval)>len_long:
          interval_list.append(loss_interval)
          logger.info(f'Add anomaly interval, len {len(loss_interval)}')
          if i-len(loss_interval) > 0:
            idx_list.append((i-len(loss_interval),i))
          else: 
            idx_list.append((0,i))
          sum_anomaly+=len(loss_interval)
         count = 0
         loss_interval.clear()
  i = 0
  for val in loss:
    i+=1
    if val>threshold_short:
      loss_interval.append(val)
    else:
       count+=1
       loss_interval.append(val)
       if count>count_continue_short:
         if len(loss_interval)>len_short:
          interval_list.append(loss_interval)
          logger.info(f'Add anomaly interval, len {len(loss_interval)}')
          if i-len(loss_interval) > 0:
                idx_list.append((i-len(loss_interval),i))
          else: 
            idx_list.append((0,i))
          sum_anomaly+=len(loss_interval)
         count = 0
         loss_interval.clear()
      
  logger.info(f'Sum anomaly {sum_anomaly}, part of anomaly {round(sum_anomaly/len(loss),3)}')
  return interval_list, idx_list