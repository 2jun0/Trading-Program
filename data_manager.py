import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import data_config

def load_chart_data(fpath: str) -> DataFrame:
	chart_data = pd.read_csv(fpath, thousands=',', header=None) # 파일에 헤더값이 이미 있고 ',' 단위로 나누어져있다.
	selected = chart_data[data_config.DATA_COLUMN.keys()]

	return selected.copy()

def preprocess(chart_data: DataFrame) -> DataFrame:
	prep_data = chart_data
	for window in data_config.MA_WINDOWS:
		prep_data[f'close_ma{window}'] = prep_data['close'].rolling(window).mean()
		prep_data[f'volume_ma{window}'] = (
			prep_data['volume'].rolling(window).mean())

	return prep_data

def build_training_data(prep_data:DataFrame) -> DataFrame:
	training_data = prep_data

	# 시가/전일종가 비율
	training_data['open_lastclose_ratio'] = np.zeros(len(training_data)) 
	training_data.loc[1:, 'open_lastclose_ratio'] = \
		(training_data['open'][1:].values - training_data['close'][:-1].values) / training_data['close'][:-1].values
	# 고가/종가 비율
	training_data.loc['high_close_ratio'] = \
		(training_data['high'].values - training_data['close'].values) / training_data['close'].values
	# 저가/종가 비율
	training_data.loc['low_close_ratio'] = \
		(training_data['low'].values - training_data['close'].values) / training_data['close'].values
	# 종가/전일종가 비율
	training_data['close_lastclose_ratio'] = np.zeros(len(training_data)) 
	training_data.loc[1:, 'close_lastclose_ratio'] = \
		(training_data['close'][1:].values - training_data['close'][:-1].values) / training_data['close'][:-1].values
	# 거래량/전일거래량 비율
	training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data)) 
	training_data.loc[1:, 'volume_lastvolume_ratio'] = \
		(training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
		training_data['volume'][:-1].values \
			.replace(to_replce=0, method='ffill').replace(to_replace=0, method='bfill').values
			# 처음 거래량은 0으로 결측값이 되므로 오류! => 그래서 앞이나 뒤의 값을 가져온다.

	for window in data_config.MA_WINDOWS:
		# 종가/이동평균종가 비율
		training_data[f'close_ma{window}_ratio'] = \
			training_data['close'] - training_data[f'close_ma{window}'] / training_data[f'close_ma{window}']

		# 거래량/이동평균거래량 비율
		training_data[f'volume_ma{window}_ratio'] = \
			training_data['volume'] - training_data[f'volume_ma{window}'] / training_data[f'volume_ma{window}']

	return training_data