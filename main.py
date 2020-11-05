import logging
from logging import log
import os
import settings
import data_manager
from policy_learner import PolicyLearner
import data_config

from common import get_time_str

def _init_log_things(filename):
	"""로그 파일 핸들러 초기화
	"""
	file_handler = logging.FileHandler(filename=filename, encoding='utf-8')
	file_handler.setLevel(logging.DEBUG)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.INFO)

	logging.basicConfig(format='%(message)s', handlers=[file_handler, stream_handler], level=logging.DEBUG)


if __name__ == '__main__':
	stock_code = '104040' # 대성 파인텍

	# 로그가 저장될 폴더 이름 (없으면 생성)
	log_dir = os.path.join(settings.BASE_DIR, f'logs\\{stock_code}')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	timestr = get_time_str()

	# 로그가 저장될 파일 이름
	log_filename = os.path.join(log_dir, f'{stock_code}_{timestr}.log')
	_init_log_things(log_filename)

	# 주식 데이터 준비
	chart_data_date = '2020_11_05'
	chart_data_filename = os.path.join(settings.BASE_DIR, f'chart_data\\{chart_data_date}_{stock_code}.csv')

	chart_data = data_manager.load_chart_data(chart_data_filename)
	prep_data = data_manager.preprocess(chart_data)
	training_data = data_manager.build_training_data(prep_data)

	# 기간 필터링
	training_data = training_data[(training_data['date'] >= '2019-01-01') &
																(training_data['date'] <= '2020-11-05')]
	training_data = training_data.dropna() # 1개의 요소라도 비어있는 행을 지움

	# 차트 데이터 분리
	features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
	chart_data = training_data[features_chart_data]

	# 학습 데이터 분리
	features_training_data = [
		'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
		'close_lastclose_ratio', 'volume_lastvolume_ratio'
	]
	for window in data_config.MA_WINDOWS:
		features_training_data.append(f'close_ma{window}_ratio')
		features_training_data.append(f'volume_ma{window}_ratio')
	
	# 학습에 사용할 데이터만 뽑아냄
	training_data = training_data[features_training_data]

	# 강화학습 시작
	policy_learner = PolicyLearner(
		stock_code=stock_code, chart_data=chart_data, training_data=training_data,
		min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=2, lr=.001)
	policy_learner.fit(balance=10000000, num_epoches=1000, discount_factor=0, start_epsilon=.5)

	# 정색 신경망을 파일로 저장
	model_dir = os.path.join(settings.BASE_DIR, f'models\\{stock_code}')
	model_path = os.path.join(model_dir, f'model_{timestr}.h5')
	policy_learner.policy_network.save_model(model_path)

	
