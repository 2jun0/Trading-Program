from datetime import datetime

def get_time_str():
  now = datetime.now()
  return now.strftime('%Y-%m-%d %H.%M.%S')

def get_date_str():
  now = datetime.now()
  return now.strftime('%Y-%m-%d')