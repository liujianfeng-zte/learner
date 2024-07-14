from datetime import datetime


def get_current_time():
    # 获取当前日期和时间
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    return current_time

def get_current_date():
    # 获取当前日期和时间
    current_time = datetime.now().strftime('%Y%m%d')
    return current_time