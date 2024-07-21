# log_config.py

import logging
import os
from datetime import datetime


def setup_logging():
    # 获取工程根目录并创建日志目录（如果不存在）
    project_root = "G:\\01Python\\03Project\\learner\\"

    # 获取当前日期
    current_date = datetime.now().strftime('%Y%m%d')

    # 创建包含当前日期的日志目录
    log_dir = os.path.join(project_root, f'logs/{current_date}')
    os.makedirs(log_dir, exist_ok=True)

    # 获取当前时间并设置日志文件路径
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'selenium_{current_time}.log')

    # 配置日志记录
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    stream_handler = logging.StreamHandler()

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 设置格式化器到处理器
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 创建日志记录器并设置级别
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
