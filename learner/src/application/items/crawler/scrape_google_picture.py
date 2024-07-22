#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import random
import re
import tempfile
import time

import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.infrastructure.utils.log_utils import setup_logging

logger = setup_logging()

class Crawler:
    def __init__(self):
        self.__time_sleep = 1
        self.__round = 0
        self.__counter = 0
        self.__url_file_path = ""
        self.__new_click_url_path = ""
        self.__original_url_path = ""
        self.__small_url_path = ""
        self.__img_url_list = []
        self.__new_click_img_url_list = []
        self.__original_img_url_list = []
        self.__small_img_url_list = []
        self.__new_click_driver = None

        # self.__chrome_options = Options()  # 用于配置 WebDriver 启动时的选项
        # chrome_options.add_argument("--headless")  # 启动无头模式，这意味着 Chrome 浏览器将以无图形界面的方式运行。这通常用于在服务器上执行自动化任务，因为不需要图形界面
        # chrome_options.add_argument("--disable-gpu")  # 禁用 GPU 硬件加速。这通常用于解决在无头模式下的兼容性问题
        # chrome_options.add_argument("--no-sandbox")  # 禁用沙箱模式。这是为了提高兼容性和避免某些权限问题，尤其是在容器化环境（如 Docker）中运行时。
        # chrome_options.add_argument("--disable-dev-shm-usage")  # 禁用 /dev/shm（共享内存）使用。这是为了避免在某些环境（如容器）中由于共享内存不足而导致的问题。

        try:
            self.search_driver = self.create_webdriver_instance()  # 创建一个 Chrome WebDriver 实例，使用之前配置的选项和服务。
            print("ChromeDriver 启动成功")
        except Exception as e:
            print(f"启动ChromeDriver时出错: {e}")

    @staticmethod
    def get_suffix(name):
        m = re.search(r'\.[^\.]*$', name)
        if m and m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    def sleep(self):
        sleep_time = round(self.__time_sleep + random.uniform(0, 1), 1)
        time.sleep(sleep_time)
        # logger.info(f'【睡眠时间: {sleep_time}】')

    def save_image(self, url, word):
        original_path = "G:\\data\\images\\download\\"
        if not os.path.exists(original_path + word):
            os.mkdir(original_path + word)
        self.__counter = len(os.listdir(original_path + word)) - 3

        try:
            self.save_driver = self.create_webdriver_instance()
            self.save_driver.get(url)
            time.sleep(2)
            try:
                i = 0
                logger.info(f"开始准备下载图片")
                wait = WebDriverWait(self.save_driver, 5)
                image_elements = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "img")))
                for image_element in image_elements:
                    # 获取图像的URL（注意：这里假设每个<img>标签都有src属性）
                    img_url = image_element.get_attribute('src')
                    # 过滤掉无效的url, 将无效goole图标筛去, 每次爬取当前窗口，或许会重复，因此进行去重
                    check_result, mes = self.check_url(img_url, self.__img_url_list)
                    if check_result is False:
                        if mes == "exist":
                            logger.warning(f"当前url已存在url.txt中：{img_url}")
                        continue
                    check_result, mes = self.check_url(img_url, self.__small_img_url_list)
                    if check_result is False:
                        if mes == "exist":
                            logger.warning(f"当前url已存在small_url.txt中：{img_url}")
                        continue
                    # 使用requests库下载图像
                    response = requests.get(img_url)
                    image_data = response.content
                    suffix = self.get_suffix(img_url)
                    filepath = original_path + word + "\\" + str(self.__counter) + suffix
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    if os.path.getsize(filepath) / 1024 < 20:
                        # logger.info(f"下载到了空文件，跳过!,url:{img_url}")
                        self.write_url_to_file(img_url, self.__small_url_path)
                        os.unlink(filepath)
                    else:
                        logger.info(f"下载图片:{self.__counter}, url：{img_url}")
                        self.write_url_to_file(f"【{self.__counter}】" + img_url, self.__url_file_path)
                        self.__img_url_list.append(img_url)
                        self.__counter += 1
                        i += 1
                logger.info(f"下载图片结束, 本次共计下载图片{i}张")
            except NoSuchElementException:
                logger.error("图片元素未找到，跳过保存")
        except Exception as e:
            logger.error(f"下载失败，未知错误: {e}")
        finally:
            self.save_driver.quit()

    def download_images(self, img_url):
        # 过滤掉无效的url, 将无效goole图标筛去, 每次爬取当前窗口，或许会重复，因此进行去重
        check_result, mes = self.check_url(img_url, self.__new_click_img_url_list)
        if check_result is False:
            if mes == "exist":
                logger.warning(f"当前url已存在new_click_url.txt中：{img_url}")
            return
        # 下载并保存图片到当前目录下
        self.save_image(img_url, word)
        self.write_url_to_file(img_url, self.__new_click_url_path)
        self.__new_click_img_url_list.append(img_url)
        # 防止反爬机制
        self.sleep()

    def get_images(self, word, round):

        original_url = 'https://www.google.com.hk/search?q=' + word + '&tbm=isch'
        self.search_driver.get(original_url)
        time.sleep(3)
        self.__url_file_path = os.path.join(r"G:\data\images\download", word, "url.txt")
        self.__img_url_list = self.url_file_reader(self.__url_file_path)

        self.__new_click_url_path = os.path.join(r"G:\data\images\download", word, "new_click_url.txt")
        self.__new_click_img_url_list = self.url_file_reader(self.__new_click_url_path)

        self.__original_url_path = os.path.join(r"G:\data\images\download", word, "original_url.txt")
        self.__original_img_url_list = self.url_file_reader(self.__original_url_path)

        self.__small_url_path = os.path.join(r"G:\data\images\download", word, "small_url.txt")
        self.__small_img_url_list = self.url_file_reader(self.__small_url_path)


        pos = 2000
        for i in range(round):
            try:
                logger.info(f"开始第{i + 1}轮测试")
                try:
                    self.__new_click_driver = self.get_new_driver_by_copy_driver(self.search_driver)
                    if i > 0:
                        self.element_scroll(self.__new_click_driver, pos * i)
                    # 获取谷歌图片所在的标签名，即'img'
                    logger.info(f"开始点击获取大图, 当前轮次:{i}")
                    # 显式等待元素可见并可点击
                    wait = WebDriverWait(self.__new_click_driver, 5)
                    new_click_img_elements = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "img")))
                    for img_element in new_click_img_elements:
                        original_img_url = img_element.get_attribute('src')
                        # 过滤掉无效的url, 将无效goole图标筛去, 每次爬取当前窗口，或许会重复，因此进行去重
                        check_result, mes = self.check_url(original_img_url, self.__original_img_url_list)
                        if check_result is False:
                            if mes == "exist":
                                logger.warning(f"当前url已存在original_url.txt中：{original_img_url}")
                            continue
                        try:
                            img_element.click()
                        except Exception as e:
                            self.__new_click_driver.quit()
                            logger.error(f"点击元素失败:{original_img_url}")
                            continue
                        time.sleep(3)
                        page = self.__new_click_driver.page_source
                        # 使用正则表达式查找所有符合条件的URL
                        pattern = re.compile(r'"(https://[^"]*?(?:\.jpg|\.jpeg|/images/|images\.)[^"]*?)"')
                        matches = pattern.findall(page)
                        # 打印所有找到的URL
                        i = 0
                        logger.info("开始匹配图片")
                        for img_url in matches:
                            self.download_images(img_url)
                            i += 1
                        logger.info(f"本次匹配图片{i}张")
                        self.write_url_to_file(original_img_url, self.__original_url_path)
                except Exception as e:
                    logger.error(f"查找图片url失败:{e}")
                    continue
                finally:
                    self.__new_click_driver.quit()
                logger.info(f"退出当前轮次:{i}")

            except Exception as e:
                logger.error(f"查找原始img_elements失效:{e}")
                continue

    def element_scroll(self, driver, pos):
        logger.info("开始滚动")
        for i in range(pos // 1000):
            # 向下滑动
            js = f'var q=document.documentElement.scrollTop={(i + 1) * 1000}'
            # 执行js代码，使滚动条每次滚动500像素
            driver.execute_script(js)
            # 执行完滚动条之后等待3秒
            time.sleep(3)
        logger.info("滚动完成")

    def get_new_driver_by_copy_driver(self, origianl_driver):
        # 保存当前页面的 URL
        current_url = origianl_driver.current_url
        # 保存 cookies
        cookies = origianl_driver.get_cookies()
        # 明确指定ChromeDriver路径
        new_driver = self.create_webdriver_instance()  # 创建一个 Chrome WebDriver 实例，使用之前配置的选项和服务。
        # 打开相同的 URL
        new_driver.get(current_url)
        # 应用 cookies
        for cookie in cookies:
            new_driver.add_cookie(cookie)
        # 刷新页面以应用 cookies
        new_driver.refresh()
        # 等待页面加载完成
        time.sleep(3)
        return new_driver

    def url_file_reader(self, file_path):
        url_list = []
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 检查文件是否存在，如果不存在则创建一个空文件
        if not os.path.isfile(file_path):
            logger.info(f"文件 {file_path} 不存在。正在创建一个空的文件。")
            with open(file_path, 'w', encoding='utf-8') as file:
                pass  # 创建一个空文件
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                url_list.append(line)
        return url_list

    def create_webdriver_instance(self):
        chrome_options = Options()
        user_data_dir = tempfile.mkdtemp()  # 创建临时目录
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
        chrome_service = Service(executable_path='G:\\resources\\chromedriver-win64\\chromedriver.exe')
        return webdriver.Chrome(service=chrome_service, options=chrome_options)

    def check_url(self, url, img_url_list):
        if not isinstance(url, str):
            return False, None
        for used_img_url in img_url_list:
            if url in used_img_url:
                return False, "exist"
        filter_list = ["/ui/", "icon", "googlelogo", "googleadservices", ".svg", "data.image/"]
        for filter_element in filter_list:
            if filter_element in url:
                return False, "skip"
        return True, ""

    def write_url_to_file(self, url, filename):
        """
        将URL写入到指定的txt文件中，每次写入都会换行。

        :param url: 要写入的URL字符串。
        :param filename: txt文件的名称，默认为'urls.txt'。
        """
        # 以追加模式打开文件
        with open(filename, 'a', encoding='utf-8') as file:
            # 写入URL，并在末尾添加换行符
            file.write(url + '\n')

    def start(self, word, round):
        self.__round = round
        self.get_images(word, round)
        self.search_driver.quit()


if __name__ == '__main__':
    logger.info("请输入搜索关键词：")
    word = input().strip()
    crawler = Crawler()
    crawler.start(word, 20)
