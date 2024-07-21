#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import random
import re
import time

import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class Crawler:
    def __init__(self):
        self.__time_sleep = 1
        self.__round = 0
        self.__counter = 0
        self.__url_file_path = ""
        self.__img_url_list = []
        chrome_options = Options()  # 用于配置 WebDriver 启动时的选项
        # chrome_options.add_argument("--headless")  # 启动无头模式，这意味着 Chrome 浏览器将以无图形界面的方式运行。这通常用于在服务器上执行自动化任务，因为不需要图形界面
        # chrome_options.add_argument("--disable-gpu")  # 禁用 GPU 硬件加速。这通常用于解决在无头模式下的兼容性问题
        # chrome_options.add_argument("--no-sandbox")  # 禁用沙箱模式。这是为了提高兼容性和避免某些权限问题，尤其是在容器化环境（如 Docker）中运行时。
        # chrome_options.add_argument("--disable-dev-shm-usage")  # 禁用 /dev/shm（共享内存）使用。这是为了避免在某些环境（如容器）中由于共享内存不足而导致的问题。

        verify_chrome_options = Options()  # 用于配置 WebDriver 启动时的选项
        verify_chrome_options.add_argument("--headless")  # 启动无头模式，这意味着 Chrome 浏览器将以无图形界面的方式运行。这通常用于在服务器上执行自动化任务，因为不需要图形界面

        # 明确指定ChromeDriver路径
        chrome_service = Service(executable_path='G:\\resources\\chromedriver-win64\\chromedriver.exe')

        try:
            self.search_driver = webdriver.Chrome(service=chrome_service, options=chrome_options)  # 创建一个 Chrome WebDriver 实例，使用之前配置的选项和服务。
            print("ChromeDriver 启动成功")
            self.verify_driver = webdriver.Chrome(service=chrome_service, options=verify_chrome_options)  # 创建一个 Chrome WebDriver 实例，使用之前配置的选项和服务。
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
        print(f'【睡眠时间: {sleep_time}】')

    def save_image(self, url, word):
        original_path = "G:\\data\\images\\download\\"
        if not os.path.exists(original_path + word):
            os.mkdir(original_path + word)
        self.__counter = len(os.listdir(original_path + word))

        try:
            # 使用requests库下载图像
            response = requests.get(url)
            image_data = response.content
            suffix = self.get_suffix(url)
            filepath = original_path + word + "\\" + str(self.__counter) + suffix
            with open(filepath, 'wb') as f:
                f.write(image_data)
            if os.path.getsize(filepath) < 100:
                print("下载到了空文件，跳过!")
                os.unlink(filepath)
            else:
                print(url)
                print(f"图片+1, 已有 {self.__counter} 张图片")
                self.write_url_to_file(f"【{self.__counter}】" + url, self.__url_file_path)
                self.__img_url_list.append(url)
                self.__counter += 1
        except Exception as e:
            print(f"未知错误: {e}")

    def get_images(self, word, round):

        original_url = 'https://www.google.com.hk/search?q=' + word + '&tbm=isch'
        self.search_driver.get(original_url)
        time.sleep(3)
        # 记录爬取当前的所有url
        self.__url_file_path = os.path.join(r"G:\data\images\download", word, "url.txt")
        # 确保目录存在
        os.makedirs(os.path.dirname(self.__url_file_path), exist_ok=True)
        # 检查文件是否存在，如果不存在则创建一个空文件
        if not os.path.isfile(self.__url_file_path):
            print(f"文件 {self.__url_file_path} 不存在。正在创建一个空的文件。")
            with open(self.__url_file_path, 'w', encoding='utf-8') as file:
                pass  # 创建一个空文件
        with open(self.__url_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.__img_url_list.append(line)
        pos = 0
        for i in range(round):
            try:
                # 获取谷歌图片所在的标签名，即'img'
                img_elements = self.search_driver.find_elements(by=By.TAG_NAME, value='img')
                # 遍历抓到的所有webElement
                for index, img_element in enumerate(img_elements):
                    try:
                        if isinstance(img_element.get_attribute('src'), str):
                            try:
                                # 过滤掉无效的url, 将无效goole图标筛去, 每次爬取当前窗口，或许会重复，因此进行去重
                                if self.check_url(img_element.get_attribute('src'), self.__img_url_list) is False:
                                    continue
                                new_click_driver = self.get_new_driver_by_copy_driver(self.search_driver)
                                # 获取谷歌图片所在的标签名，即'img'
                                new_click_img_elements = new_click_driver.find_elements(by=By.TAG_NAME, value='img')
                                new_click_img_element = new_click_img_elements[index]
                                new_click_img_element.click()
                                time.sleep(2)
                                page = new_click_driver.page_source
                                # 使用正则表达式查找所有符合条件的URL
                                pattern = re.compile(r'"(https://[^"]*?\.(?:jpg|jepg))"')
                                matches = pattern.findall(page)
                                # 打印所有找到的URL
                                for img_url in matches:
                                    # 过滤掉无效的url, 将无效goole图标筛去, 每次爬取当前窗口，或许会重复，因此进行去重
                                    if self.check_url(img_url, self.__img_url_list) is False:
                                        continue
                                    # 下载并保存图片到当前目录下
                                    self.save_image(img_url, word)
                                    # 防止反爬机制
                                    self.sleep()
                            except (Exception):
                                print("failure")
                                continue
                    except Exception as e:
                        print("failure")
                        continue

                print("完成当前页面下载：{}".format(i))
                print("开始滚动")
                pos += 500
                # 向下滑动
                js = 'var q=document.documentElement.scrollTop=' + str(pos)
                # 执行js代码，使滚动条每次滚动500像素
                self.search_driver.execute_script(js)
                # 执行完滚动条之后等待5秒
                time.sleep(5)
                print("滚动完成")

            except (Exception):
                print("failure")
                continue

    def get_new_driver_by_copy_driver(self, origianl_driver):
        # 保存当前页面的 URL
        current_url = origianl_driver.current_url
        # 保存 cookies
        cookies = origianl_driver.get_cookies()
        # 创建第二个 WebDriver 实例
        chrome_options = Options()  # 用于配置 WebDriver 启动时的选项
        # 明确指定ChromeDriver路径
        chrome_service = Service(executable_path='G:\\resources\\chromedriver-win64\\chromedriver.exe')
        new_driver = webdriver.Chrome(service=chrome_service, options=chrome_options)  # 创建一个 Chrome WebDriver 实例，使用之前配置的选项和服务。
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

    def check_url(self, url, img_url_list):
        for used_img_url in img_url_list:
            if url in used_img_url:
                return False
        filter_list = ["/ui/", "icon", "googlelogo", "googleadservices", ".svg"]
        for filter_element in filter_list:
            if filter_element in url:
                return False
        return True

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
        self.driver.quit()


if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument("-w", "--word", type=str, help="抓取关键词", required=True)
    #     parser.add_argument("-r", "--round", type=int, help="滑动次数", required=True)
    #     args = parser.parse_args()
    #
    #     crawler = Crawler()
    #     crawler.start(args.word, args.round)
    # else:
    #     crawler = Crawler()
    #     crawler.start('美女', 2)

    print("请输入搜索关键词：")
    word = input().strip()
    crawler = Crawler()
    crawler.start(word, 10)
