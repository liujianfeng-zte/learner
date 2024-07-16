#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os
import random
import re
import sys
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException


class Crawler:
    def __init__(self):
        self.__time_sleep = 1
        self.__amount = 0
        self.__start_amount = 0
        self.__counter = 0
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # 明确指定ChromeDriver路径
        chrome_service = Service(executable_path='G:\\resources\\chromedriver-win64\\chromedriver.exe')

        try:
            self.driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
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

    def save_image(self, img_url, word):
        if not os.path.exists("./" + word):
            os.mkdir("./" + word)
        self.__counter = len(os.listdir('./' + word)) + 1

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.driver.get(img_url)
                try:
                    image_element = self.driver.find_element(By.TAG_NAME, 'img')
                    image_data = image_element.screenshot_as_png
                    suffix = self.get_suffix(img_url)
                    filepath = './%s/%s' % (word, str(self.__counter) + str(suffix))
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    if os.path.getsize(filepath) < 5:
                        print("下载到了空文件，跳过!")
                        os.unlink(filepath)
                    else:
                        print(f"图片+1, 已有 {self.__counter} 张图片")
                        self.__counter += 1
                    break  # 成功保存图片，跳出重试循环
                except NoSuchElementException:
                    print("图片元素未找到，跳过保存")
                    break
            except WebDriverException as err:
                print(f"WebDriver 错误: {err}")
                print("当前 URL: ", img_url)
                print(f"重试 {attempt + 1} 次")
                self.sleep()  # 在重试之前等待
                if attempt == max_retries - 1:
                    print("超过最大重试次数，放弃保存")
            except Exception as e:
                print(f"未知错误: {e}")
                break

    def get_images(self, word):
        search = word
        pn = self.__start_amount
        while pn < self.__amount:
            url = f'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={search}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word={search}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn={pn}&rn={self.__per_page}&gsm=1e&1594447993172='
            self.driver.get(url)
            try:
                WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                rsp = self.driver.find_element(By.TAG_NAME, 'pre').text
                rsp_data = json.loads(rsp, strict=False)
                if 'data' not in rsp_data:
                    print("触发了反爬机制，自动重试！")
                else:
                    for image_info in rsp_data['data']:
                        if 'replaceUrl' in image_info and len(image_info['replaceUrl']) > 0:
                            obj_url = image_info['replaceUrl'][0]['ObjUrl']
                            if self.is_valid_url(obj_url):
                                self.save_image(obj_url, word)
                            else:
                                print(f"无效 URL，跳过: {obj_url}")
                    print("下载下一页")
                    pn += self.__per_page
                    self.sleep()
            except (TimeoutException, WebDriverException) as e:
                print(f"请求错误: {e}")
                self.sleep()  # 等待一段时间再重试
                continue
        print("下载任务结束")

    def is_valid_url(self, url):
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 2).until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
            return True
        except Exception:
            return False

    def start(self, word, total_page=1, start_page=1, per_page=30):
        self.__per_page = per_page
        self.__start_amount = (start_page - 1) * self.__per_page
        self.__amount = total_page * self.__per_page + self.__start_amount
        self.get_images(word)
        self.driver.quit()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--word", type=str, help="抓取关键词", required=True)
        parser.add_argument("-tp", "--total_page", type=int, help="需要抓取的总页数", required=True)
        parser.add_argument("-sp", "--start_page", type=int, help="起始页数", required=True)
        parser.add_argument("-pp", "--per_page", type=int, help="每页大小",
                            choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], default=30, nargs='?')
        parser.add_argument("-d", "--delay", type=float, help="抓取延时（间隔）", default=0.05)
        args = parser.parse_args()

        crawler = Crawler()
        crawler.start(args.word, args.total_page, args.start_page, args.per_page)
    else:
        crawler = Crawler()
        crawler.start('美女', 10, 2, 30)
