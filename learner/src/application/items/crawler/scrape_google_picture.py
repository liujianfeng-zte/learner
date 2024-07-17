#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import random
import re
import time

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
        chrome_options = Options()  # 用于配置 WebDriver 启动时的选项
        # chrome_options.add_argument("--headless")  # 启动无头模式，这意味着 Chrome 浏览器将以无图形界面的方式运行。这通常用于在服务器上执行自动化任务，因为不需要图形界面
        # chrome_options.add_argument("--disable-gpu")  # 禁用 GPU 硬件加速。这通常用于解决在无头模式下的兼容性问题
        # chrome_options.add_argument("--no-sandbox")  # 禁用沙箱模式。这是为了提高兼容性和避免某些权限问题，尤其是在容器化环境（如 Docker）中运行时。
        # chrome_options.add_argument("--disable-dev-shm-usage")  # 禁用 /dev/shm（共享内存）使用。这是为了避免在某些环境（如容器）中由于共享内存不足而导致的问题。

        # 明确指定ChromeDriver路径
        chrome_service = Service(executable_path='G:\\resources\\chromedriver-win64\\chromedriver.exe')

        try:
            self.driver = webdriver.Chrome(service=chrome_service, options=chrome_options)  # 创建一个 Chrome WebDriver 实例，使用之前配置的选项和服务。
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

        max_retries = 2
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
                    if os.path.getsize(filepath) < 100:
                        print("下载到了空文件，跳过!")
                        os.unlink(filepath)
                    else:
                        print(img_url)
                        print(f"图片+1, 已有 {self.__counter} 张图片")
                        self.__counter += 1
                    break  # 成功保存图片，跳出重试循环
                except NoSuchElementException:
                    print("图片元素未找到，跳过保存")
                    break
            except WebDriverException as err:
                print(f"WebDriver 错误")
                print("当前 URL: ", img_url)
                print(f"重试 {attempt + 1} 次")
                self.sleep()  # 在重试之前等待
                if attempt == max_retries - 1:
                    print("超过最大重试次数，放弃保存")
            except Exception as e:
                print(f"未知错误: {e}")
                break

    def get_images(self, word, round):
        original_url = 'https://www.google.com.hk/search?q=' + word + '&tbm=isch'
        self.driver.get(original_url)
        time.sleep(3)
        pos = 0
        for i in range(round):
            try:
                # 记录爬取当前的所有url
                img_url_dic = []
                # 获取谷歌图片所在的标签名，即'img'
                img_elements = self.driver.find_elements(by=By.TAG_NAME, value='img')
                # 遍历抓到的所有webElement
                for img_element in img_elements:
                    # 获取每个标签元素内部的url所在连接
                    url = img_element.get_attribute('src')
                    if isinstance(url, str):
                        try:
                            # 过滤掉无效的url, 将无效goole图标筛去, 每次爬取当前窗口，或许会重复，因此进行去重
                            if self.check_url(url, img_url_dic) is False:
                                continue
                            img_element.click()
                            page = self.driver.page_source
                            # 使用正则表达式查找所有符合条件的URL
                            pattern = re.compile(r'"(https://[^"]*?\.(?:jpg|png))"')
                            matches = pattern.findall(page)

                            # 打印所有找到的URL
                            for img_url in matches:
                                # 过滤掉无效的url, 将无效goole图标筛去, 每次爬取当前窗口，或许会重复，因此进行去重
                                if self.check_url(img_url, img_url_dic) is False:
                                    continue
                                img_url_dic.append(img_url)
                                # 下载并保存图片到当前目录下
                                if self.is_valid_url(img_url):
                                    self.save_image(img_url, word)
                                else:
                                    print(f"无效 URL，跳过: {img_url}")
                                # 防止反爬机制
                                self.sleep()
                        except (Exception):
                            print("failure")
                            continue

                print("完成当前页面下载：{}".format(i + 1))
                self.driver.get(original_url)
                time.sleep(3)
                print("开始滚动")
                pos += 500
                # 向下滑动
                js = 'var q=document.documentElement.scrollTop=' + str(pos)
                # 执行js代码，使滚动条每次滚动500像素
                self.driver.execute_script(js)
                # 执行完滚动条之后等待3秒
                time.sleep(3)
                print("滚动完成")

            except (Exception):
                print("failure")
                continue


    def is_valid_url(self, url):
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 2).until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
            return True
        except Exception:
            return False

    def check_url(self, url, img_url_dic):
        if len(url) > 200 or 'images' not in url or url in img_url_dic:
            return False
        filter_list = ["/ui/", "icon", "googlelogo"]
        for filter_element in filter_list:
            if filter_element in url:
                return False
        return True

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
    crawler.start(word, 3)
