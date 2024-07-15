#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os
import random
import re
import sys
import json
import socket
import requests
import time
from itertools import cycle

timeout = 5
socket.setdefaulttimeout(timeout)

class Crawler:

    def __init__(self):
        # 睡眠时长
        self.__time_sleep = 1
        self.__amount = 0
        self.__start_amount = 0
        self.__counter = 0

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
        'Cookie': '',
        'Referer': 'https://image.baidu.com'
    }
    __per_page = 30

    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
        # 更多User-Agent
    ]

    proxies = [
        # 更多代理
    ]

    @staticmethod
    def get_suffix(name):
        m = re.search(r'\.[^\.]*$', name)
        if m and m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    @staticmethod
    def handle_baidu_cookie(original_cookie, cookies):
        if not cookies:
            return original_cookie
        result = original_cookie
        for cookie in cookies:
            result += cookie.split(';')[0] + ';'
        result = result.rstrip(';')
        return result

    def sleep(self):
        sleep_time = round(self.__time_sleep + random.uniform(0, 3), 1)
        time.sleep(sleep_time)
        print(f'【sleep: {sleep_time}】')

    def save_image(self, rsp_data, word):
        if not os.path.exists("./" + word):
            os.mkdir("./" + word)
        self.__counter = len(os.listdir('./' + word)) + 1
        for image_info in rsp_data['data']:
            try:
                if 'replaceUrl' not in image_info or len(image_info['replaceUrl']) < 1:
                    continue
                obj_url = image_info['replaceUrl'][0]['ObjUrl']
                thumb_url = image_info['thumbURL']
                url = 'https://image.baidu.com/search/down?tn=download&ipn=dwnl&word=download&ie=utf8&fr=result&url=%s&thumburl=%s' % (
                    requests.utils.quote(obj_url), requests.utils.quote(thumb_url))
                self.sleep()
                suffix = self.get_suffix(obj_url)
                self.headers['User-Agent'] = random.choice(self.user_agents)
                proxy_used = None
                use_local_ip = False

                for proxy in self.proxies:
                    try:
                        response = requests.get(url, headers=self.headers, proxies={"http": proxy, "https": proxy},
                                                timeout=5)
                        proxy_used = proxy
                        break
                    except requests.exceptions.RequestException:
                        self.sleep()
                        print(f"代理不可用，切换到下一个代理: {proxy}")

                if not proxy_used:
                    try:
                        response = requests.get(url, headers=self.headers, timeout=5)
                        use_local_ip = True
                    except requests.exceptions.RequestException as e:
                        print(f"使用本地IP请求失败: {e}")
                        continue

                if response.status_code == 200:
                    filepath = './%s/%s' % (word, str(self.__counter) + str(suffix))
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    if os.path.getsize(filepath) < 5:
                        print("下载到了空文件，跳过!")
                        os.unlink(filepath)
                        continue
                else:
                    print(f"HTTP error: {response.status_code}")
                    continue
            except Exception as err:
                time.sleep(1)
                print(err)
                print("产生未知错误，放弃保存")
                continue
            else:
                print(f"图片+1, 已有 {self.__counter} 张图片")
                self.__counter += 1
        return

    def get_images(self, word):
        search = requests.utils.quote(word)
        pn = self.__start_amount
        while pn < self.__amount:
            url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%s&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word=%s&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn=%s&rn=%d&gsm=1e&1594447993172=' % (
                search, search, str(pn), self.__per_page)
            self.headers['User-Agent'] = random.choice(self.user_agents)
            proxy_used = None
            use_local_ip = False

            for proxy in self.proxies:
                try:
                    response = requests.get(url, headers=self.headers, proxies={"http": proxy, "https": proxy},
                                            timeout=5)
                    proxy_used = proxy
                    break
                except requests.exceptions.RequestException:
                    self.sleep()
                    print(f"代理不可用，切换到下一个代理: {proxy}")

            if not proxy_used:
                try:
                    response = requests.get(url, headers=self.headers, timeout=5)
                    use_local_ip = True
                except requests.exceptions.RequestException as e:
                    print(f"使用本地IP请求失败: {e}")
                    continue

            try:
                self.sleep()
                self.headers['Cookie'] = self.handle_baidu_cookie(self.headers['Cookie'], response.cookies.get_dict())
                rsp = response.text
            except requests.exceptions.RequestException as e:
                print(e)
                print("请求错误:", url)
                continue

            rsp_data = json.loads(rsp, strict=False)
            if 'data' not in rsp_data:
                print("触发了反爬机制，自动重试！")
            else:
                self.save_image(rsp_data, word)
                print("下载下一页")
                pn += self.__per_page
        print("下载任务结束")
        return

    def start(self, word, total_page=1, start_page=1, per_page=30):
        self.__per_page = per_page
        self.__start_amount = (start_page - 1) * self.__per_page
        self.__amount = total_page * self.__per_page + self.__start_amount
        self.get_images(word)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--word", type=str, help="抓取关键词", required=True)
        parser.add_argument("-tp", "--total_page", type=int, help="需要抓取的总页数", required=True)
        parser.add_argument("-sp", "--start_page", type=int, help="起始页数", required=True)
        parser.add_argument("-pp", "--per_page", type=int, help="每页大小", choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], default=30, nargs='?')
        parser.add_argument("-d", "--delay", type=float, help="抓取延时（间隔）", default=0.05)
        args = parser.parse_args()

        crawler = Crawler()
        crawler.start(args.word, args.total_page, args.start_page, args.per_page)
    else:
        crawler = Crawler()
        crawler.start('美女', 10, 2, 30)
        # crawler.start('二次元 美女', 10, 1)
        # crawler.start('帅哥', 5)
