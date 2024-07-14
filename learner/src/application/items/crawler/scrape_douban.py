import requests
from bs4 import BeautifulSoup

for start_num in range(0, 250, 25):
    heards = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"}
    response = requests.get(f'https://movie.douban.com/top250?start={start_num}', headers=heards)
    content = response.text
    soup = BeautifulSoup(content, "html.parser")
    all_titles = soup.findAll("span", attrs={"class": "title"})
    for title in all_titles:
        title_string = title.string
        if "/" not in title_string:
            print(title_string)

