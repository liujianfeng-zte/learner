import requests
import os

def download_image(folder_path, url, image_number):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            with open(os.path.join(folder_path, f'image_{image_number}.jpg'), 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to download {url} - status code {response.status_code}")
    except Exception as e:
        print(f"Could not download {url} - {e}")