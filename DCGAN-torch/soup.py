from bs4 import BeautifulSoup
from urllib.request import urlopen

def scrape_from_url(url):
    htmldata = urlopen(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    images = soup.find_all('img')

    if len(images) == 0:
        print("No images found")
        return

    i = 0
    while i < 100:
        print(images[i]['src'])
        i += 1

if __name__ == '__main__':
    scrape_from_url('https://www.wikiart.org/en/paintings-by-style/minimalism?select=featured#!#filterName:featured,viewType:masonry')