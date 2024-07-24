import requests 
import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

url = 'https://neurips.cc/virtual/2023/papers.html?filter=titles&search=#tab-browse'

# Send a GET request to the webpage
ua = UserAgent()
headers = {
    'User-Agent': ua.random
}
response = requests.get(url, headers=headers)
print(response.text)
html_content = response.content
# Parse the webpage content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Paper links
main = soup.find('main')
li_elements = main.find_all('li')
papers = [li for li in li_elements if li.a and li.a['href'].startswith('/virtual/2023/poster/')]
print(len(papers)) # 3583

titles = []
authors = []
locations = []
abstracts = []

root_url = "https://neurips.cc"
for paper in papers:
    href = paper.a['href']
    url = root_url+href
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(response.content, 'html.parser')
    div_element = soup.find('div', class_='card-header')
    abstracts.append(soup.find('div', id='abstractExample').text.replace("Abstract:", "").strip())
    titles.append(div_element.find('h2', class_='card-title main-title text-center').text.strip())
    author = div_element.find('h3', class_='card-subtitle mb-2 text-muted text-center').text.strip()
    authors.append(author)
    try:
        locations.append(div_element.find('h5', class_='text-center text-muted').text)
    except:
        locations.append("")

# Need to add a column Topic that captures the largest directory it falls under
data = {
    'Title': titles,
    'Authors': authors,
    'Location': locations,
    'Abstract': abstracts
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
# df.to_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_Papers.csv', index=False)


