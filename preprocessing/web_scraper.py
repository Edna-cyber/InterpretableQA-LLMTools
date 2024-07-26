import time
import requests 
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

url = 'https://neurips.cc/virtual/2023/papers.html?filter=titles&search=#tab-browse'

# # Send a GET request to the webpage
# response = requests.get(url)
# html_content = response.content
# # Parse the webpage content with BeautifulSoup
# soup = BeautifulSoup(response.content, 'html.parser')

# # Paper links
# main = soup.find('main')
# li_elements = main.find_all('li')
# papers = [li for li in li_elements if li.a and li.a['href'].startswith('/virtual/2023/poster/')]
# print(len(papers)) # 3583

# titles = []
# authors = []
# locations = []
# abstracts = []

# root_url = "https://neurips.cc"
# for paper in papers:
#     href = paper.a['href']
#     url = root_url+href
#     response = requests.get(url)
#     html_content = response.content
#     soup = BeautifulSoup(response.content, 'html.parser')
#     div_element = soup.find('div', class_='card-header')
#     abstracts.append(soup.find('div', id='abstractExample').text.replace("Abstract:", "").strip())
#     titles.append(div_element.find('h2', class_='card-title main-title text-center').text.strip())
#     author = div_element.find('h3', class_='card-subtitle mb-2 text-muted text-center').text.strip()
#     authors.append(author)
#     try:
#         locations.append(div_element.find('h5', class_='text-center text-muted').text)
#     except:
#         locations.append("")

# data = {
#     'Title': titles,
#     'Authors': authors,
#     'Location': locations,
#     'Abstract': abstracts
# }
# df = pd.DataFrame(data)

original_df = pd.read_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_Papers.csv')
titles = original_df['Title']
unvisited_titles = set(titles)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url)
    # Click topic
    page.wait_for_selector('#main > div.container > div > div.d-flex.justify-content-between > div > div > label.btn.btn-outline-secondary.topic-format')
    page.click('#main > div.container > div > div.d-flex.justify-content-between > div > div > label.btn.btn-outline-secondary.topic-format')
    iterations = 0
    while len(unvisited_titles)>0:
        # Click shuffle
        page.wait_for_selector('#main > div.container > div > div.row.mt-3.mb-0 > div.col-auto.d-none.d-lg-inline > div > button')
        page.click('#main > div.container > div > div.row.mt-3.mb-0 > div.col-auto.d-none.d-lg-inline > div > button')
        page.wait_for_selector('div.cards.row') 
        # page.wait_for_selector('h5.card-title')
        title_handle = page.query_selector_all('h5.card-title') 
        visited_titles = set([page.evaluate('(element) => element.innerHTML', query).strip() for query in title_handle])
        unvisited_titles -= visited_titles
        print(len(unvisited_titles))
        iterations += 1
        time.sleep(10)
    print("iterations", iterations)
    # divs = page.query_selector_all('#main > div.container > div > div.cards.row > div')
    # num_divs = len(divs)
    # print(num_divs)
    # titles = page.query_selector_all('#main > div.container > div > div.cards.row > div > div > a')
    # print(titles)
    # content = page.inner_html('div.cards.row')
    # for paper in content:
    #     title = paper.text_content('h5.card-title')
    #     print(title)
    # content = page.content()
    # main_handle = page.query_selector('main')
    # main_content = page.evaluate('(element) => element.innerHTML', main_handle)
    # print(content)
    browser.close()


# Save the DataFrame to a CSV file
# df.to_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_New_Papers.csv', index=False)


