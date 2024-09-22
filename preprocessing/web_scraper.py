import time
import requests 
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# url = 'https://neurips.cc/virtual/2023/papers.html?filter=titles&search=#tab-browse'

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

# df = pd.read_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_Papers.csv')
# titles = df['Title']
# print(len(df)) ###
# unvisited_titles = set(titles)
# df['Topic'] = ['' for i in range(len(df))]
# df['Oral'] = [False for i in range(len(df))]
# df['Poster Session'] = ['' for i in range(len(df))]

# orals_url = 'https://nips.cc/virtual/2023/events/oral'
# with sync_playwright() as p:
#     browser = p.chromium.launch()
#     page = browser.new_page()
#     page.goto(orals_url)
#     page.wait_for_selector('div.virtual-card > a')
#     orals_handle = page.query_selector_all('div.virtual-card > a')
#     for oral in orals_handle:
#         which_oral = page.evaluate('(element) => element.innerHTML', oral).strip()
#         try:
#             ind = df.index[df['Title'] == which_oral][0]
#             df.at[ind,'Oral'] = True
#         except:
#             continue
#     browser.close()

# def value_of_query(page, card, selector):
#     query = card.query_selector(selector)
#     if not query:
#         return ''
#     else:
#         return page.evaluate('(element) => element.innerHTML', query).strip()

# with sync_playwright() as p:
#     browser = p.chromium.launch()
#     page = browser.new_page()
#     page.goto(url)
#     # Click topic
#     page.wait_for_selector('#main > div.container > div > div.d-flex.justify-content-between > div > div > label.btn.btn-outline-secondary.topic-format')
#     page.click('#main > div.container > div > div.d-flex.justify-content-between > div > div > label.btn.btn-outline-secondary.topic-format')
#     iteration = 0
#     while len(unvisited_titles)>10:
#         # Click sort
#         page.click('#main > div.container > div > div.row.mt-3.mb-0 > div.col-auto.d-none.d-lg-inline > div > button')
#         page.wait_for_selector('div.myCard.col-sm-6.col-lg-4')  
#         cards_handle = page.query_selector_all('div.myCard.col-sm-6.col-lg-4')
#         for card in cards_handle:
#             title_query = card.query_selector('h5.card-title')
#             title = page.evaluate('(element) => element.innerHTML', title_query).strip()
#             if title not in unvisited_titles:
#                 repeat += 1
#                 continue
#             if '<' in title:
#                 continue
#             unvisited_titles.remove(title)
#             ind = df.index[df['Title'] == title][0]
#             topic = value_of_query(page, card, 'span.text-muted.card-topic > a.has_tippy')
#             df.at[ind,'Topic'] = topic
#             poster_session = value_of_query(page, card, 'div.card-subtitle.text-muted.mt-2 > a')
#             if poster_session=='': 
#                 df.at[ind,'Poster Session'] = poster_session 
#             else:
#                 start_ind = poster_session.find('Poster Session')+len('Poster Session')+1
#                 df.at[ind,'Poster Session'] = poster_session[start_ind:]
#         iteration += 1
#         print("iteration", iteration)
#         print("remaining", len(unvisited_titles))
#         time.sleep(10)
#     print("unvisited_titles", unvisited_titles)
#     print("Total iterations:", iteration)
#     browser.close()

# # Save the DataFrame to a CSV file
# df.to_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_New_Papers.csv', index=False)

# Have to manually add missing data due to the browser's shuffling way of presentation.
# df = pd.read_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_New_Papers.csv')
# print(df.dtypes)
# ind = df.index[df['Title'] == 'Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks'][0]
# df.at[ind,'Topic'] = 'Reinforcement Learning'
# df.at[ind,'Poster Session'] = float('6')
# ind = df.index[df['Title'] == 'Diversify \\& Conquer: Outcome-directed Curriculum RL via Out-of-Distribution Disagreement'][0]
# df.at[ind,'Topic'] = 'Reinforcement Learning/Everything Else'
# df.at[ind,'Poster Session'] = float('5')
# ind = df.index[df['Title'] == 'Masked Two-channel Decoupling Framework for Incomplete Multi-view Weak Multi-label Learning'][0]
# df.at[ind,'Topic'] = 'Deep Learning/Everything Else'
# ind = df.index[df['Title'] == 'Reproducibility Study of "Label-Free Explainability for Unsupervised Models"'][0]
# df.at[ind,'Topic'] = 'Social Aspects/Accountability, Transparency and Interpretability'
# df.at[ind,'Poster Session'] = float('6')
# ind = df.index[df['Title'] == 'Attentive Transfer Entropy to Exploit Transient Emergence of Coupling Effect'][0]
# df.at[ind,'Topic'] = 'Deep Learning/Attention Mechanisms'
# df.at[ind,'Poster Session'] = float('3')
# ind = df.index[df['Title'] == 'Robust Bayesian Satisficing'][0]
# df.at[ind,'Topic'] = 'Optimization/Zero-order and Black-box Optimization'
# df.at[ind,'Poster Session'] = float('6')
# ind = df.index[df['Title'] == 'Graph Clustering with Graph Neural Networks'][0]
# df.at[ind,'Topic'] = 'Deep Learning/Graph Neural Networks'
# df.at[ind,'Poster Session'] = float('6')
# ind = df.index[df['Title'] == 'Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework'][0]
# df.at[ind,'Topic'] = 'Deep Learning/Other Representation Learning'
# df.at[ind,'Poster Session'] = float('3')
# ind = df.index[df['Title'] == 'TopP&R: Robust Support Estimation Approach for Evaluating Fidelity and Diversity in Generative Models'][0]
# df.at[ind,'Topic'] = 'Deep Learning/Generative Models and Autoencoders'
# df.at[ind,'Poster Session'] = float('2')
# ind = df.index[df['Title'] == 'Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack'][0]
# df.at[ind,'Topic'] = 'Social Aspects/Privacy-preserving Statistics and Machine Learning'
# df.at[ind,'Poster Session'] = float('4')
# df.to_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_Newest_Papers.csv', index=False)

# Topic / Subtopic, oral / not oral
# df = pd.read_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_Papers.csv')
# df[['Topic', 'Subtopic']] = df['Topic'].str.split('/', expand=True)
# df['Oral'] = df['Oral'].replace({np.bool_(True): "oral", np.bool_(False): "not oral"})
# df.to_csv('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/NeurIPS_2023_Newest_Papers.csv', index=False)