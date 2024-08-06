import os
import copy
import pandas as pd
import numpy as np
import json
import random
import tqdm
import jsonlines
import ast

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"

# HUPD Template 1: What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
def average_pendency(start_year, end_year):
    total_len = 0
    pedencies_sum = 0
    
    for year in range(start_year, end_year+1):
        df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))
        df = df[df["decision"] == "ACCEPTED"]
        total_len += len(df)
        df["filing_date"] = pd.to_datetime(df['filing_date'])
        df["patent_issue_date"] = pd.to_datetime(df['patent_issue_date'])
        # Calculate pendencies directly as days
        pendencies = (df["patent_issue_date"] - df["filing_date"]).dt.days
        pedencies_sum += pendencies.sum()
        del df
    
    return pedencies_sum / total_len

# HUPD Template 2: What were the top {#} {IPCR/CPC categories} with the highest percentage of patent acceptance in {year}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list of {IPCR/CPC categories}. 
def top_accepted_category(num, category, year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))
    if category=="IPCR categories":
        cat = "main_ipcr_label"
    else:
        cat = "main_cpc_label"
    col = cat.replace("label", "code")
    df[col] = df[cat].apply(lambda x:x[:3] if isinstance(x, str) else x)
    df["acceptance"] = df["decision"].apply(lambda x: 1 if x =="ACCEPTED" else 0)
    top_categories = df.groupby(col)['acceptance'].mean().nlargest(num)
    del df
    top_n = top_categories.index.tolist()
    return top_n

# HUPD Template 3: # How does the number of patent applications filed in {year1} compare proportionally to those filed in the {year2}?
def compare_applications_year(year1, year2):
    df1 = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year1))))
    len_year1 = len(df1)
    del df1
    df2 = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year2))))
    len_year2 = len(df2)
    del df2
    return len_year1 / len_year2

# HUPD Template 4: What is the title of the patent filed between {start_year} and {end_year} that took the longest time to be published?
def longest_time(start_year, end_year):
    def convert_date(series):
        if series.dtype==np.float64:
            series = series.astype('Int64')
        series = series.astype(str)
        if "-" in series.iloc[0]:
            series = pd.to_datetime(series)
        else:
            series = pd.to_datetime(series, format="%Y%m%d", errors='coerce')
        return series
    df_lst = []
    for year in range(start_year, end_year+1):
        df_year = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))
        df_lst.append(df_year)
    df = pd.concat(df_lst, axis=0, ignore_index=True)
    del df_lst
    df["date_published"] = convert_date(df["date_published"])
    df["filing_date"] = convert_date(df["filing_date"])
    df["duration"] = df["date_published"]-df["filing_date"]
    sorted_df = df.sort_values(by="duration", ascending=False).reset_index()  
    del df  
    return sorted_df.at[0,"title"]

# NeurIPS Template 1: Who were the top {#} authors with the most publications {containing 'Large Language Models' in the title} at NeurIPS 2023?
def top_authors(num, llm_keyword):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
    if llm_keyword:
        df["keyword"] = df["Title"].str.contains("Large Language Models") 
        df = df[df["keyword"]==True]
    df['Authors'] = df['Authors'].str.split(' · ')
    exploded_df = df.explode('Authors')
    del df
    author_counts = exploded_df['Authors'].value_counts()
    author_df = author_counts.reset_index()
    del exploded_df
    author_df.columns = ['Author', 'Number of Papers']
    sorted_author_df = author_df.sort_values(by=['Number of Papers', 'Author'], ascending=[False, True])
    del author_df
    return sorted_author_df.head(num)['Author'].tolist()

# NeurIPS Template 2: What proportion of papers have {compare} {n} authors? Return a value between 0 and 1.
def author_num(compare,n):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv")) 
    total_papers = len(df)
    df["author_num"] = df["Authors"].str.split(' · ').apply(len)
    if compare=="more than":
        papers = len(df[df["author_num"]>n])
    elif compare=="fewer than":
        papers = len(df[df["author_num"]<n])
    elif compare=="greater than or equal to": 
        papers = len(df[df["author_num"]>=n])
    elif compare=="fewer than or equal to": 
        papers = len(df[df["author_num"]<=n])
    else:
        papers = len(df[df["author_num"]==n])
    del df
    return papers / total_papers

question_id = 1
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/easy.jsonl', mode='w') as writer:
    while question_id<=10: # 600 
        question_type = random.randint(6,6) 
        if question_type == 1:
            # What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
            start_year = random.randint(2004,2018)
            end_year = random.randint(start_year,2018)
            question_phrasings = ["What was the average time between the filing and issuance of patents from {} to {}? Return a float representing the number of days.", 
                                "What was the average duration between the filing and issuance of patents from {} to {}? Return a float representing the number of days.", 
                                "What was the typical time span between the submission and approval of patents from {} to {}? Return a float representing the number of days."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year, end_year)
            answer = average_pendency(start_year, end_year)
            # use None to signify not adding to the questions / answers
            if answer:
                writer.write({"qid": "easy-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_id += 1
        elif question_type == 2:
            # What were the top {#} {IPCR/CPC categories} with the highest percentage of patent acceptance in {year}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list of {IPCR/CPC categories}.
            num = random.randint(2,5)
            category = random.choice(["IPCR categories", "CPC categories"]) 
            year = random.randint(2004,2018) 
            question_choice = random.randint(0,1)
            if question_choice==0:
                question = "What were the top {} {} with the highest percentage of patent acceptance in {}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list of {}.".format(num, category, year, category)
            else:
                question = "Which {} were among the top {} with the highest percentage of patent approvals in {}? Calculate the approval percentage for each category, then return the top categories with the highest approval rates as a list of {}.".format(category, num, year, category)
            answer = top_accepted_category(num, category, year)
            if answer and len(answer)==num: # deal with some datasets with missing values
                writer.write({"qid": "easy-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_id += 1
        elif question_type == 3:
            # How does the number of patent applications filed in {year1} compare proportionally to those filed in the {year2}?
            year_1 = random.randint(2004,2018)
            year_2 = random.randint(2004,2018)
            while year_2==year_1:
                year_2 = random.randint(2004,2018)
            question = "How does the number of patent applications filed in {} compare proportionally to those filed in the {}? Return a number.".format(year_1, year_2)
            answer = compare_applications_year(year_1, year_2)
            if answer:
                writer.write({"qid": "easy-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_id += 1
        elif question_type == 4:
            # What is the title of the patent filed between {start_year} and {end_year} that took the longest time to be published?
            start_year = random.randint(2004,2017) # not include 2018, as most applications are still pending
            end_year = random.randint(start_year,2017)
            question_phrasings = ["What is the title of the patent filed between {} and {} that took the longest time to be published?", "What is the title of the patent filed between {} and {} that had the longest publication delay?", "What is the title of the patent filed between {} and {} with the longest time elapsed between filing and publication?"]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year, end_year)
            answer = longest_time(start_year, end_year)
            if answer:
                writer.write({"qid": "easy-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_id += 1
        elif question_type == 5:
            # Who were the top {#} authors with the most publications at NeurIPS 2023? Break ties alphabetically, return as a list of authors.
            num = random.randint(2,10)
            llm_keyword = random.choice([True, False])
            question_phrasings = ["Who were the top {} authors with the most publications at NeurIPS 2023? Break ties alphabetically, return as a list of authors.", "Who were the top {} authors with the highest number of publications at NeurIPS 2023? Break ties alphabetically, return as a list of authors.", "Which {} authors had the most publications at NeurIPS 2023? Break ties alphabetically, return as a list of authors."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(num)
            if llm_keyword:
                insert_pos = question.find("publications")+len("publications")
                question = question[:insert_pos]+" containing 'Large Language Models' in the title"+question[insert_pos:]
            answer = top_authors(num, llm_keyword)
            if answer:
                writer.write({"qid": "easy-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_id += 1
        else:
            # What proportion of papers have {compare} {n} authors? Return a value between 0 and 1.
            compare = random.choice(["more than", "fewer than", "exactly", "greater than or equal to", "fewer than or equal to"]) 
            n = random.randint(2,10)
            question_phrasings = ["What proportion of papers have {} {} authors? Return a value between 0 and 1.", "What percentage of papers have {} {} authors? Return a value between 0 and 1.", "What's the ratio of papers that have {} {} authors? Return a value between 0 and 1."] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(compare,n)
            answer = author_num(compare,n)
            if answer:
                writer.write({"qid": "easy-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_id += 1

