import os
import copy
import pandas as pd
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

# HUPD Template 2: What were the top {#} {IPCR/CPC categories} with the highest percentage of patent acceptance in {year}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list. 
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

# HUPD Template 3: How does the number of patent applications filed in the {quarter1} quarter compare proportionally to those filed in the {quater2} quarter in {year}?
def compare_applications(quarter_1, quarter_2, year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))
    def get_quarter(dt):
        month = dt.month
        if 1 <= month <= 3:
            return 1
        elif 4 <= month <= 6:
            return 2
        elif 7 <= month <= 9:
            return 3
        elif 10 <= month <= 12:
            return 4
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['quarter'] = df['filing_date'].apply(get_quarter)
    quarter_counts = df['quarter'].value_counts()
    del df
    return quarter_counts.get(quarter_1, 0) / quarter_counts.get(quarter_2, 0)

# HUPD Template 4: What is the title of the patent filed between {start_year} and {end_year} that took the longest time to be published?
def longest_time(start_year, end_year):
    df_lst = []
    for year in range(start_year, end_year+1):
        df_year = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))
        df_lst.append(df_year)
    df = pd.concat(df_lst, axis=0, ignore_index=True)
    del df_lst
    df["date_published"] = pd.to_datetime(df["date_published"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["duration"] = df["date_published"]-df["filing_date"]
    sorted_df = df.sort_values(by="duration", ascending=False).reset_index()  
    del df  
    return sorted_df.at[0,"title"]

# NeurIPS Template 1: Who were the top {#} authors with the most publications at NeurIPS 2023?
def top_authors(num, llm_keyword):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
    if llm_keyword:
        df["keyword"] = df["Title"].str.contains("Large Language Models") 
        df = df[df["keyword"]==True]
    df['Authors'] = df['Authors'].apply(ast.literal_eval)
    # df['Authors'] = df['Authors'].str.split(' · ')
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
    df["author_num"] = df["Authors"].str.split("").apply(len)
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
    while question_id<=1: # 600 
        question_type = random.randint(1,6) 
        if question_type == 1:
            # What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
            start_year = random.randint(2014,2018)
            end_year = random.randint(start_year,2018)
            question_phrasings = ["What was the average time between the filing and issuance of patents from {} to {}? Return a float representing the number of days.", 
                                "What was the average duration between the filing and issuance of patents from {} to {}? Return a float representing the number of days.", 
                                "What was the typical time span between the submission and approval of patents from {} to {}? Return a float representing the number of days."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year, end_year)
            answer = average_pendency(start_year, end_year)
        elif question_type == 2:
            # What were the top {#} {IPCR/CPC categories} with the highest percentage of patent acceptance in {year}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list. 
            num = random.randint(2,5)
            category = random.choice(["IPCR categories", "CPC categories"]) 
            category_to_col = {"IPCR categories": "ipcr_category", "CPC categories": "cpc_category"}
            year = random.randint(2014,2018) 
            question_choice = random.randint(0,1)
            if question_choice==0:
                question = "What were the top {} {} with the highest percentage of patent acceptance in {}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list. Use the '{}' column.".format(num, category, year, category_to_col[category])
            else:
                question = "Which {} were among the top {} with the highest percentage of patent approvals in {}? Calculate the approval percentage for each category, then return the top categories with the highest approval rates as a list. Use the '{}' column.".format(category, num, year, category_to_col[category])
            answer = top_accepted_category(num, category, year)
        elif question_type == 3:
            # How does the number of patent applications filed in the {quarter1} quarter compare proportionally to those filed in the {quater2} quarter in {year}?
            quarter_map = {1:"1st", 2:"2nd", 3:"3rd", 4:"4th"}
            quarter_1 = random.randint(1,4)
            quarter_2 = random.randint(1,4)
            while quarter_2==quarter_1:
                quarter_2 = random.randint(1,4)
            year = random.randint(2014,2017)
            question_phrasings = ["How does the number of patent applications filed in the {} quarter compare proportionally to those filed in the {} quarter in {}?", "What's the ratio of patent applications filed in the {} quarter to those filed in the {} quarter in {}?", "What is the ratio between the number of patent applications filed in the {} quarter and the {} quarter in {}?"]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(quarter_map[quarter_1], quarter_map[quarter_2], year)
            answer = compare_applications(quarter_1, quarter_2, year)
        elif question_type == 4:
            # What is the title of the patent filed between {start_year} and {end_year} that took the longest time to be published?
            start_year = random.randint(2014,2017) # not include 2018, as most applications are still pending
            end_year = random.randint(start_year,2017)
            question_phrasings = ["What is the title of the patent filed between {} and {} that took the longest time to be published?", "What is the title of the patent filed between {} and {} that had the longest publication delay?", "What is the title of the patent filed between {} and {} with the longest time elapsed between filing and publication?"]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year, end_year)
            answer = longest_time(start_year, end_year)
        elif question_type == 5:
            # Who were the top {#} authors with the most publications at NeurIPS 2023?
            num = random.randint(2,10)
            llm_keyword = random.choice([True, False])
            question_phrasings = ["Who were the top {} authors with the most publications at NeurIPS 2023? Break ties alphabetically, return as a list.", "Who were the top {} authors with the highest number of publications at NeurIPS 2023? Break ties alphabetically, return as a list.", "Which {} authors had the most publications at NeurIPS 2023? Break ties alphabetically, return as a list."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(num)
            if llm_keyword:
                insert_pos = question.find("publications")+len("publications")
                question = question[:insert_pos]+" containing 'Large Language Models' in the title"+question[insert_pos:]
            answer = top_authors(num, llm_keyword)
        else:
            # What proportion of papers have {compare} {n} authors? Return a value between 0 and 1.
            compare = random.choice(["more than", "fewer than", "exactly", "greater than or equal to", "fewer than or equal to"]) 
            n = random.randint(2,10)
            question_phrasings = ["What proportion of papers have {} {} authors? Return a value between 0 and 1.", "What percentage of papers have {} {} 5 authors? Return a value between 0 and 1.", "What's the ratio of papers that have {} {} 5 authors? Return a value between 0 and 1."] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(compare,n)
            answer = author_num(compare,n)
        # use None to signify not adding to the questions / answers
        if answer:
            writer.write({"qid": "easy-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
            question_id += 1

