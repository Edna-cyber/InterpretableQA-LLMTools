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

def convert_date(series):
    if series.dtype==np.float64:
        series = series.astype('Int64')
    series = series.astype(str)
    if series.str.contains("-").any(): 
        series = pd.to_datetime(series)
    else:
        series = pd.to_datetime(series, format="%Y%m%d", errors='coerce')
    return series

# HUPD Template 1: What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
def average_pendency(start_year, end_year):
    df_lst = []
    for year in range(start_year, end_year+1):
        df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))
        df["filing_date"] = convert_date(df['filing_date'])
        df["patent_issue_date"] = convert_date(df['patent_issue_date'])
        df_lst.append(df)
    df = pd.concat(df_lst, ignore_index=True)
    df['duration'] = (df['patent_issue_date'] - df['filing_date']).dt.days
    average_duration = df['duration'].mean()
    
    return average_duration

# HUPD Template 2: What were the top {#} {IPCR/CPC categories} with the highest number of accepted patents in {year}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list of {IPCR/CPC categories}. 
def top_accepted_category(category, year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))
    if category=="IPCR categories":
        cat = "main_ipcr_label"
    else:
        cat = "main_cpc_label"
    col = cat.replace("label", "code")
    df[col] = df[cat].apply(lambda x:x[:3] if isinstance(x, str) else x)
    df["acceptance"] = df["decision"].apply(lambda x: 1 if x =="ACCEPTED" else 0)
    top_categories = df.groupby(col)['acceptance'].sum().reset_index()
    del df
    top_categories = top_categories.sort_values(by="acceptance", ascending=False)
    acceptance_sums = top_categories['acceptance'].tolist()
    max_acceptance_sum = acceptance_sums[0]
    cnt = 0
    while acceptance_sums[cnt]==max_acceptance_sum:
        cnt += 1
    top_n = top_categories.head(cnt).index.tolist()
    return cnt, top_n

# HUPD Template 3: # How does the number of patent applications filed in {year1} compare proportionally to those filed in the {year2}?
def compare_applications_year(year1, year2):
    df1 = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year1))))
    len_year1 = len(df1)
    del df1
    df2 = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year2))))
    len_year2 = len(df2)
    del df2
    return len_year1 / len_year2

# HUPD Template 4: What is the title of the patent filed between {start_year} and {end_year} that took the longest number of days to be published?
def longest_time(start_year, end_year):
    df_lst = []
    for year in range(start_year, end_year+1):
        df_year = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))
        df_lst.append(df_year)
    df = pd.concat(df_lst, axis=0, ignore_index=True)
    del df_lst
    df["date_published"] = convert_date(df["date_published"])
    df["filing_date"] = convert_date(df["filing_date"])
    df["duration"] = (df["date_published"]-df["filing_date"]).dt.days()
    sorted_df = df.sort_values(by="duration", ascending=False).reset_index()  
    del df
    durations = sorted_df["duration"].tolist()
    max_duration = durations[0]
    i = 0
    if durations[i]==max_duration:
        i += 1
    return set(durations[:i])

# NeurIPS Template 1: Who were the top {#} authors with the most publications {containing 'Large Language Models' in the title} at NeurIPS? 
def top_authors(row_num, llm_keyword):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
    df = df.iloc[:row_num+1]
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
    sorted_author_df = author_df.sort_values(by='Number of Papers', ascending=False)
    del author_df
    num_papers_lst = sorted_author_df['Number of Papers'].tolist()
    max_num_papers = num_papers_lst[0]
    new_num = 0
    while num_papers_lst[new_num]==max_num_papers:
        new_num += 1
    return new_num, sorted_author_df.head(new_num)['Author'].tolist()

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
question_type_count = {1:100, 2:100, 3:100, 4:100, 5:100, 6:100}
question_types = [1,2,3,4,5,6]
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/easy.jsonl', mode='w') as writer:
    while question_id<=60:
        question_type = random.choice(question_types) 
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
            if answer and not np.isnan(answer):
                writer.write({"qid": "easy-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[1] -= 1
                if question_type_count[1]==0:
                    question_types.remove(1)
                question_id += 1
        elif question_type == 2:
            # What were the top {#} {IPCR/CPC categories} with the highest number of accepted patents in {year}? Return them as a list of {IPCR/CPC categories}.
            category = random.choice(["IPCR categories", "CPC categories"]) 
            year = random.randint(2004,2018) 
            question_choice = random.randint(0,1)
            num, answer = top_accepted_category(category, year)
            if not answer or num!=len(answer): # deal with some datasets with missing values
                continue
            if question_choice==0:
                if num==1:
                    question = "What was the top {} {} with the highest number of accepted patents in {}? Return it as a list of {}.".format(num, category.replace("ies","y"), year, category.replace("ies","y"))
                else:
                    question = "What were the top {} {} with the highest number of accepted patents in {}? Return them as a list of {}.".format(num, category, year, category)
            else:
                if num==1:
                    question = "Which {} was among the top {} with the highest number of accepted patents in {}? Return the top category with the highest approval rates as a list of {}.".format(category.replace("ies","y"), num, year, category.replace("ies","y"))
                else:
                    question = "Which {} were among the top {} with the highest number of accepted patents in {}? Return the top categories with the highest approval rates as a list of {}.".format(category, num, year, category)
            writer.write({"qid": "easy-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
            question_type_count[2] -= 1
            if question_type_count[2]==0:
                question_types.remove(2)
            question_id += 1
        elif question_type == 3:
            # How does the number of patent applications filed in {year1} compare proportionally to those filed in the {year2}?
            year_1 = random.randint(2004,2018)
            year_2 = random.randint(2004,2018)
            while year_2==year_1:
                year_2 = random.randint(2004,2018)
            question = "How does the number of patent applications filed in {} compare proportionally to those filed in the {}? Return a number.".format(year_1, year_2)
            answer = compare_applications_year(year_1, year_2)
            if answer and not np.isnan(answer):
                writer.write({"qid": "easy-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[3] -= 1
                if question_type_count[3]==0:
                    question_types.remove(3)
                question_id += 1
        elif question_type == 4:
            # What is the title of the patent filed between {start_year} and {end_year} that took the longest number of days to be published?
            start_year = random.randint(2004,2017) # not include 2018, as most applications are still pending
            end_year = random.randint(start_year,2017)
            question_phrasings = ["What is the title of the patent filed between {} and {} that took the longest number of days to be published?", "What is the title of the patent filed between {} and {} that had the longest publication delay in terms of number of days?", "What is the title of the patent filed between {} and {} with the longest number of days elapsed between filing and publication?"]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year, end_year)
            answer = longest_time(start_year, end_year)
            if answer:
                writer.write({"qid": "easy-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[4] -= 1
                if question_type_count[4]==0:
                    question_types.remove(4)
                question_id += 1
        elif question_type == 5:
            # Who were the top {#} authors with the most publications at NeurIPS? 
            all_bool = random.choices([True,False], weights=[0.1,0.9], k=1)[0]
            if all_bool:
                row_num = 3585
            else:
                row_num = random.randint(list(range(1000, 3600, 100))) 
            llm_keyword = random.choice([True, False])
            new_num, answer = top_authors(row_num, llm_keyword)
            question_phrasings = ["Who were the top {} authors with the most publications at NeurIPS? In the authors column of the database, each entry is a list, not a single string. Return as a list of authors.", "Who were the top {} authors with the highest number of publications at NeurIPS? In the authors column of the database, each entry is a list, not a single string. Return as a list of authors.", "Which {} authors had the most publications at NeurIPS? In the authors column of the database, each entry is a list, not a single string. Return as a list of authors."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(new_num)
            if llm_keyword:
                insert_pos = question.find("publications")+len("publications")
                question = question[:insert_pos]+" containing 'Large Language Models' in the title"+question[insert_pos:]
            if not all_bool:
                insert_pos = question.find("at NeurIPS")
                question = question[:insert_pos]+"amongst the first {} papers ".format(row_num)+question[insert_pos:]
            if answer:
                writer.write({"qid": "easy-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[5] -= 1
                if question_type_count[5]==0:
                    question_types.remove(5)
                question_id += 1
        else:
            # What proportion of papers have {compare} {n} authors? Return a value between 0 and 1.
            compare = random.choice(["more than", "fewer than", "exactly", "greater than or equal to", "fewer than or equal to"]) 
            n = random.randint(2,10)
            question_phrasings = ["What proportion of papers have {} {} authors? In the authors column of the database, each entry is a list, not a single string. Return a value between 0 and 1.", "What percentage of papers have {} {} authors? In the authors column of the database, each entry is a list, not a single string. Return a value between 0 and 1.", "What's the ratio of papers that have {} {} authors? In the authors column of the database, each entry is a list, not a single string. Return a value between 0 and 1."] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(compare,n)
            answer = author_num(compare,n)
            if answer and not np.isnan(answer):
                writer.write({"qid": "easy-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":answer})
                question_type_count[6] -= 1
                if question_type_count[6]==0:
                    question_types.remove(6)
                question_id += 1

