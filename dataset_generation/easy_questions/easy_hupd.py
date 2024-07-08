import os
import copy
import pandas as pd
import json
import random
import tqdm
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/hupd/"

# Template 0: What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
def average_pendency(start_year, end_year):
    total_len = 0
    pedencies_sum = 0
    
    for year in range(start_year, end_year+1):
        df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
        df = df[df["decision"] == "ACCEPTED"]
        total_len += len(df)
        df["filing_date"] = pd.to_datetime(df['filing_date'])
        df["patent_issue_date"] = pd.to_datetime(df['patent_issue_date'])
        # Calculate pendencies directly as days
        pendencies = (df["patent_issue_date"] - df["filing_date"]).dt.days
        pedencies_sum += pendencies.sum()
        del df
    
    return pedencies_sum / total_len

# Template 1: What were the top{#} {IPCR/CPC categories} with the highest percentage of patent acceptance in {year}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list. 
def top_accepted_category(num, category, year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
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

# Template 2: How does the number of patent applications filed in the {quarter1} quarter compare proportionally to those filed in the {quater2} quarter in {year}?
def compare_applications(quarter_1, quarter_2, year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
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

# Template 3: What is the title of the patent that took the longest time to be published after filing in {year}?
def longest_time(year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
    df["date_published"] = pd.to_datetime(df["date_published"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["duration"] = df["date_published"]-df["filing_date"]
    sorted_df = df.sort_values(by="duration", ascending=False).reset_index()  
    del df  
    return sorted_df.at[0,"title"]
    
# Template 4: How many examiners reviewed patent applications in every single year between {start_year} and {end_year}?
def common_examiners(start_year, end_year):
    examiners = set()
    for year in range(start_year, end_year+1):
        df = pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year))))
        if not examiners:
            examiners = set(df["examiner_id"].unique())
        else:
            examiners = examiners & set(df["examiner_id"].unique())
        del df
    return len(examiners)

questions = []
question_id = 1
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/easy/hupd-easy.jsonl', mode='w') as writer:
    while question_id<=300: 
        question_type = random.randint(0,4) #(0, 8)
        if question_type == 0:
            # What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
            start_year = random.randint(2015,2018)
            end_year = random.randint(start_year,2018)
            question_phrasings = ["What was the average time between the filing and issuance of patents from {} to {}?", 
                                "What was the average duration between the filing and issuance of patents from {} to {}?", 
                                "What was the typical time span between the submission and approval of patents from {} to {}?"]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year, end_year)
            answer = average_pendency(start_year, end_year)
        elif question_type == 1:
            # What were the top{#} {IPCR/CPC categories} with the highest percentage of patent acceptance in {year}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list. 
            num = random.randint(2,5)
            category = random.choice(["IPCR categories", "CPC categories"]) 
            year = random.randint(2015,2018) 
            question_choice = random.randint(0,1)
            if question_choice==0:
                question = "What were the top{} {} with the highest percentage of patent acceptance in {}? First, calculate the approval percentage for each category, then identify the categories with the highest approval rates and return them as a list.".format(num, category, year)
            else:
                question = "Which {} were among the top{} with the highest percentage of patent approvals in {}? Calculate the approval percentage for each category first, then return the top categories with the highest approval rates as a list.".format(category, num, year)
            answer = top_accepted_category(num, category, year)
        elif question_type == 2:
            # How does the number of patent applications filed in the {quarter1} quarter compare proportionally to those filed in the {quater2} quarter in {year}?
            quarter_map = {1:"1st", 2:"2nd", 3:"3rd", 4:"4th"}
            quarter_1 = random.randint(1,4)
            quarter_2 = random.randint(1,4)
            while quarter_2==quarter_1:
                quarter_2 = random.randint(1,4)
            year = random.randint(2015,2017)
            question_phrasings = ["How does the number of patent applications filed in the {} quarter compare proportionally to those filed in the {} quarter in {}?", "What's the ratio of patent applications filed in the {} quarter to those filed in the {} quarter in {}?", "What is the ratio between the number of patent applications filed in the {} quarter and the {} quarter in {}?"]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(quarter_map[quarter_1], quarter_map[quarter_2], year)
            answer = compare_applications(quarter_1, quarter_2, year)
        elif question_type == 3:
            # What is the title of the patent that took the longest time to be published after filing in {year}?
            year = random.randint(2015,2017) # not include 2018, as most applications are still pending
            question_phrasings = ["What is the title of the patent that took the longest time to be published after filing in {}?", "What is the title of the patent that had the longest publication delay after filing in {}?", "What is the title of the patent with the longest time elapsed between filing and publication in {}?"]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(year)
            answer = longest_time(year)
        elif question_type == 4:
            # How many examiners reviewed patent applications in every single year between {start_year} and {end_year}?
            start_year = random.randint(2015,2017)
            end_year = random.randint(start_year+1,2018)
            question_phrasings = ["How many examiners reviewed patent applications between {} and {}? Each examiner needs to have reviewed applications in all the years in this range. Return a number.", "How many examiners reviewed patent applications consistently in all the years between {} and {}? Each examiner needs to have reviewed applications in all the years in this range. Return a number.", "What's the number of common examiners who reviewed patent applications in all the years from {} to {}? Each examiner needs to have reviewed applications in all the years in this range. Return a number."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year, end_year)
            answer = common_examiners(start_year, end_year)
        # use None to signify not adding to the questions / answers
        if answer:
            writer.write({"qid": "easy-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
            question_id += 1

