import os
import pandas as pd
import json
import random
import tqdm
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/hupd/"

# What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
def average_pendency(start_year, end_year):
    df = []
    for year in range(start_year, end_year+1):
        df.append(pd.read_csv(os.path.join(corpus_dir, "hupd_{}.csv".format(str(year)))))
    df = pd.concat(df, ignore_index=True) 
    
    # Filter and convert dates in one step
    df = df[df["decision"] == 1].copy()
    df["filing_date"] = pd.to_datetime(df['filing_date'])
    df["patent_issue_date"] = pd.to_datetime(df['patent_issue_date'])

    # Calculate pendencies directly as days
    pendencies = (df["patent_issue_date"] - df["filing_date"]).dt.days

    # Compute the mean pendency
    mean_pendency = pendencies.mean()
    return mean_pendency

questions = []
question_id = 0
while question_id<=2: #100
    question_type = random.randint(0, 0) # (0, 4)
    if question_type == 0:
        # What was the average time between the filing and issuance of patents from {start_year} to {end_year}?
        start_year = random.randint(2015,2017)
        end_year = random.randint(start_year,2017)
        question = "What was the average time between the filing and issuance of patents from {} to {}?".format(start_year, end_year)
        answer = average_pendency(start_year, end_year)
    # use None to signify not adding to the questions / answers
    if answer:
        questions.append({"qid": "easy-hupd-{:0>4d}".format(question_id), "question":question, "answer":answer})
        question_id += 1

with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/easy/hupd-easy.jsonl', mode='w') as writer:
    for row in questions:
        writer.write(row)