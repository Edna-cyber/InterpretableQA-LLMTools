import os
import pandas as pd
import random
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"

# HUPD Template 5: Using the patent applications from {start_year} to {end_year}, forecast the percentage of accepted patents in {end_year+1}.
def forecast(year):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv".format(str(year))))  
    total_len = len(df)
    accepted_len = len(df[df["decision"]=="ACCEPTED"])
    del df
    return accepted_len / total_len

# HUPD Template 6: Using the {section} of patent applications from {start_year} to {end_year} for training, what proportion of applications from {year_not_in_the_range} are predicted to be accepted if they fall into the {IPCR/CPC} category of {A-H}?
def predict_decision(year_not_in_the_range, label):
    total_len = len(df)
    accepted_cat_df_len = len(df[df["decision"]=="ACCEPTED"] and df["cpc_category"]==label) 
    del df
    return accepted_cat_df_len / total_len

# NeurIPS Template 3: Using the {section} of NeurIPS 2023 papers, what proportion of papers are predicted to belong to {topic}?
def predict_topic(topic):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv")) 
    total_len = len(df)
    topic_df_len = len(df[df["Topic"]==topic])
    return topic_df_len / total_len

# NeurIPS Template 4: Are papers with more than {num} authors more likely to be selected for oral presentations rather than poster presentations? 
def oral_vs_poster(num):
    pass

# NeurIPS Template 5: Are the abstract '{abstract}' and the title '{title}' from the same submission? 
def match(abstract, title):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
    return bool(len(df[df["Abstract"]==abstract and df["Title"]==title]))

question_id = 1
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/medium/hupd-medium.jsonl', mode='w') as writer:
    while question_id<=500: 
        question_type = random.randint(7,11) 
        if question_id==7:
            # Using the patent applications from {start_year} to {end_year}, forecast the percentage of accepted patents in {end_year+1}.
            start_year = random.randint(2014,2016)
            end_year = random.randint(start_year,2016)
            question_phrasings = ["Using the patent applications from {} to {}, forecast the percentage of accepted patents in {}.", "Based on the patent applications from {} to {}, estimate the percentage of patents that will be accepted in {}.", "Given the patent applications from {} to {}, predict the proportion of patents that will be accepted in {}."] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year,end_year,end_year+1)
            answer = forecast(end_year+1)
        elif question_id==8:
            # Using the {section} of patent applications from {start_year} to {end_year} for training, what proportion of applications from {year_not_in_the_range} are predicted to be accepted if they fall into the CPC category of {A-H}?
            section = random.choice(["abstracts", "backgrounds", "summaries", "full descriptions"]) 
            start_year = random.randint(2014,2016)
            end_year = random.randint(start_year,2016)
            year_not_in_the_range = random.choice(list(set([i for i in range(2014,2018)])-set([i for i in range(start_year, end_year+1)])))
            label = random.choice() ### in the format of e.g. A01 Technology
            question = "Using the {} of patent applications from {} to {} for training, what proportion of applications from {} are predicted to be accepted if they fall into the CPC category of {}?".format(section, start_year, end_year, year_not_in_the_range, label)
            answer = predict_decision(year_not_in_the_range, label)
        elif question_id==9:
            # Using the {section} of NeurIPS 2023 papers, what proportion of these papers are predicted to belong to the {topic} topic? 
            section = random.choice(["abstracts", "titles"]) 
            topic = random.choice() ### need all
            question_phrasings = ["Using the {} of NeurIPS 2023 papers, what proportion of these papers are predicted to belong to the {} topic?", "Based on the {} from NeurIPS 2023 papers, what proportion of these papers are predicted to fall into the {} category?", "Given the {} of NeurIPS 2023 papers, what fraction of these papers is anticipated to be classified under the {} topic?"] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(section, topic)
            answer = predict_topic(topic)
        elif question_id==10:
            # Are papers with more than {num} authors more likely to be selected for oral presentations rather than poster presentations? 
            num = random.randint(1,20)
            order = random.randint(0,1)
            if order==0:
                question_phrasings = ["Are papers with more than {} authors more likely to be selected for oral presentations rather than poster presentations?", "Do papers with more than {} authors have a higher chance of being chosen for oral presentations instead of poster presentations?", "Do papers with more than {} authors tend to be selected for oral presentations rather than poster presentations?"]
            else:
                question_phrasings = ["Are papers with more than {} authors more likely to be selected for poster presentations rather than oral presentations?", "Do papers with more than {} authors have a higher chance of being chosen for poster presentations instead of oral presentations?", "Do papers with more than {} authors tend to be selected for poster presentations rather than oral presentations?"]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(num)
            answer = oral_vs_poster(num)
        else:
            # Are the abstract '{abstract}' and the title '{title}' from the same submission? 
            abstract = random.choice ###
            title = random.choice ###
            question = "Are the abstract '{}' and the title '{}' from the same submission?".format(abstract,title)
            answer = match(abstract,title)
        # use None to signify not adding to the questions / answers
        if answer:
            writer.write({"qid": "medium-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
            question_id += 1

