import os
import pandas as pd
import random
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"

# HUPD Template 7: Generate the abstract for the patent application with patent number {No.}.
def abstract_generation(patent_number):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
    return df[df["patent_number"]==patent_number]["abstract"]

# NeurIPS Template 6: Are the abstract '{abstract}' and the title '{title}' from the same submission? 
def match(abstract, title):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
    return bool(len(df[df["Abstract"]==abstract and df["Title"]==title]))

# NeurIPS Template 7: Does this title '{title}' fall into the topic '{topic}'?
def topic_match(title, topic):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
    return bool(len(df[df["Title"]==title and df["Topic"]==topic]))

question_id = 1
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/hard.jsonl', mode='w') as writer:
    while question_id<=300: 
        question_type = random.randint(12,14) 
        if question_id==12:
            # Generate the abstract for the patent application with patent number {No.}.
            patent_number = random.randint ###
            question = "Generate the abstract for the patent application with patent number {}.".format(patent_number)
            answer = abstract_generation(patent_number)
        elif question_id==13:
            # Are the abstract '{abstract}' and the title '{title}' from the same submission? 
            abstract = random.choice ###
            title = random.choice ###
            question = "Are the abstract '{}' and the title '{}' from the same submission?".format(abstract,title)
            answer = match(abstract,title)
        else:
            # Does this title '{title}' fall into the topic '{topic}'?
            title = random.choice ###
            topic = random.choice ###
            question = "Does this title '{title}' fall into the topic '{topic}'?".format(title, topic)
            answer = topic_match(title, topic)
        # use None to signify not adding to the questions / answers
        if answer:
            writer.write({"qid": "medium-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
            question_id += 1

### need to rethink template 7 question
### Need to make sure for question_id 13, it's a held out set