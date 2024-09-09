import os
import pandas as pd
import random
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"

# HUPD Template 5: Based on the patent applications per month from {start_year} to 2012, estimate the percentage of patents filed in the first {n} months of 2013 that will be accepted. Return a list of percentages, with each value between 0 and 100.
def forecast(n):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_2013.csv"))
    df["filing_date"] = pd.to_datetime(df['filing_date'])
    df["month"] = df["filing_date"].dt.month
    df["acceptance"] = df["decision"].apply(lambda x: 1 if x =="ACCEPTED" else 0)
    first_few_months = df.groupby("month")['acceptance'].mean().sort_index()*100 ## maybe need a way to fill in when a month of data is missing
    del df
    first_n = first_few_months.head(n).tolist()
    return first_n

question_id = 1
question_type_count = {7:100, 8:100, 9:100, 10:100}
question_types = [7,8,9,10]
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/medium.jsonl', mode='w') as writer:
    while question_id<=10: # 400
        question_type = random.choice(question_types) 
        if question_type==7:
            # Based on the patent applications per month from {} to 2012, estimate the percentage of patents filed in the first {} months of 2013 that will be accepted. Return a list of percentages, with each value between 0 and 100.
            start_year = random.randint(2004,2012)
            n = random.randint(1,12)
            question_phrasings = ["Based on the patent applications per month from {} to 2012, estimate the percentage of patents filed in the first {} months of 2013 that will be accepted. Return a list of percentages, with each value between 0 and 100.", "Given the patent applications per month from {} to 2012, predict the acceptance percentage for patents filed in the first {} months of 2013. Return a list of percentages ranging from 0 to 100."] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year,n)
            answer = forecast(n)
            # use None to signify not adding to the questions / answers
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[7] -= 1
                if question_type_count[7]==0:
                    question_types.remove(7)
                question_id += 1
        elif question_type==8:
            # For a patent application with an abstract {abstract_content}, predict whether it will get accepted. Return either 'ACCEPTED' or 'REJECTED'.
            year_not_in_the_range = random.randint(2013,2018)
            df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv").format(year_not_in_the_range))
            indices_to_choose = df.index[(df['decision']=="ACCEPTED") | (df['decision']=="REJECTED")].to_list()
            index = random.sample(indices_to_choose,1)
            abstract_content = df.at[index,"abstract"]
            decision = df.at[index,"decision"]
            question = "For a patent application with an abstract {}, predict whether it will get accepted. Return either 'ACCEPTED' or 'REJECTED'.".format(abstract_content)
            answer = decision
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[8] -= 1
                if question_type_count[8]==0:
                    question_types.remove(8)
                question_id += 1
        elif question_type==9:
            # For a NeurIPS 2023 paper with title {title_content}, predict whether it belongs to {topic}? Return either '{topic}' or 'not {topic}'.
            topic = random.choice(["Deep Learning", "Reinforcement Learning", "Health", "Applications", "Theory", "Data-centric AI", "Probabilistic Methods", "Social Aspects", "Optimization"]) 
            df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
            df = df.iloc[3001:]
            indices_to_choose = df.index[df["Topic"].notna()].to_list()
            index = random.sample(indices_to_choose,1)
            title_content = df.at[index,"Title"]
            if topic in df.at[index,"Topic"]:
                belong = topic
            else:
                belong = "not "+topic
            question = "For a NeurIPS 2023 paper with title {}, predict whether it belongs to {}? Return either '{}' or 'not {}'.".format(title_content, topic, topic, topic)
            answer = belong
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[9] -= 1
                if question_type_count[9]==0:
                    question_types.remove(9)
                question_id += 1
        else:
            # For a NeurIPS 2023 paper with abstract {abstract_content}, predict whether it will be accepted as an oral presentation? Return either ‘oral’ or ‘not oral’.
            df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
            df = df.iloc[3001:]
            indices_to_choose = df.index[df["Oral"].notna()].to_list()
            index = random.sample(indices_to_choose,1)
            abstract_content = df.at[index,"Abstract"]
            if df.at[index,"Oral"]:
                oral = "oral"
            else:
                oral = "not oral"
            question = "For a NeurIPS 2023 paper with abstract {}, predict whether it will be accepted as an oral presentation? Return either ‘oral’ or ‘not oral’.".format(abstract_content)
            answer = oral
            if answer:
                writer.write({"qid": "hard-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[10] -= 1
                if question_type_count[10]==0:
                    question_types.remove(10)
                question_id += 1

