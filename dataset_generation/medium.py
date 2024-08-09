import os
import pandas as pd
import random
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"

# HUPD Template 5: Based on the patent applications per month from {start_year} to 2012, estimate the percentage of patents filed in the first {n} months of 2013 that will be accepted. Return a list of percentages, with each value between 0 and 100.
def forecast(n):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_2013.csv"))
    print(1) ###
    df["filing_date"] = pd.to_datetime(df['filing_date'])
    print(2)
    df["month"] = df["filing_date"].dt.month
    print(3)
    df["acceptance"] = df["decision"].apply(lambda x: 1 if x =="ACCEPTED" else 0)
    print(4)
    first_few_months = df.groupby("month")['acceptance'].mean().sort_index()*100 ## maybe need a way to fill in when a month of data is missing
    print("first few months", first_few_months) ###
    del df
    first_n = first_few_months.head(n).tolist()
    return first_n

# HUPD Template 6: Using patent applications from {start_year} to {end_year} for training, predict the decisions of applications from {year_not_in_the_range}. Return a list of "ACCEPTED" and "REJECTED".
def predict_decision(year_not_in_the_range):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv").format(year_not_in_the_range))
    filtered_df = df[(df['decision']=="ACCEPTED") | (df['decision']=="REJECTED")]
    return filtered_df['decision'].tolist()

# NeurIPS Template 3: Using the rows 1-{num} of NeurIPS papers for training, predict whether the rows {num+1}-3585 of NeurIPS papers belong to {topic}?
def predict_topic(row_num, topic):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv")) 
    df = df.iloc[row_num+1:]
    def contains_topic(string):
        if topic in string:
            return topic
        else:
            return "not "+topic
    df["Topic"] = df["Topic"].apply(contains_topic)    
    return df["Topic"].tolist()

# NeurIPS Template 4: Using the first {num-1} papers for training, determine a threshold for the number of authors among the {num}-3585 NeurIPS papers. Papers with more authors than this threshold should be more often oral presentations compared to papers with fewer authors.
def threshold(num):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv")) 
    df = df.iloc[num:]
    df["num_authors"] = df["Authors"].str.split(" · ").apply(len)
    df["oral_bool"] = df["Oral"].apply(lambda x: 1 if x =="TRUE" else 0)
    number_of_authors = df["num_authors"].unique()
    thresholds = []
    for threshold in range(min(number_of_authors), max(number_of_authors)+1):
        more_than_threshold = df[df["num_authors"]>threshold]["oral_bool"].mean()
        less_than_threshold = df[df["num_authors"]<=threshold]["oral_bool"].mean()
        if more_than_threshold>less_than_threshold:
            thresholds.append(threshold)
    return thresholds

question_id = 1
question_type_count = {7:100, 8:100, 9:100, 10:100}
question_types = [7,8,9,10]
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/medium.jsonl', mode='w') as writer:
    while question_id<=1: # 400
        question_type = random.choice(question_types) 
        if question_id==7:
            # Based on the patent applications per month from {} to 2012, estimate the percentage of patents filed in the first {} months of 2013 that will be accepted. Return a list of percentages, with each value between 0 and 100.
            start_year = random.randint(2004,2012)
            n = random.randint(1,12)
            question_phrasings = ["Based on the patent applications per month from {} to 2012, estimate the percentage of patents filed in the first {} months of 2013 that will be accepted. Return a list of percentages, with each value between 0 and 100.", "Given the patent applications per month from {} to 2012, predict the acceptance percentage for patents filed in the first {} months of 2013. Return a list of percentages ranging from 0 to 100."] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(start_year,n)
            answer = forecast(n)
            # use None to signify not adding to the questions / answers
            if answer:
                writer.write({"qid": "medium-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[7] -= 1
                if question_type_count[7]==0:
                    question_types.remove(7)
                question_id += 1
        elif question_id==8:
            # Using patent applications from {start_year} to {end_year} for training, predict the decisions of applications from {year_not_in_the_range}. Return a list of "ACCEPTED" and "REJECTED".
            start_year = random.randint(2004,2012)
            end_year = random.randint(start_year,2012)
            year_not_in_the_range = random.randint(2013,2018)
            question = "Using patent applications from {} to {} for training, predict the decisions of applications from {}. Return a list of 'ACCEPTED' and 'REJECTED'.".format(start_year, end_year, year_not_in_the_range)
            answer = predict_decision(year_not_in_the_range)
            if answer:
                writer.write({"qid": "medium-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[8] -= 1
                if question_type_count[8]==0:
                    question_types.remove(8)
                question_id += 1
        elif question_id==9:
            # Using the rows 1-{num} of NeurIPS papers for training, predict whether the rows {num+1}-3585 of NeurIPS papers belong to {topic}? Return a list of '{topic}' and 'not {topic}'.
            topic = random.choice(["Deep Learning", "Reinforcement Learning", "Applications", "Theory", "Data-centric AI", "Probabilistic Methods", "Social Aspects", "Optimization"]) 
            row_num = random.choice(list(range(1000, 3000+1, 250)))
            question_phrasings = ["Using the rows 1-{} of NeurIPS papers for training, predict whether the rows {}-3585 of NeurIPS papers belong to {}? Return a list of '{}' and 'not {}'.", "Given the rows 1-{} from NeurIPS papers for training, predict whether the NeurIPS papers in rows {}-3585 fall into the {} category? Return a list of '{}' and 'not {}'.", "Given the {} of NeurIPS papers, predict whether the NeurIPS papers in rows {} through 3585 are anticipated to be classified under the {} topic? Return a list of '{}' and 'not {}'."] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(row_num, row_num+1, topic, topic, topic)
            answer = predict_topic(row_num, topic)
            if answer:
                writer.write({"qid": "medium-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[9] -= 1
                if question_type_count[9]==0:
                    question_types.remove(9)
                question_id += 1
        else:
            # Using the first {num-1} papers for training, determine a threshold for the number of authors among the {num}-3585 NeurIPS papers. Papers with more authors than this threshold should be more often oral presentations compared to papers with fewer authors.
            num = random.choice(list(range(100,3500+1,100)))
            question_phrasings = ["Using the first {} papers for training, determine a threshold for the number of authors among the {}-3585 NeurIPS papers. Papers with more authors than this threshold should be more often oral presentations compared to papers with fewer authors.", "Using the first {} papers for training, identify a threshold number of authors for the {}-3585 NeurIPS papers. Papers with more authors than this threshold are more often oral presentations compared to those with fewer authors.", "Using the first {} papers for training, find the number of authors threshold for the {}-3585 NeurIPS papers, where papers exceeding this threshold are more often oral presentations than those with fewer authors."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(num-1,num)
            answer = threshold(num)
            if answer:
                writer.write({"qid": "medium-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[10] -= 1
                if question_type_count[10]==0:
                    question_types.remove(10)
                question_id += 1

