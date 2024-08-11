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

# NeurIPS Template 4: Train the model with the first {num} NeurIPS papers and then predict whether the papers with indices {indices} in the index column will be accepted as oral presentations.
def predict_oral(num):
    df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
    df = df.iloc[num+1:]
    def oral_or_not(string):
        if string=="True":
            return "oral"
        else:
            return "not oral"
    df["presentation"] = df["Oral"].apply(oral_or_not)
    return df["presentation"].tolist()

question_id = 1
question_type_count = {7:100, 8:100, 9:100, 10:100}
question_types = [8] #[7,8,9,10]
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
            # Using patent applications from {start_year} to {end_year} for training, predict the decisions of applications from {year_not_in_the_range} with unique indices {indices_str}. Return a list of "ACCEPTED" and "REJECTED".
            start_year = random.randint(2004,2012)
            end_year = random.randint(start_year,2012)
            year_not_in_the_range = random.randint(2013,2018)
            df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv").format(year_not_in_the_range))
            indices_to_choose = df.index[(df['decision']=="ACCEPTED") | (df['decision']=="REJECTED")].to_list()
            indices = random.sample(indices_to_choose,5)
            indices_str = ",".join(["ID-"+str(x) for x in indices])
            decisions = [df.at[index,"decision"] for index in indices]
            question = "Using patent applications from {} to {} for training, predict the decisions of applications from {} with unique indices {}. Return a list of 'ACCEPTED' and 'REJECTED'.".format(start_year, end_year, year_not_in_the_range, indices_str)
            answer = decisions
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[8] -= 1
                if question_type_count[8]==0:
                    question_types.remove(8)
                question_id += 1
        elif question_type==9:
            # Using the rows 1-{num} of NeurIPS papers for training, predict whether the NeurIPS papers with unique indices {indices} belong to {topic}?
            topic = random.choice(["Deep Learning", "Reinforcement Learning", "Health", "Applications", "Theory", "Data-centric AI", "Probabilistic Methods", "Social Aspects", "Optimization"]) 
            row_num = random.choice(list(range(1000, 3000+1, 250)))
            df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
            df = df.iloc[row_num+1:]
            indices_to_choose = df.index[df["Topic"].notna()].to_list()
            indices = random.sample(indices_to_choose,5)
            indices_str = ",".join(["ID-"+str(x) for x in indices])
            topics = []
            for index in indices:
                if topic in df.at[index,"Topic"]:
                    topics.append(topic)
                else:
                    topics.append("not "+topic)
            question_phrasings = ["Using the rows 1-{} of NeurIPS papers for training, predict whether the NeurIPS papers with unique indices {} belong to {}? Return a list of '{}' and 'not {}'.", "Given the rows 1-{} from NeurIPS papers for training, predict whether the NeurIPS papers with unique indices {} fall into the {} category? Return a list of '{}' and 'not {}'.", "Given the {} of NeurIPS papers, predict whether the NeurIPS papers with unique indices {} are anticipated to be classified under the {} topic? Return a list of '{}' and 'not {}'."] 
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(row_num, indices_str, topic, topic, topic)
            answer = topics
            if answer:
                writer.write({"qid": "medium-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[9] -= 1
                if question_type_count[9]==0:
                    question_types.remove(9)
                question_id += 1
        else:
            # Train the model with the first {num} NeurIPS papers and then predict whether the papers with unique indices {indices} will be accepted as oral presentations.
            num = random.choice(list(range(1000,3500+1,100)))
            df = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
            df = df.iloc[num+1:]
            indices_to_choose = df.index[df["Oral"].notna()].to_list()
            indices = random.sample(indices_to_choose,5)
            indices_str = ",".join(["ID-"+str(x) for x in indices])
            orals = []
            for index in indices:
                if df.at[index,"Oral"]:
                    orals.append("Oral")
                else:
                    orals.append("not Oral")
            question_phrasings = ["Train the model with the first {} papers and then predict whether the papers with unique indices {} will be accepted as oral presentations. Return a list of 'Oral' and 'not Oral'.", "Using the first {} papers as the training set, predict whether the papers with unique indices {} will be accepted as oral presentations or not. Return a list of 'Oral' and 'not Oral'.", "Train the model using the initial {} papers, and then predict whether the papers with unique indices {} will be selected for oral presentations. Return a list of 'Oral' and 'not Oral'."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(num, indices)
            answer = orals
            if answer:
                writer.write({"qid": "hard-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[14] -= 1
                if question_type_count[14]==0:
                    question_types.remove(14)
                question_id += 1

