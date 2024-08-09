import os
import pandas as pd
import random
import jsonlines

corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/"

# HUPD Template 7: Given the patent applications from {start_year} to {end_year}, generate the titles for patents in {year_not_in_the_range}.
def title_generation(year_not_in_the_range):
    df = pd.read_csv(os.path.join(corpus_dir, "hupd/hupd_{}.csv").format(year_not_in_the_range))
    return df["Title"].tolist()

# NeurIPS Template 5: Are the abstract '{abstract}' and the title '{title}' from the same submission? 
def match(yes_or_no):
    return yes_or_no

# NeurIPS Template 6: Which topic amongst {topic1}, {topic2}, {topic3} fit {title} the best?
def topic_match(topic):
    return topic

# NeurIPS Template 7: Train the model with the first {num} papers and then predict whether the remaining papers will be accepted as oral presentations.
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
question_type_count = {11:100, 12:100, 13:100, 14:100}
question_types = [11,12,13,14]

df_neurips = pd.read_csv(os.path.join(corpus_dir, "neurips/NeurIPS_2023_Papers.csv"))
with jsonlines.open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/hard.jsonl', mode='w') as writer:
    while question_id<=400: 
        question_type = random.choice(question_types)
        if question_id==11:
            # Given the patent applications from {start_year} to {end_year}, generate the titles for patents in {year_not_in_the_range}.
            start_year = random.randint(2004,2012)
            end_year = random.randint(start_year,2012)
            year_not_in_the_range = random.randint(2013,2018)
            question = "Given the patent applications from {} to {}, generate the titles for patents in {}.".format(start_year, end_year, year_not_in_the_range)
            answer = title_generation(year_not_in_the_range)
            # use None to signify not adding to the questions / answers
            if answer:
                writer.write({"qid": "hard-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[11] -= 1
                if question_type_count[11]==0:
                    question_types.remove(11)
                question_id += 1
        elif question_id==12:
            # Are the abstract '{abstract}' and the title '{title}' from the same submission? 
            abstract_id = random.randint(0,3585)
            abstract = df_neurips.at[abstract_id,"Abstract"]
            yes_or_no = random.choice(["Yes", "No"])
            if yes_or_no=="Yes":
                title = df_neurips.at[abstract_id,"Title"]
            else:
                title_id = random.randint(0,3585)
                while title_id==abstract_id:
                    title_id = random.randint(0,3585)
                title = df_neurips.at[title_id,"Title"]
            question = "Are the abstract '{}' and the title '{}' from the same submission? Return 'Yes' or 'No'.".format(abstract,title)
            answer = match(yes_or_no)
            if answer:
                writer.write({"qid": "hard-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[12] -= 1
                if question_type_count[12]==0:
                    question_types.remove(12)
                question_id += 1
        elif question_id==13:
            # Which topic amongst {topic1}, {topic2}, {topic3} fit {title} the best?
            title_id = random.randint(0,3585)
            title = df_neurips.at[title_id, "Title"]
            true_topics = df_neurips.at[title_id, "Topic"].split(" · ")
            false_topics = set(["Deep Learning", "Reinforcement Learning", "Applications", "Theory", "Data-centric AI", "Probabilistic Methods", "Social Aspects", "Optimization"])-set(true_topics)
            three_topics = random.sample(false_topics, 2)+[true_topics[0]]
            random.shuffle(three_topics)
            topic1, topic2, topic3 = three_topics
            question = "Which topic amongst {topic1}, {topic2}, {topic3} fit {title} the best?".format(topic1, topic2, topic3, title)
            answer = topic_match(true_topics[0])
            if answer:
                writer.write({"qid": "hard-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[13] -= 1
                if question_type_count[13]==0:
                    question_types.remove(13)
                question_id += 1
        else:
            # Train the model with the first {} papers and then predict whether the remaining papers will be accepted as oral presentations.
            num = random.choice(list(range(1000,3500+1,100)))
            question_phrasings = ["Train the model with the first {} papers and then predict whether the remaining papers will be accepted as oral presentations.", "Using the first {} papers as the training set, predict whether the remaining papers will be accepted as oral presentations or not.", "Train the model using the initial {} papers, and then predict whether the subsequent papers will be selected for oral presentations."]
            question = question_phrasings[random.randint(0,len(question_phrasings)-1)].format(num)
            answer = predict_oral(num)
            if answer:
                writer.write({"qid": "hard-hupd-{:0>4d}".format(question_id), "question_type":str(question_type), "question":question, "answer":str(answer)})
                question_type_count[14] -= 1
                if question_type_count[14]==0:
                    question_types.remove(14)
                question_id += 1

