import os
import json
import pandas as pd

lst_of_columns = ['patent_number',
        'decision',
        'title',
        'abstract',
        'claims',
        'background',
        'summary',
        'full_description',
        'main_cpc_label',
        'main_ipcr_label',
        'filing_date',
        'patent_issue_date',
        'date_published',
        'examiner_id']

def to_dataframe(directory):
    df_year = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file = os.path.join(directory, filename)
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print(f"Error decoding JSON in file: {file}. Error: {e}")
                continue
            df = pd.json_normalize(data,meta=lst_of_columns,errors='ignore')
            df = df[lst_of_columns]
            df_year.append(df)
    df_year = pd.concat(df_year, ignore_index=True)
    return df_year

dataset_name = "hupd" 
data_dir = "/home/users/rz95/"
corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus"

df_2014 = to_dataframe(os.path.join(data_dir, "2014"))
# df_2015 = to_dataframe(os.path.join(data_dir, "2015"))
# df_2016 = to_dataframe(os.path.join(data_dir, "2016"))
# df_2017 = to_dataframe(os.path.join(data_dir, "2017"))
# df_2018 = to_dataframe(os.path.join(data_dir, "2018"))

df_2014.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2014.csv"), index=False) 
# df_2015.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2015.csv"), index=False) 
# df_2016.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2016.csv"), index=False) 
# df_2017.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2017.csv"), index=False)
# df_2018.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2018.csv"), index=False)
