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
            with open(file, 'r') as f:
                data = json.load(f)
            df = pd.json_normalize(data,meta=lst_of_columns,errors='ignore')
            df = df[lst_of_columns]
            df_year.append(df)
    df_year = pd.concat(df_year, ignore_index=True)
    return df_year

dataset_name = "hupd" 
data_dir = "/home/users/rz95/"
corpus_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus"

df_2015 = to_dataframe(os.path.join(data_dir, "2015"))
df_2016 = to_dataframe(os.path.join(data_dir, "2016"))
df_2017 = to_dataframe(os.path.join(data_dir, "2017"))

def preprocess(df): 
    df['patent_number'] = df['patent_number'].fillna(0)
    df['patent_number'] = df['patent_number'].replace({'None':0, 'nan':0})
    df['patent_number'] = df['patent_number'].astype(int)
    df['examiner_id'] = df['examiner_id'].fillna(0)
    df['examiner_id'] = df['examiner_id'].replace({'':0})
    df['examiner_id'] = df['examiner_id'].astype(float).astype(int)
    
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['patent_issue_date'] = pd.to_datetime(df['patent_issue_date'])
    df['date_published'] = pd.to_datetime(df['date_published'])
    
    return df

df_2015 = preprocess(df_2015)
df_2016 = preprocess(df_2016)
df_2017 = preprocess(df_2017)

# print(df_2015.dtypes)

df_2015.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2015.csv"), index=False) 
df_2016.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2016.csv"), index=False) 
df_2017.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2017.csv"), index=False)
