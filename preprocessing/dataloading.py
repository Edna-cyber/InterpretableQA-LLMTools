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

# df_2004 = to_dataframe(os.path.join(data_dir, "2004"))
# df_2005 = to_dataframe(os.path.join(data_dir, "2005"))
# df_2006 = to_dataframe(os.path.join(data_dir, "2006"))
# df_2007 = to_dataframe(os.path.join(data_dir, "2007"))
# df_2008 = to_dataframe(os.path.join(data_dir, "2008"))
# df_2009 = to_dataframe(os.path.join(data_dir, "2009"))
# df_2010 = to_dataframe(os.path.join(data_dir, "2010"))
# df_2011 = to_dataframe(os.path.join(data_dir, "2011"))
# df_2012 = to_dataframe(os.path.join(data_dir, "2012"))
# df_2013 = to_dataframe(os.path.join(data_dir, "2013"))
# df_2014 = to_dataframe(os.path.join(data_dir, "2014"))
# df_2015 = to_dataframe(os.path.join(data_dir, "2015"))
# df_2016 = to_dataframe(os.path.join(data_dir, "2016"))
# df_2017 = to_dataframe(os.path.join(data_dir, "2017"))
# df_2018 = to_dataframe(os.path.join(data_dir, "2018"))

# df_2004.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2004.csv"), index=False)
# df_2005.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2005.csv"), index=False)
# df_2006.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2006.csv"), index=False)
# df_2007.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2007.csv"), index=False)
# df_2008.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2008.csv"), index=False)
# df_2009.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2009.csv"), index=False)
# df_2010.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2010.csv"), index=False)
# df_2011.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2011.csv"), index=False)
# df_2012.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2012.csv"), index=False)
# df_2013.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2013.csv"), index=False) 
# df_2014.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2014.csv"), index=False) 
# df_2015.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2015.csv"), index=False) 
# df_2016.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2016.csv"), index=False) 
# df_2017.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2017.csv"), index=False)
# df_2018.to_csv(os.path.join(corpus_dir, dataset_name, "hupd_2018.csv"), index=False)
