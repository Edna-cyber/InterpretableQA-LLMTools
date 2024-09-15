import re
import os
import math
import types
import random
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Hugging Face datasets
from datasets import Dataset, DatasetDict, concatenate_datasets

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Good old Transformer models
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import PreTrainedTokenizerFast, BatchEncoding

# Tokenizer-releated dependencies
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# Confusion matrix
from sklearn.metrics import confusion_matrix

import time
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from pathlib import Path

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_metric
from datasets import Dataset, DatasetDict

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from sklearn.metrics import f1_score


# Fixing the random seeds
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class LogisticRegression (nn.Module):
    """ Simple logistic regression model """

    def __init__ (self, vocab_size, embed_dim, n_classes, pad_idx):
        super (LogisticRegression, self).__init__ ()
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim, 
            padding_idx = pad_idx)
        # Linear layer
        self.fc = nn.Linear (embed_dim, n_classes)
        
    def forward (self, input_ids):
        # Apply the embedding layer
        embed = self.embedding(input_ids)
        # Apply the linear layer
        output = self.fc (embed)
        # Take the sum of the overeall word embeddings for each sentence
        output = output.sum (dim=1)
        return output


class BasicCNNModel (nn.Module):
    """ Simple 2D-CNN model """
    def __init__(self, vocab_size, embed_dim, n_classes, n_filters, filter_sizes, dropout, pad_idx):
        super(BasicCNNModel, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embed_dim, 
            padding_idx = pad_idx)
        # Conv layer
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1, 
                out_channels = n_filters, 
                kernel_size = (fs, embed_dim)) 
            for fs in filter_sizes])
        # Linear layer
        self.fc = nn.Linear(
            len(filter_sizes) * n_filters, 
            n_classes)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        embed = embed.unsqueeze(1)
        conved = [F.relu(conv(embed)).squeeze(3) for conv in self.convs]    
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        output = self.fc(cat) 
        return output

class table_toolkits():
    def __init__(self):
        self.data = None
        self.prediction = False
        self.path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools" #<YOUR_OWN_PATH>
    
    def db_loader(self, target_db, duration): # e.g. duration can be 2005-2012 or 0-2000, string type, both sides inclusive
        """
        Loads the needed dataframe(s).
        """     
        def extract_start_end(duration):
            if duration=="None":
                return None, None
            hyphen_ind = duration.index("-")
            start = int(duration[:hyphen_ind])
            end = int(duration[hyphen_ind+1:])
            return start, end
        start, end = extract_start_end(duration)
    
        def preprocess_hupd(start_year, end_year):
            def convert_date(series):
                if series.dtype == np.float64:
                    series = series.astype('Int64')
                series = series.astype(str)
                series = pd.to_datetime(series, errors='coerce')
                if series.isna().any():
                    series = pd.to_datetime(series, format="%Y%m%d", errors='coerce')
                return series
            
            if not start_year and not end_year:
                return None
            df = []
            for sub in range(start_year, end_year+1):
                file_path = "{}/data/external_corpus/hupd/hupd_{}.csv".format(self.path, sub)
                df_raw = pd.read_csv(file_path)
                df_raw['patent_number'] = pd.to_numeric(df_raw['patent_number'], errors='coerce').astype('Int64').replace(0, pd.NA) # so that the LLM is aware of which patent numbers are invalid 
                df_raw['examiner_id'] = pd.to_numeric(df_raw['examiner_id'], errors='coerce').astype('Int64') 
                
                df_raw['filing_date'] = convert_date(df_raw['filing_date'])
                df_raw['patent_issue_date'] = convert_date(df_raw['patent_issue_date'])
                df_raw['date_published'] = convert_date(df_raw['date_published'])
                
                df_raw["icpr_category"] = df_raw["main_ipcr_label"].apply(lambda x:x[:3] if isinstance(x, str) else x)
                df_raw["cpc_category"] = df_raw["main_cpc_label"].apply(lambda x:x[:3] if isinstance(x, str) else x)
                df_raw.drop(columns=["main_ipcr_label", "main_cpc_label"], inplace=True)
                # print(df_raw.dtypes)
                # print(df_raw.head())
                df.append(df_raw)
            df = pd.concat(df, ignore_index=True)
            df = df.reset_index(drop=False)
            return df
        
        def preprocess_neurips(start_row, end_row):
            if not start_row and not end_row:
                return None
            file_path = "{}/data/external_corpus/neurips/NeurIPS_2023_Papers.csv".format(self.path)
            df = pd.read_csv(file_path)
            # print(df.dtypes)
            df = df.iloc[start_row:end_row+1]
            df['Authors'] = df['Authors'].str.split(' · ')
            df['Authors_Num'] = df['Authors'].apply(len)
            column_names = ["'"+x+"'" for x in df.columns.tolist()]
            column_names_str = ', '.join(column_names)
            return df   
        
        try:   
            if target_db=="hupd":
                if self.prediction and end>=2013: # uncomment for medium and hard tasks
                    return "Error: The end year of the dataframe cannot be later than year 2012 for prediction tasks."
                df = preprocess_hupd(start, end)
            elif target_db=="neurips":
                if end>3585:
                    return "Error: the dataframe contains 3585 rows in total; the number of rows cannot exceed this limit."
                if self.prediction and end>3000: # uncomment for medium and hard tasks
                    return "Error: The end year of the dataframe cannot exceed row 3000 for prediction tasks."
                df = preprocess_neurips(start, end)
            else:
                return "Error: the only possible choices for target_db are hupd (a patent dataset) and neurips (a papers dataset)."
            
            if isinstance(df, str) and "Error:" in df:
                return df
            
            self.data = df
            length = len(self.data)
            examples_lst = []
            for column in self.data.columns:
                if column=="decision":
                    examples_lst.append("'"+column+"'"+"(e.g.'ACCEPTED', <class 'str'>)")
                else:
                    i = 0
                    example_data = self.data.at[i, column]
                    while i<len(self.data) and not isinstance(example_data, list) and pd.isna(example_data):
                        i += 1
                        example_data = self.data.at[i, column]
                    if isinstance(example_data, str) and len(example_data)>=10:
                        example_data = str(example_data[:10])+"..."
                    examples_lst.append("'"+column+"'"+"(e.g."+str(example_data)+", {})".format(type(example_data)))
            examples_lst_str = ', '.join(examples_lst)
            return "We have successfully loaded the {} dataframe, including the following columns: {}.".format(target_db, examples_lst_str)+"\nIt has {} rows.".format(length)
        except Exception as e:
            return "Error: "+str(e)
        
    def pandas_interpreter(self, pandas_code): 
        """
        Executes the provided Pandas code.
        """
        if self.data is not None:
            global_var = {"df": self.data.copy()}
        else:
            return "Error: Dataframe does not exist. Make sure the dataframe is loaded with LoadDB first."
        try: 
            exec(pandas_code, globals(), global_var)
            variable_values = {}
            excluded_types = (types.ModuleType, types.FunctionType, type, pd.DataFrame)
            for var_name, var_value in locals().items(): 
                if var_name in ["self", "pandas_code","variable_values"]:
                    continue
                elif var_name=="global_var" and isinstance(var_value, dict):
                    for global_var_name, global_var_value in var_value.items(): 
                        if global_var_name not in ["__builtins__", "quit", "copyright", "credit", "license", "help"] and not isinstance(global_var_value, excluded_types):
                            if global_var_value is None:
                                continue
                            elif isinstance(global_var_value, pd.Series):
                                variable_values[global_var_name] = global_var_value.head().to_dict()
                            elif isinstance(global_var_value, (list, np.ndarray)):
                                variable_values[global_var_name] = global_var_value[:10]
                            elif isinstance(global_var_value, dict):
                                variable_values[global_var_name] = dict(list(global_var_value.items())[:10])
                            else:
                                variable_values[global_var_name] = global_var_value
                elif not var_name.startswith('__') and not isinstance(var_value, excluded_types) and var_name!='excluded_types':
                    if var_value is None:
                        continue
                    if isinstance(var_value, pd.Series):
                        variable_values[var_name] = var_value.head().to_dict()
                    elif isinstance(var_value, (list, dict, np.ndarray)):
                        variable_values[var_name] = var_value[:10]
                    else:
                        variable_values[var_name] = var_value
            return variable_values
        except KeyError as e:
            column_names = ["'"+x+"'" for x in global_var["df"].columns.tolist()]
            column_names_str = ', '.join(column_names)
            return "Error: "+str(e)+" column does not exist.\nThe dataframe contains the following columns: "+column_names_str+". It has the following structure: {}".format(self.data.head())
        except NameError as e:
            if "'pd'" in str(e):
                return "Error: "+str(e)+"\nImport the pandas library using the pandas_interpreter."
            else:
                return "Error: "+str(e)
        except Exception as e:
            return "Error: "+str(e)
            
    def textual_classifier(self, database, model_name, section, text, target, one_v_all): 
        """
        Runs a classificaiton prediction task given a textual input.
        """
        try:
            if database=="hupd":
                hupd_features = ["patent_number", "decision", "title", "abstract", "claims",
                                "background", "summary", "full_description", "main_cpc_label", "main_ipcr_label", 
                                "filing_date", "patent_issue_date", "date_published","examiner_id"]
                if target not in hupd_features:
                    return "Error: {} column does not exist.\nPlease select target from the following features: {}".format(target, hupd_features)
                if section not in hupd_features:
                    return "Error: {} column does not exist.\nPlease select section from the following features: {}".format(section, hupd_features)
            if database=="neurips":
                neurips_features = ["Title","Authors","Location","Abstract","Topic","Oral","Poster Session","Subtopic"]
                if target not in neurips_features:
                    return "Error: {} column does not exist.\nPlease select target from the following features: {}".format(target, neurips_features)
                if section not in neurips_features:
                    return "Error: {} column does not exist.\nPlease select section from the following features: {}".format(section, neurips_features)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            unique_classes = ["not "+one_v_all, one_v_all]
            CLASSES = 2
            CLASS_NAMES = [0,1]
            
            vocab_size = 10000
            batch_size=64
            test_every=500
            n_filters=25
            filter_sizes=[[3,4,5], [5,6,7], [7,9,11]]
            dropout=0.25
            epoch_n=5
            lr=2e-5
            eps=1e-8
            embed_dim=200
            max_length=512 #256
            alpha_smooth_val = 1.0

            # Create model and tokenizer
            def create_model_and_tokenizer(model_name=model_name, vocab_size=10000, embed_dim=200, n_classes=CLASSES, max_length=512): #'bert-base-uncased'
                # Finetune
                if model_name == 'bert-base-uncased':
                    config = AutoConfig.from_pretrained(model_name, num_labels=CLASSES, output_hidden_states=False)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    tokenizer.max_length = max_length
                    tokenizer.model_max_length = max_length
                    model = AutoModelForSequenceClassification.from_config(config=config)
                elif model_name in ['cnn', 'logistic_regression']:
                    if database=="hupd":
                        tokenizer = PreTrainedTokenizerFast(tokenizer_file="/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/temp/hupd_tokenizer.json") # <YOUR_OWN_PATH>
                    if database=="neurips":
                        tokenizer = PreTrainedTokenizerFast(tokenizer_file="/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/temp/neurips_tokenizer.json") # <YOUR_OWN_PATH>
                    pad_idx = tokenizer.encode('[PAD]')[0]

                    tokenizer.model_max_length = max_length
                    tokenizer.max_length = max_length
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    tokenizer.pad_token = '[PAD]'
                    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
                    tokenizer.sep_token = '[SEP]'

                    model = None
                    if model_name == 'logistic_regression':
                        model = LogisticRegression(vocab_size=vocab_size, embed_dim=embed_dim, n_classes=CLASSES, pad_idx=pad_idx)
                    elif model_name == 'cnn':
                        model = BasicCNNModel(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx, n_classes=CLASSES, n_filters=n_filters, filter_sizes=filter_sizes[0], dropout=dropout)
                else:
                    raise NotImplementedError()
                        
                print(model)
                return tokenizer, model, vocab_size

            # Map target2string
            def map_target_to_label(example):
                if not isinstance(example[target], str):
                    return {'output': int(one_v_all==str(int(example[target])))}
                return {'output': int(one_v_all==example[target])}
            
            def create_data(tokenizer, text=text):
                zero_encoding = tokenizer('', truncation=True, padding='max_length', max_length=max_length)
                if text is None:
                    encoded_text = zero_encoding
                else:
                    encoded_text = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
                input_ids = torch.tensor(encoded_text['input_ids']).unsqueeze(0)
                attention_mask = torch.tensor(encoded_text['attention_mask']).unsqueeze(0)
                return {'input_ids': input_ids, 'attention_mask': attention_mask}

            # Convert ids2string
            def convert_ids_to_string(tokenizer, input):
                return ' '.join(tokenizer.convert_ids_to_tokens(input)) # tokenizer.decode(input)

            # Evaluation procedure (for the neural models)
            def test(processed_text, model, criterion, device): 
                try:
                    with open('/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/temp/{}_{}_{}_{}_model.pkl'.format(database, model_name, section, target), 'rb') as file:
                        model = pickle.load(file)
                except:
                    return "Error: no pretrained model is available for the specified task. Please use other tools to solve the problem."
                model.eval()
                
                inputs = processed_text['input_ids']
                inputs = inputs.to(device)
                with torch.no_grad():
                    if model_name in ['cnn', 'logistic_regression']:
                        outputs = model(input_ids=inputs)
                    else:
                        outputs = model(input_ids=inputs).logits
                    prediction = torch.argmax(outputs, dim=1)[0].item()
                return prediction
                
            # Create a model and an appropriate tokenizer
            tokenizer, model, vocab_size = create_model_and_tokenizer(
                model_name = model_name, 
                vocab_size = vocab_size,
                embed_dim = embed_dim,
                n_classes = CLASSES,
                max_length=max_length
                )

            # GPU specifications 
            model.to(device)

            processed_text = create_data(
                tokenizer = tokenizer, 
                text = text
            )
                
            # Optimizer
            if model_name in ['logistic_regression', 'cnn']:
                optim = torch.optim.Adam(params=model.parameters())
            else:
                optim = torch.optim.AdamW(params=model.parameters(), lr=lr, eps=eps)
            # Loss function 
            criterion = torch.nn.CrossEntropyLoss() 
            # Test
            prediction = test(processed_text, model, criterion, device)
            if isinstance(prediction, str) and prediction.startswith("Error:"):
                return prediction
            return {"prediction": unique_classes[prediction]} 
        except Exception as e:
            return "Error: "+str(e)

if __name__ == "__main__":
    db = table_toolkits()
    # db.db_loader('hupd', '2016-2016')
#     pandas_code = """
# import pandas as pd
# df['filing_month'] = df['filing_date'].apply(lambda x: x.month)
# month = df['filing_month'].mode()[0]
# """
    # print(db.pandas_interpreter(pandas_code))

    # print(db.textual_classifier('hupd', 'logistic_regression', 'abstract', abstract, 'decision', 'ACCEPTED'))
    # print(db.textual_classifier('hupd', 'cnn', 'abstract', abstract, 'decision', 'ACCEPTED'))
    # print(db.textual_classifier('hupd', 'bert-base-uncased', 'abstract', abstract, 'decision', 'ACCEPTED'))
    # print(db.textual_classifier('neurips', 'logistic_regression', 'Abstract', abstract, 'Oral', 'oral'))
    # print(db.textual_classifier('neurips', 'cnn', 'Abstract', abstract, 'Oral', 'oral'))
    # print(db.textual_classifier('neurips', 'bert-base-uncased', 'Abstract', abstract, 'Oral', 'oral'))
    
    db.db_loader('hupd', '2007-2015')
    code = """
print(df[df['title']=='FUEL CARTRIDGE AND FUEL CELL USING THE SAME']["filing_date"])
print(df[df['title']=='FUEL CARTRIDGE AND FUEL CELL USING THE SAME']["date_published"])
df['days_elapsed'] = (df['date_published'] - df['filing_date']).dt.days
print(df[df['title']=='FUEL CARTRIDGE AND FUEL CELL USING THE SAME']["days_elapsed"])
max_days_elapsed = df.loc[df['days_elapsed'].idxmax()]['title']
"""
    print(db.pandas_interpreter(code))


    

    