import re
import os
import math
import types
import random
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

# Import the sklearn Multinomial Naive Bayes
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

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
        # embed = [batch size, sent len, emb dim]
        embed = embed.unsqueeze(1)
        # embed = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embed)).squeeze(3) for conv in self.convs]    
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        output = self.fc(cat) #.sigmoid ().squeeze()
        return output

class table_toolkits():
    def __init__(self):
        self.data = None
        self.dataset_dict = None
        self.train_groundtruth = None # pandas series
        self.test_groundtruth = None # pandas series
        self.path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools" #<YOUR_OWN_PATH>

    def db_loader(self, target_db, train_duration="None", test_duration="None", outcome_col="None"): # e.g. duration can be 2005-2012 or 0-2000, string type, both sides inclusive
        """
        Loads the needed dataframe(s).
        """     
        if test_duration=="None" and outcome_col!="None":
            return "Error: outcome_col needs to be set to string None when test_duration is None."
        
        def extract_start_end(duration):
            if duration=="None":
                return None, None
            hyphen_ind = duration.index("-")
            start = int(duration[:hyphen_ind])
            end = int(duration[hyphen_ind+1:])
            return start, end
        train_start, train_end = extract_start_end(train_duration)
        test_start, test_end = extract_start_end(test_duration)
    
        def preprocess_hupd(start_year, end_year):
            def convert_date(series):
                if series.dtype==np.float64:
                    series = series.astype('Int64')
                series = series.astype(str)
                if series.str.contains("-").any(): 
                    series = pd.to_datetime(series)
                else:
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
                
                column_names = ["'"+x+"'" for x in df_raw.columns.tolist()]
                column_names_str = ', '.join(column_names)
                if outcome_col not in df_raw.columns and outcome_col!="None":
                    return "Error: The outcome_col does not exist in the dataframe. Choose from the following columns: {}.".format(column_names_str)
                # print(df_raw.dtypes)
                # print(df_raw.head())
                df.append(df_raw)
            df = pd.concat(df, ignore_index=True)
            df['Unique_ID'] = df.index.map(lambda x:"ID-"+str(x))
            df = df.reset_index(drop=False)
            # remove rows where the predicted target is NA
            if outcome_col!="None":
                df.dropna(subset=[outcome_col], inplace=True)
            if outcome_col=="decision": # only use "ACCEPTED" and "REJECTED" column
                df = df[(df[outcome_col]=="ACCEPTED") | (df[outcome_col]=="REJECTED")]
            return df
        
        def preprocess_neurips(start_row, end_row):
            if not start_row and not end_row:
                return None
            file_path = "{}/data/external_corpus/neurips/NeurIPS_2023_Papers.csv".format(self.path)
            df = pd.read_csv(file_path)
            # print(df.dtypes)
            df['Unique_ID'] = df.index.map(lambda x:"ID-"+str(x))
            df = df.reset_index(drop=False)
            df = df.iloc[start_row:end_row+1]
            df['Authors'] = df['Authors'].str.split(' · ')
            df['Authors_Num'] = df['Authors'].apply(len)
            column_names = ["'"+x+"'" for x in df.columns.tolist()]
            column_names_str = ', '.join(column_names)
            if outcome_col not in df.columns and outcome_col!="None":
                return "Error: The outcome_col does not exist in the dataframe. Choose from the following columns: {}.".format(column_names_str)
            # remove rows where the predicted target is NA
            if outcome_col!="None":
                df = df.dropna(subset=[outcome_col])
            return df   
           
        if target_db=="hupd":
            if train_end>=2013 and test_duration!="None":
                return "Error: The end year of the training dataframe cannot be later than year 2012."
            train_df = preprocess_hupd(train_start, train_end)
            test_df = preprocess_hupd(test_start, test_end)   
        elif target_db=="neurips":
            if train_end>3585 or (test_end and test_end>3585):
                return "Error: the dataframe contains 3585 rows in total; the number of rows cannot exceed this limit."
            if train_end>3000 and test_duration!="None":
                return "Error: The end year of the training dataframe cannot exceed row 3000."
            if test_start is not None and test_start!=train_end+1:
                return "Error: test_start must be one more than train_end."
            train_df = preprocess_neurips(train_start, train_end)
            test_df = preprocess_neurips(test_start, test_end)   
        else:
            return "Error: the only possible choices for target_db are hupd (a patent dataset) and neurips (a papers dataset)."
        
        # outcome_col not in columns error
        if isinstance(train_df, str) and "Error:" in train_df:
            return train_df
        if test_df is not None and isinstance(test_df, str) and "Error:" in test_df:
            return test_df
        
        if test_df is None:
            self.data = train_df
            length = len(self.data)
            examples_lst = []
            for column in self.data.columns:
                if column=="decision":
                    examples_lst.append("'"+column+"'"+"(e.g.'ACCEPTED', <class 'str'>)")
                else:
                    i = 0
                    example_data = self.data.at[i, column]
                    while not isinstance(example_data, list) and pd.isna(example_data):
                        i += 1
                        example_data = self.data.at[i, column]
                    if isinstance(example_data, str) and len(example_data)>=10:
                        example_data = str(example_data[:10])+"..."
                    examples_lst.append("'"+column+"'"+"(e.g."+str(example_data)+", {})".format(type(example_data)))
            examples_lst_str = ', '.join(examples_lst)
            self.dataset_dict = None
            return "We have successfully loaded the {} dataframe, including the following columns: {}.".format(target_db, examples_lst_str)+"\nIt has {} rows.".format(length)
        else:
            test_df['Unique_ID'] = test_df.index.map(lambda x:"ID-"+str(x))
            test_df = test_df.reset_index(drop=False)
            train_dataset = Dataset.from_pandas(train_df)
            if outcome_col!="None":
                self.train_groundtruth = train_df[outcome_col]
                self.test_groundtruth = test_df[outcome_col]
                test_df.drop(columns=[outcome_col], inplace=True)
            test_dataset = Dataset.from_pandas(test_df)
            self.dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
            self.data = None
            return "We have successfully loaded the {} dataset dict that has the following structure: {}".format(target_db, self.dataset_dict)
        
    def test_sampler(self, indices):
        if not self.dataset_dict or not self.dataset_dict["test"]:
            return "Error: There is no test set available for sampling."
        test_df = self.dataset_dict["test"].to_pandas()
        indices_lst = indices.split(",")
        new_test_df = test_df[test_df["Unique_ID"].isin(indices_lst)]
        new_test_dataset = Dataset.from_pandas(new_test_df)
        self.dataset_dict["test"] = new_test_dataset
        return "Done sampling the test set according to the specified indices."
    
    def pandas_interpreter(self, pandas_code): 
        """
        Executes the provided Pandas code.
        """
        if self.dataset_dict is not None:
            global_var = {"df": self.dataset_dict["train"].to_pandas()}
        elif self.data is not None:
            global_var = {"df": self.data.copy()}
        else:
            return "Error: Dataframe does not exist. Make sure the dataframe is loaded with LoadDB first."
        try: 
            exec(pandas_code, globals(), global_var)
            variable_values = {}
            for var_name, var_value in locals().items(): 
                if var_name in ["self", "pandas_code","variable_values"]:
                    continue
                elif var_name=="global_var":
                    for global_var_name, global_var_value in var_value.items(): 
                        excluded_types = (types.ModuleType, types.FunctionType, type, pd.DataFrame)
                        if global_var_name not in ["__builtins__", "quit", "copyright", "credit", "license", "help"] and not isinstance(global_var_value, excluded_types):
                            if isinstance(global_var_value, pd.Series):
                                variable_values[global_var_name] = global_var_value.head().to_dict()
                            elif isinstance(global_var_value, (list, dict, np.ndarray)):
                                variable_values[global_var_name] = global_var_value[:10]
                            else:
                                variable_values[global_var_name] = global_var_value
                elif not var_name.startswith('__') and not isinstance(var_value, excluded_types):
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
            return "Error: "+str(e)+"column does not exist.\nThe dataframe contains the following columns: "+column_names_str+". It has the following structure: {}".format(self.data.head())
        except NameError as e:
            if "'pd'" in str(e):
                return "Error: "+str(e)+"\nImport the pandas library using the pandas_interpreter."
        except Exception as e:
            return "Error: "+str(e)
        # other exceptions
    
    # def numerical_classifier(self, model_name, feature, target):
    #     combined_series = pd.concat([self.train_groundtruth, self.test_groundtruth])
    #     value_counts = combined_series.value_counts().to_dict()
    #     unique_classes = combined_series.unique().tolist()
    #     num_classes = len(unique_classes)
    #     class_weights = []
    #     for ind in range(num_classes):
    #         class_weights.append(1/value_counts[unique_classes[ind]])
    #     class_weights = [x/sum(class_weights) for x in class_weights]
    #     print("BEFORE", class_weights)
    #     target_to_label = {} 
    #     for ind in range(num_classes):
    #         target_to_label[unique_classes[ind]] = ind
    #     max_occurrence = max(value_counts.values())
    #     min_occurrence = min(value_counts.values())
    #     if max_occurrence / min_occurrence>10:
    #         for k in range(len(unique_classes)):
    #             unique_class = unique_classes[k]
    #             if value_counts[unique_class] < min_occurrence * 5:
    #             # if class_weights[target_to_label[unique_class]]<1/20:
    #                 train_dataset = self.dataset_dict["train"]
    #                 df = train_dataset.to_pandas()
    #                 subset_df = df[df[target] == unique_class]
    #                 repeated_df = pd.concat([subset_df] * 5, ignore_index=True)
    #                 class_weights[k] /= 6 
    #                 # class_weights[k] *= 6 
    #                 repeated_dataset = Dataset.from_pandas(repeated_df)
    #                 combined_dataset = concatenate_datasets([train_dataset, repeated_dataset])
    #                 self.dataset_dict["train"] = combined_dataset.shuffle(seed=RANDOM_SEED)
    #     class_weights = [x/sum(class_weights) for x in class_weights]
    #     class_weights_dic = {}
    #     for i in range(num_classes):
    #         class_weights_dic[i] = class_weights[i]
    #     CLASS_WEIGHTS = class_weights_dic
    #     print("AFTER", CLASS_WEIGHTS)
        
    #     scaler = MinMaxScaler()
    #     df_train = self.dataset_dict["train"].to_pandas()
    #     df_test = self.dataset_dict["test"].to_pandas()
    #     X_train = df_train[[feature]]
    #     X_test = df_test[[feature]]
    #     scaler.fit(X_train)
    #     X_train_scaled = scaler.transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)
    #     y_train = self.dataset_dict["train"].to_pandas()[target]
    #     if model_name == "decision_tree":
    #         dt_model = DecisionTreeClassifier(class_weight=CLASS_WEIGHTS, random_state=42)
    #         dt_model.fit(X_train_scaled, y_train)
    #         preds = dt_model.predict(X_test_scaled)
    #     elif model_name=="random_forest":
    #         rf_model = RandomForestClassifier(class_weight=CLASS_WEIGHTS, n_estimators=100, random_state=42)
    #         rf_model.fit(X_train_scaled, y_train)
    #         preds = rf_model.predict(X_test_scaled)
    #     elif model_name=="svm":
    #         svm_model = SVC(class_weight=CLASS_WEIGHTS, kernel='linear', probability=True, random_state=42)
    #         svm_model.fit(X_train_scaled, y_train)
    #         preds = svm_model.predict(X_test_scaled)
    #     # print(len(probabilities))
    #     return preds
            
    def textual_classifier(self, model_name, section, target, one_v_all="None"): # text instead of section
        """
        Runs a classificaiton prediction task given a textual input.
        """
        if not self.dataset_dict:
            return "Error: Dataset_dict does not exist."
        if section not in self.dataset_dict["train"].features:
            return "Error: {} column does not exist.\nThe dataset_dict has the following features: {}".format(section, self.dataset_dict["train"].features)
        if target not in self.dataset_dict["train"].features:
            return "Error: {} column does not exist.\nThe dataset_dict has the following features: {}".format(target, self.dataset_dict["train"].features)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        combined_series = pd.concat([self.train_groundtruth, self.test_groundtruth])
        if one_v_all!="None":
            unique_classes = ["not "+one_v_all, one_v_all]
            combined_series_df = pd.DataFrame(combined_series)
            combined_series_df["contains_"+one_v_all] = combined_series_df[target].str.contains(one_v_all).astype(int)
            ind_value_counts = combined_series_df.groupby("contains_"+one_v_all).count().to_dict()[target]
            value_counts = {}
            for key in ind_value_counts.keys():
                value_counts[unique_classes[key]] = ind_value_counts[key]
            num_classes = 2
        else:
            value_counts = combined_series.value_counts().to_dict()
            unique_classes = combined_series.unique().tolist()
            num_classes = len(unique_classes)
        CLASSES = num_classes
        CLASS_NAMES = [i for i in range(CLASSES)]
            
        target_to_label = {} 
        for ind in range(num_classes):
            target_to_label[unique_classes[ind]] = ind
        
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
        alpha_smooth_val=1.0
        
        if num_classes==2:
            naive_bayes_version='Bernoulli' 
        else:
            naive_bayes_version='Multinomial'
                                                    
        # Create a BoW (Bag-of-Words) representation
        def text2bow(input, vocab_size):
            arr = []
            for i in range(input.shape[0]):
                query = input[i]
                if num_classes==2:
                    features = [0] * vocab_size
                else:
                    features = [1] * vocab_size
                for j in range(query.shape[0]):
                    features[query[j]] += 1 
                arr.append(features)
            return np.array(arr)

        # Create model and tokenizer
        def create_model_and_tokenizer(model_name=model_name, dataset=None, section=section, vocab_size=10000, embed_dim=200, n_classes=CLASSES, max_length=512): #'bert-base-uncased'
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            # Finetune
            if model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2']:
                config = AutoConfig.from_pretrained(model_name, num_labels=CLASSES, output_hidden_states=False)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if model_name == 'gpt2':
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.max_length = max_length
                tokenizer.model_max_length = max_length
                model = AutoModelForSequenceClassification.from_config(config=config)
            elif model_name in ['cnn', 'naive_bayes', 'logistic_regression']:
                # Word-level tokenizer
                tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
                # Normalizers
                tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
                # World-level trainer
                trainer = WordLevelTrainer(vocab_size=vocab_size, min_frequency=3, show_progress=True, 
                    special_tokens=special_tokens)
                # Whitespace (pre-tokenizer)
                tokenizer.pre_tokenizer = Whitespace()
                # Train from iterator
                tokenizer.train_from_iterator(dataset['train'][section], trainer=trainer)                
                # Update the vocab size
                vocab_size = tokenizer.get_vocab_size()
                # [PAD] idx
                pad_idx = tokenizer.encode('[PAD]').ids[0]

                tokenizer.enable_padding(pad_type_id=pad_idx)
                tokenizer.pad_token = '[PAD]'
                vocab_size = vocab_size

                if model_name != 'naive_bayes': # CHANGE 'naive_bayes' (shannon)
                    tokenizer.model_max_length = max_length
                    tokenizer.max_length = max_length
                tokenizer.save("/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/temp/temp_tokenizer.json")  # <YOUR_OWN_PATH>
                tokenizer = PreTrainedTokenizerFast(tokenizer_file="/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/temp/temp_tokenizer.json") # <YOUR_OWN_PATH>

                if model_name != 'naive_bayes': 
                    tokenizer.model_max_length = max_length
                    tokenizer.max_length = max_length
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    tokenizer.pad_token = '[PAD]'
                    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
                    tokenizer.sep_token = '[SEP]'

                model = None
                if model_name == 'logistic_regression':
                    model = LogisticRegression(vocab_size=vocab_size, embed_dim=embed_dim, n_classes=n_classes, pad_idx=pad_idx)
                elif model_name == 'cnn':
                    model = BasicCNNModel(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx, n_classes=n_classes, n_filters=n_filters, filter_sizes=filter_sizes[0], dropout=dropout)
            else:
                raise NotImplementedError()
                    
            if model == 'distilbert-base-uncased':
                print(f'Model name: {model_name} \nModel params: {model.num_parameters()}')
            else:
                print(model)
            return tokenizer, dataset, model, vocab_size

        # Map target2string
        def map_target_to_label(example):
            if one_v_all!="None":
                return {'output': int(one_v_all in example[target])}
            else:
                return {'output': target_to_label[example[target]]}
        
        def map_groundtruth_to_label(lst):
            if one_v_all!="None":
                return [int(one_v_all in x) for x in lst]
            else:
                return [target_to_label[x] for x in lst]
        
        # Create dataset
        def create_dataset(tokenizer, section=section):
            data_loaders = []
            dataset = self.dataset_dict['test']
            print('*** Tokenizing...')
            # Tokenize the input
            zero_encoding = tokenizer('', truncation=True, padding='max_length', max_length=max_length)
            dataset = dataset.map(
                lambda e: {
                    section: [
                        tokenizer(text, truncation=True, padding='max_length', max_length=max_length) if text is not None else zero_encoding
                        for text in e[section]
                    ]
                },
                batched=True
            )

            # Flatten the lists of dictionaries into separate columns
            dataset = dataset.map(
                lambda e: {
                    'input_ids': torch.tensor([item['input_ids'] for item in e[section]]),
                    'attention_mask': torch.tensor([item['attention_mask'] for item in e[section]]),
                },
                batched=True
            )
            # Set the dataset format
            gt_list = self.test_groundtruth.to_list()
            gt_list = map_groundtruth_to_label(gt_list)
            dataset = dataset.map(lambda example, idx: {'output': torch.tensor(gt_list[idx])}, with_indices=True)
            dataset.set_format(type='torch', 
                columns=['input_ids', 'attention_mask', 'output'])
            data_loaders.append(DataLoader(dataset, batch_size=batch_size))
            return data_loaders
        
        # def create_data(tokenizer, text=text):
        #     zero_encoding = tokenizer('', truncation=True, padding='max_length', max_length=max_length)
        #     if text is None:
        #         encoded_text = zero_encoding
        #     else:
        #         encoded_text = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
        #     input_ids = torch.tensor(encoded_text['input_ids']).unsqueeze(0)
        #     attention_mask = torch.tensor(encoded_text['attention_mask']).unsqueeze(0)
        #     return {'input_ids': input_ids, 'attention_mask': attention_mask}

        def measure_accuracy(outputs, labels):
            preds = np.argmax(outputs, axis=1).flatten()
            labels = labels.flatten()
            correct = np.sum(preds == labels)
            c_matrix = confusion_matrix(labels, preds, labels=CLASS_NAMES)
            return correct, len(labels), c_matrix

        # Convert ids2string
        def convert_ids_to_string(tokenizer, input):
            return ' '.join(tokenizer.convert_ids_to_tokens(input)) # tokenizer.decode(input)

        # Evaluation procedure (for the neural models)
        def test(test_loader, model, criterion, device, name='test'): # processed_text
            model.eval()
            total_loss = 0.
            total_correct = 0
            total_sample = 0
            total_confusion = np.zeros((CLASSES, CLASSES))
            predictions = []
            
            # inputs = processed_text['input_ids']
            # inputs = inputs.to(device)
            # with torch.no_grad():
            #         if model_name in ['cnn', 'naive_bayes', 'logistic_regression']:
            #             outputs = model(input_ids=inputs)
            #         else:
            #             outputs = model(input_ids=inputs, labels=decisions).logits
            #     preds = torch.argmax(outputs, dim=1)
            #     predictions.extend(preds.cpu().numpy().tolist())
            # return predictions
                
            for i, batch in enumerate(tqdm(test_loader)):                
                inputs, decisions = batch['input_ids'], batch['output']
                inputs = inputs.to(device)
                decisions = decisions.to(device)
                with torch.no_grad():
                    if model_name in ['cnn', 'naive_bayes', 'logistic_regression']:
                        outputs = model(input_ids=inputs)
                    else:
                        outputs = model(input_ids=inputs, labels=decisions).logits
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy().tolist())
                loss = criterion(outputs, decisions) 
                logits = outputs 
                total_loss += loss.cpu().item()
                correct_n, sample_n, c_matrix = measure_accuracy(logits.cpu().numpy(), decisions.cpu().numpy())
                total_confusion += c_matrix
                total_correct += correct_n
                total_sample += sample_n
            
            # Loop over the examples in the test set
            for i, batch in enumerate(tqdm(test_loader)):                
                inputs, decisions = batch['input_ids'], batch['output']
                inputs = inputs.to(device)
                decisions = decisions.to(device)
                with torch.no_grad():
                    if model_name in ['cnn', 'naive_bayes', 'logistic_regression']:
                        outputs = model(input_ids=inputs)
                    else:
                        outputs = model(input_ids=inputs, labels=decisions).logits
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy().tolist())
                loss = criterion(outputs, decisions) 
                logits = outputs 
                total_loss += loss.cpu().item()
                correct_n, sample_n, c_matrix = measure_accuracy(logits.cpu().numpy(), decisions.cpu().numpy())
                total_confusion += c_matrix
                total_correct += correct_n
                total_sample += sample_n
                    
            # Print the performance of the model on the test set 
            print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
            print(f'*** Confusion matrix:\n{total_confusion}')
            
            return predictions

        # Evaluation procedure (for the Naive Bayes models)
        def test_naive_bayes(test_loader, model, vocab_size, name='test', pad_id=-1): # preprocessed_text
            total_loss = 0.
            total_correct = 0
            total_sample = 0
            total_confusion = np.zeros((CLASSES, CLASSES))
            predictions = []
            
            # input = preprocessed_text["input_ids"]
            # input = text2bow(input, vocab_size)
            # input[:, pad_id] = 0
            # logit = model.predict_log_proba(input)
            # probs = np.exp(logit)
            # preds = np.argmax(probs, axis=1)
            # predictions.extend(preds.tolist())
            # return predictions
            
            # Loop over all the examples in the evaluation set
            for i, batch in enumerate(tqdm(test_loader)):
                input, label = batch['input_ids'], batch['output']
                input = text2bow(input, vocab_size)
                input[:, pad_id] = 0
                logit = model.predict_log_proba(input)
                probs = np.exp(logit)
                preds = np.argmax(probs, axis=1)
                predictions.extend(preds.tolist())
                
                label = np.array(label.flatten()) 
                correct_n, sample_n, c_matrix = measure_accuracy(logit, label)
                total_confusion += c_matrix
                total_correct += correct_n
                total_sample += sample_n
            print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
            print(f'*** Confusion matrix:\n{total_confusion}')
            return predictions
        
        if model_name == 'naive_bayes':
                batch_size = 1
    
        # Create a model and an appropriate tokenizer
        tokenizer, self.dataset_dict, model, vocab_size = create_model_and_tokenizer(
            model_name = model_name, 
            dataset = self.dataset_dict,
            section = section,
            vocab_size = vocab_size,
            embed_dim = embed_dim,
            n_classes = CLASSES,
            max_length=max_length
            )

        # GPU specifications 
        if model_name != 'naive_bayes':
            model.to(device)

        # Load the dataset
        data_loaders = create_dataset(
            tokenizer = tokenizer, 
            section = section
            )
        # preprocessed_text = create_data(
        #     tokenizer = tokenizer, 
        #     text = text
        # )
        del self.dataset_dict
            
        if model_name == 'naive_bayes': 
            print('Here we are!')
            predictions = test_naive_bayes(data_loaders[0], tokenizer, vocab_size, naive_bayes_version, alpha_smooth_val) # preprocessed_text
        else:
            # Optimizer
            if model_name in ['logistic_regression', 'cnn']:
                optim = torch.optim.Adam(params=model.parameters())
            else:
                optim = torch.optim.AdamW(params=model.parameters(), lr=lr, eps=eps)
            # Loss function 
            criterion = torch.nn.CrossEntropyLoss() 
            
            # Train and validate
            predictions = test(data_loaders[0], epoch_n, model, optim, criterion, device) # preprocessed_text
        predictions_to_categories = [unique_classes[x] for x in predictions]
        return {"predictions": predictions_to_categories[:10]} # limit to the first 10 values to prevent content limit exceeded

if __name__ == "__main__":
    db = table_toolkits()
    # db.db_loader('hupd', '2016-2016', 'None', 'None')
    print(db.db_loader('neurips', '0-3585', 'None', 'None'))
    # pandas_code = "import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0]"
    # print(db.pandas_interpreter(pandas_code))
    
    # print("1")
    # db.db_loader('hupd', '2004-2006', '2004-2007', 'decision')
    # db.textual_classifier('naive_bayes', 'summary', 'decision', 'ACCEPTED')
    # print("2")
    # db.db_loader('hupd', '2004-2006', '2004-2007', 'decision')
    # db.textual_classifier('naive_bayes', 'summary', 'decision', 'ACCEPTED')
    # print("3")
    # db.db_loader('hupd', '2004-2006', '2004-2007', 'decision')
    # db.textual_classifier('cnn', 'summary', 'decision', 'ACCEPTED')
    # print("4")
    # db.db_loader('hupd', '2004-2006', '2004-2007', 'decision')
    # db.textual_classifier('naive_bayes', 'summary', 'decision', 'ACCEPTED')
    # print("5")
    # db.db_loader('hupd', '2004-2006', '2004-2007', 'decision')
    # db.textual_classifier('logistic_regression', 'summary', 'decision', 'ACCEPTED')
    # print("6")
    # db.db_loader('neurips', '0-1000', '1001-3585', 'Oral')
    # db.textual_classifier('cnn', 'Abstract', 'Oral')
    # print("7")
    # db.db_loader('neurips', '0-1000', '1001-3585', 'Oral')
    # db.textual_classifier('cnn', 'Abstract', 'Oral')
    # print("8")
    # db.db_loader('neurips', '0-1000', '1001-3585', 'Oral')
    # db.textual_classifier('cnn', 'Abstract', 'Oral')
    # print("9")
    # db.db_loader('neurips', '0-1000', '1001-3585', 'Oral')
    # db.textual_classifier('distilbert-base-uncased', 'Abstract', 'Oral')
    # print("10")
    # db.db_loader('neurips', '0-1000', '1001-3585', 'Oral')
    # db.textual_classifier('logistic_regression', 'Abstract', 'Oral')
    # logistic_regression, distilbert-base-uncased, cnn, naive_bayes hupd: # title, abstract, summary, claims, background, full_description # decision    

    # db.db_loader('neurips', '0-1000', 'None', 'None')
    # db.db_loader('neurips', '0-1000', '1001-3585', 'Oral')
    # # db.pandas_interpreter('df[author_num]=df[Authors].apply(len)')
    # preds = db.textual_classifier('cnn', 'Abstract', 'Oral')["predictions"]
    # # preds = db.numerical_classifier('random_forest', 'Authors_Num', 'Oral')
    # path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools" #<YOUR_OWN_PATH>
    # actual_df = pd.read_csv("{}/data/external_corpus/neurips/NeurIPS_2023_Papers.csv".format(path))
    # actual_df = actual_df.iloc[1001:]
    # actual_df.dropna(subset=['Oral'], inplace=True)
    # actual = actual_df["Oral"].tolist()
    # # print(len(actual))
    # f1_macro = f1_score(actual, preds, average='macro') 
    # print("omo cnn nb", f1_macro)

    # db.db_loader('neurips', "0-2000", "2001-3585", "Poster Session")
    # db.test_sampler("ID-2001,ID-2500,ID-2486,ID-2759,ID-3300")
    # print(db.textual_classifier("logistic_regression", "Abstract", "Poster Session", "None"))

    # db.db_loader('neurips', '0-3585', 'None', 'None')
    
    