import os
import random
import numpy as np
import collections
from tqdm import tqdm
import pandas as pd
import jsonlines
import json
import re

import torch
from torch.utils.data import DataLoader

# Hugging Face datasets
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# Good old Transformer models
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config
from transformers import LongformerForSequenceClassification, LongformerTokenizer, LongformerConfig
from transformers import PreTrainedTokenizerFast

# Import the sklearn Multinomial Naive Bayes
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# Simple LSTM, CNN, and Logistic regression models
from models import BasicCNNModel, BigCNNModel, LogisticRegression

# Tokenizer-releated dependencies
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# For scheduling 
from transformers import get_linear_schedule_with_warmup

# Confusion matrix
from sklearn.metrics import confusion_matrix

class table_toolkits():
    # init
    def __init__(self):
        self.data = None
        self.dataset_dict = None
        self.path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/" #<YOUR_OWN_PATH>

    def db_loader(self, target_db, duration, split=False): # change examples and description in prompt policy # todo: for forecasting tasks, different loading
        df = []
        hyphen_ind = duration.index("-")
        start_year = int(duration[:hyphen_ind])
        end_year = int(duration[hyphen_ind+1:])
        for sub in range(start_year, end_year+1):
            file_path = "{}/data/external_corpus/{}/{}_{}.csv".format(self.path, target_db, target_db, sub)
            df.append(pd.read_csv(file_path))
        df = pd.concat(df, ignore_index=True)
        if not split:
            self.data = df
            column_names = ', '.join(self.data.columns.tolist())
            self.dataset_dict = None
            return "We have successfully loaded the {} dataframe, including the following columns: {}.".format(target_db, column_names)
        else:
            train_df, validation_df = train_test_split(df, test_size=0.4, random_state=42)
            train_dataset = Dataset.from_pandas(train_df)
            validation_dataset = Dataset.from_pandas(validation_df)
            self.dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset})
            self.data = None
            return "We have successfully loaded the {} dataset dict that has the following structure: {}.".format(target_db, self.dataset_dict)
    
    # remove rows where the target column is NA or unwanted value
    # condition can be e.g. "not NA", "keep ACCEPT,REJECT", "remove 0,1" etc.
    def target_filter(self, target_col, condition):
        [ins, vals] = condition.split()
        val_lst = vals.split(",")
        if self.data is not None:
            self.data.dropna(subset=[target_col], inplace=True)
            if condition=="not NA":
                return 
            if ins=="keep":
                self.data = self.data[self.data[target_col].isin(val_lst)]
            elif ins=="remove":
                self.data = self.data[~self.data[target_col].isin(val_lst)]
 
        elif self.dataset_dict is not None:
            def filter_dataset(dataset):
                dataset = dataset.filter(lambda example: example[target_col] is not None)
                if ins == "keep":
                    dataset = dataset.filter(lambda example: example[target_col] in val_lst)
                elif ins == "remove":
                    dataset = dataset.filter(lambda example: example[target_col] not in val_lst)
                return dataset
            self.dataset_dict['train'] = filter_dataset(self.dataset_dict['train'])
            self.dataset_dict['validation'] = filter_dataset(self.dataset_dict['validation'])
    
    # split can be "all" for self.data, "train", "validation"
    def pandas_interpreter(self, pandas_code, split): 
        """
        Executes the provided Pandas code and updates the 'ans' in global_var from the loaded dataframe.
        """
        if self.data is not None:
            global_var = {"df": self.data.copy(), "ans": 0}
        else:
            global_var = {"df": self.dataset_dict[split].to_pandas().copy(), "ans": 0}
        exec(pandas_code, global_var)
        return str(global_var['ans'])

if __name__ == "__main__":
    db = table_toolkits()
    # print(db.db_loader('hupd', '2017-2017'))
    db.auto_db_loader('hupd', train_start='2016-12-30', val_end='2017-01-02')
    db.target_filter("decision", "not NA")
    pandas_code = "import pandas as pd\naccepted_patents = df[df['decision'] == 'ACCEPTED'].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted"
    print(db.pandas_interpreter(pandas_code, "train"))
