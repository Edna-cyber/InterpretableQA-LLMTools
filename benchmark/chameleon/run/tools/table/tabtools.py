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
from tools.table.pred_models import BasicCNNModel, BigCNNModel, LogisticRegression # tools.table.

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

# Fixing the random seeds
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class table_toolkits():
    # init
    def __init__(self):
        self.data = None
        self.dataset_dict = None
        self.train_duration = None
        self.path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools" #<YOUR_OWN_PATH>

    def db_loader(self, target_db, train_duration="None", test_duration="None", outcome_col="None"): # e.g. duration can be 2005-2012 or 0-2000, string type, both sides inclusive
        self.train_duration = train_duration
        self.test_duration = test_duration
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
            if not start_year and not end_year:
                return None
            df = []
            for sub in range(start_year, end_year+1):
                file_path = "{}/data/external_corpus/hupd/hupd_{}.csv".format(self.path, sub)
                df_raw = pd.read_csv(file_path)
                if target_db=="hupd":
                    df_raw['patent_number'] = pd.to_numeric(df_raw['patent_number'], errors='coerce').astype('Int64').replace(0, pd.NA) # so that the LLM is aware of which patent numbers are invalid 
                    df_raw['examiner_id'] = pd.to_numeric(df_raw['examiner_id'], errors='coerce').astype('Int64') 
                    
                    df_raw['filing_date'] = pd.to_datetime(df_raw['filing_date'])
                    df_raw['patent_issue_date'] = pd.to_datetime(df_raw['patent_issue_date'])
                    df_raw['date_published'] = pd.to_datetime(df_raw['date_published'])
                    
                    df_raw["icpr_category"] = df_raw["main_ipcr_label"].apply(lambda x:x[:3] if isinstance(x, str) else x)
                    df_raw["cpc_category"] = df_raw["main_cpc_label"].apply(lambda x:x[:3] if isinstance(x, str) else x)
                # print(df_raw.dtypes)
                # print(df_raw.head())
                df.append(df_raw)
            df = pd.concat(df, ignore_index=True)
            return df
        def preprocess_neurips(start_row, end_row):
            if not start_row and not end_row:
                return None
            file_path = "{}/data/external_corpus/neurips/NeurIPS_2023_Papers.csv"
            df = pd.read_csv(file_path)
            # print(df.dtypes)
            df = df.iloc[start_row:end_row+1]
            return df            
            
        if target_db=="hupd":
            train_df = preprocess_hupd(train_start, train_end)
            test_df = preprocess_hupd(test_start, test_end)    
        elif target_db=="neurips":
            train_df = preprocess_neurips(train_start, train_end)
            test_df = preprocess_neurips(test_start, test_end)   
        else:
            return "Error: the only possible choices for target_db are hupd (a patent dataset) and neurips (a papers dataset)."
        
        if not test_df:
            self.data = train_df
            column_names = ["'"+x+"'" for x in self.data.columns.tolist()]
            column_names_str = ', '.join(column_names)
            self.dataset_dict = None
            return "We have successfully loaded the {} dataframe, including the following columns: {}.".format(target_db, column_names_str)+"\nIt has the following structure: {}".format(self.data.head()) 
        else:
            train_dataset = Dataset.from_pandas(train_df)
            try:
                test_df.drop(columns=[outcome_col], inplace=True)
            except:
                column_names = ["'"+x+"'" for x in test_df.columns.tolist()]
                column_names_str = ', '.join(column_names)
                return "The outcome_col does not exist in the dataframe. Choose from the following columns: {}.".format(column_names_str)
            test_dataset = Dataset.from_pandas(test_df)
            self.dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
            self.data = None
            return "We have successfully loaded the {} dataset dict that has the following structure: {}".format(target_db, self.dataset_dict)
        
    def pandas_interpreter(self, pandas_code): 
        """
        Executes the provided Pandas code and updates the 'ans' in global_var from the loaded dataframe.
        """
        if self.data is None:
            return "Error: Dataframe does not exist. Make sure the dataframe is loaded with LoadDB first."
        else:
            global_var = {"df": self.data.copy(), "ans": None}
            try: 
                exec(pandas_code, global_var)
                if not global_var['ans']:
                    return "Error: ans is None.\nThe final result must be assigned to ans variable."
                return str(global_var['ans'])
            except KeyError as e:
                column_names = ["'"+x+"'" for x in self.data.columns.tolist()]
                column_names_str = ', '.join(column_names)
                return "Error: "+str(e)+"column does not exist.\nThe dataframe contains the following columns: "+column_names_str+". It has the following structure: {}".format(self.data.head())
            except NameError as e:
                if "'pd'" in str(e):
                    return "Error: "+str(e)+"\nImport the pandas library using the pandas_interpreter."
                return "Error: "+str(e)+"\nMake sure the dataframe is loaded with LoadDB first. If so, the dataframe is stored in variable df."
            except Exception as e:
                return "Error: "+str(e)
            # other exceptions
            
    def classifier(self, model_name, section, target, num_classes=2):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        CLASSES = num_classes
        CLASS_NAMES = [i for i in range(CLASSES)]
        
        val = False
        tokenizer_path = None
        model_path = None
        vocab_size = 10000
        batch_size=64
        val_every=500
        n_filters=25
        filter_sizes=[[3,4,5], [5,6,7], [7,9,11]]
        dropout=0.25
        epoch_n=5
        lr=2e-5
        eps=1e-8
        pos_class_weight=0
        naive_bayes_version='Bernoulli' ###
        embed_dim=200
        max_length=256
        alpha_smooth_val=1.0
        np_filename=None
        use_scheduler=False
        cpc_label=None
        ipc_label="G06F"
        train_from_scratch=False
        
        label = None
        if cpc_label:
            label = cpc_label
        else:
            label = ipc_label
            
        tokenizer_save_path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/table/models/"+model_name+"_"+label+"_"+self.duration+"_tokenizer" ### might need to change for different tasks
        save_path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/table/models/"+model_name+"_"+label+"_"+self.duration ### might need to change for different tasks
        filename = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/table/"+model_name+"_"+label+"_"+self.duration+".txt" ### might need to change for different tasks
                
        # Subject area code label
        cat_label = ''
        if cpc_label:
            cat_label = f'CPC_{cpc_label}'
        elif ipc_label:
            cat_label = f'IPC_{ipc_label}'
        else:
            cat_label = 'All_IPCs'
                    
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
        def create_model_and_tokenizer(train_from_scratch=False, val=False, model_name='bert-base-uncased', dataset=None, section=section, vocab_size=10000, embed_dim=200, n_classes=CLASSES, max_length=512):
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

            if val:
                if model_name == 'distilbert-base-uncased':
                    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path) 
                    model = DistilBertForSequenceClassification.from_pretrained(model_path)
                    # This step is actually important.
                    tokenizer.max_length = max_length
                    tokenizer.model_max_length = max_length
                else:
                    raise NotImplementedError
            else:
                # Train from scratch
                if train_from_scratch:
                    if model_name == 'bert-base-uncased':
                        config = BertConfig(num_labels=CLASSES, output_hidden_states=False) 
                        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
                        model = BertForSequenceClassification(config=config)
                    elif model_name == 'distilbert-base-uncased':
                        config = DistilBertConfig(num_labels=CLASSES, output_hidden_states=False) 
                        tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
                        model = DistilBertForSequenceClassification(config=config)
                    elif model_name == 'roberta-base':
                        config = RobertaConfig(num_labels=CLASSES, output_hidden_states=False) 
                        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
                        model = RobertaForSequenceClassification(config=config)
                    elif model_name == 'gpt2':
                        config = GPT2Config(num_labels=CLASSES, output_hidden_states=False) 
                        tokenizer = GPT2Tokenizer.from_pretrained(model_name, do_lower_case=True)
                        model = GPT2ForSequenceClassification(config=config)
                    elif model_name == 'allenai/longformer-base-4096':
                        config = LongformerConfig(num_labels=CLASSES, output_hidden_states=False) 
                        tokenizer = LongformerTokenizer.from_pretrained(model_name, do_lower_case=True)
                        model = LongformerForSequenceClassification(config=config)
                    else:
                        raise NotImplementedError()

                # Finetune
                else:
                    if model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
                        config = AutoConfig.from_pretrained(model_name, num_labels=CLASSES, output_hidden_states=False)
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        if model_name == 'gpt2':
                            tokenizer.pad_token = tokenizer.eos_token
                        tokenizer.max_length = max_length
                        tokenizer.model_max_length = max_length
                        model = AutoModelForSequenceClassification.from_config(config=config)
                    elif model_name in ['lstm', 'cnn', 'big_cnn', 'naive_bayes', 'logistic_regression']:
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

                        # Currently the call method for WordLevelTokenizer is not working.
                        # Using this temporary method until the tokenizers library is updated.
                        # Not a fan of this method, but this is the best we have right now (sad face).
                        # Based on https://github.com/huggingface/transformers/issues/7234#issuecomment-720092292
                        tokenizer.enable_padding(pad_type_id=pad_idx)
                        tokenizer.pad_token = '[PAD]'
                        vocab_size = vocab_size

                        if model_name != 'naive_bayes': # CHANGE 'naive_bayes' (shannon)
                            tokenizer.model_max_length = max_length
                            tokenizer.max_length = max_length
                        tokenizer.save("/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/table/temp_tokenizer.json")  # <YOUR_OWN_PATH>
                        if tokenizer_save_path:
                            print('*** Saving the tokenizer...')
                            tokenizer.save(f"{tokenizer_save_path}")
                        tokenizer = PreTrainedTokenizerFast(tokenizer_file="/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/table/temp_tokenizer.json") # <YOUR_OWN_PATH>

                        if model_name != 'naive_bayes': # CHANGE 'naive_bayes'
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
                        elif model_name == 'big_cnn':
                            model = BigCNNModel(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx, n_classes=n_classes, n_filters=n_filters, filter_sizes=filter_sizes, dropout=dropout)
                    else:
                        raise NotImplementedError()
                    
            if model in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
                print(f'Model name: {model_name} \nModel params: {model.num_parameters()}')
            else:
                print(model)
            return tokenizer, dataset, model, vocab_size
        
        # For filtering out CONT-apps and pending apps
        decision_to_str = {
            'REJECTED': 0, 
            'ACCEPTED': 1, 
            'PENDING': 2, 
            'CONT-REJECTED': 3, 
            'CONT-ACCEPTED': 4, 
            'CONT-PENDING': 5
        }

        # Map decision2string
        def map_decision_to_string(example):
            return {'output': decision_to_str[example[target]]}
        
        # Create dataset
        def create_dataset(tokenizer, section=section):
            data_loaders = []
            for name in ['train', 'test']:
                # Skip the training set if we are doing only inference
                if val and name=='train':
                    data_loaders.append(None)
                else:
                    dataset = self.dataset_dict[name]
                    print('*** Tokenizing...')
                    # Tokenize the input
                    dataset = dataset.map(
                        lambda e: tokenizer(e[section], truncation=True, padding='max_length'),
                        batched=True)
                    # Set the dataset format
                    dataset.set_format(type='torch', 
                        columns=['input_ids', 'attention_mask', 'output'])
                    data_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=(name=='train')))
            return data_loaders

        # Return label statistics of the dataset loader
        def dataset_statistics(dataset_loader, tokenizer):
            label_stats = collections.Counter()
            for i, batch in enumerate(tqdm(dataset_loader)):
                inputs, decisions = batch['input_ids'], batch['output']
                labels = decisions.cpu().numpy().flatten()
                label_stats += collections.Counter(labels)
            return label_stats

        # Calculate TOP1 accuracy
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
        def test(val_loader, model, criterion, device, name='test', write_file=None):
            model.eval()
            total_loss = 0.
            total_correct = 0
            total_correct_class_level = 0
            total_sample = 0
            total_confusion = np.zeros((CLASSES, CLASSES))
            
            # Loop over the examples in the evaluation set
            for i, batch in enumerate(tqdm(val_loader)):
                inputs, decisions = batch['input_ids'], batch['output']
                inputs = inputs.to(device)
                decisions = decisions.to(device)
                with torch.no_grad():
                    if model_name in ['lstm', 'cnn', 'big_cnn', 'naive_bayes', 'logistic_regression']:
                        outputs = model(input_ids=inputs)
                    else:
                        outputs = model(input_ids=inputs, labels=decisions).logits
                loss = criterion(outputs, decisions) 
                logits = outputs 
                total_loss += loss.cpu().item()
                correct_n, sample_n, c_matrix = measure_accuracy(logits.cpu().numpy(), decisions.cpu().numpy())
                total_confusion += c_matrix
                total_correct += correct_n
                total_sample += sample_n
            
            # Print the performance of the model on the test set 
            print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
            print(f'*** Class-level accuracy on the {name} set: {total_correct_class_level/total_sample}')
            print(f'*** Confusion matrix:\n{total_confusion}')
            if write_file:
                write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
                write_file.write(f'*** Class-level accuracy on the {name} set: {total_correct_class_level/total_sample}\n')
                write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')

            return total_loss, float(total_correct/total_sample) * 100.


        # Training procedure (for the neural models)
        def train(data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file=None):
            print('\n>>>Training starts...')
            if write_file:
                write_file.write(f'\n>>>Training starts...\n')
            # Training mode is on
            model.train()
            # Best test set accuracy so far.
            best_val_acc = 0
            for epoch in range(epoch_n):
                total_train_loss = 0.
                # Loop over the examples in the training set.
                for i, batch in enumerate(tqdm(data_loaders[0])):
                    inputs, decisions = batch['input_ids'], batch['output']
                    inputs = inputs.to(device, non_blocking=True)
                    decisions = decisions.to(device, non_blocking=True)
                    
                    # Forward pass
                    if model_name in ['lstm', 'cnn', 'big_cnn', 'logistic_regression']:
                        outputs = model (input_ids=inputs)
                    else:
                        outputs = model(input_ids=inputs, labels=decisions).logits
                    loss = criterion(outputs, decisions) #outputs.logits
                    total_train_loss += loss.cpu().item()

                    # Backward pass
                    loss.backward()
                    optim.step()
                    if scheduler:
                        scheduler.step()
                    optim.zero_grad()

                    # Print the loss every val_every step
                    if i % val_every == 0:
                        print(f'*** Loss: {loss}')
                        print(f'*** Input: {convert_ids_to_string(tokenizer, inputs[0])}')
                        if write_file:
                            write_file.write(f'\nEpoch: {epoch}, Step: {i}\n')
                        # Get the performance of the model on the test set
                        _, val_acc = test(data_loaders[1], model, criterion, device, write_file=write_file)
                        model.train()
                        if best_val_acc < val_acc:
                            best_val_acc = val_acc
                            # Save the model if a save directory is specified
                            if save_path:
                                # If the model is a Transformer architecture, make sure to save the tokenizer as well
                                if model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
                                    model.save_pretrained(save_path)
                                    tokenizer.save_pretrained(save_path + '_tokenizer')
                                else:
                                    torch.save(model.state_dict(), save_path)

            # Training is complete!
            print(f'\n ~ The End ~')
            if write_file:
                write_file.write('\n ~ The End ~\n')
            
            # Final evaluation on the test set
            _, val_acc = test(data_loaders[1], model, criterion, device, name='test', write_file=write_file)
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                
                # Save the best model so fare
                if save_path:
                    if model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
                        model.save_pretrained(save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
            
            # Additionally, print the performance of the model on the training set if we were not doing only inference
            if not test:
                _, train_val_acc = test(data_loaders[0], model, criterion, device, name='train')
                print(f'*** Accuracy on the training set: {train_val_acc}.')
                if write_file:
                    write_file.write(f'\n*** Accuracy on the training set: {train_val_acc}.')
            
            # Print the highest accuracy score obtained by the model on the test set
            print(f'*** Highest accuracy on the test set: {best_val_acc}.')
            if write_file:
                write_file.write(f'\n*** Highest accuracy on the test set: {best_val_acc}.')


        # Evaluation procedure (for the Naive Bayes models)
        def test_naive_bayes(data_loader, model, vocab_size, name='test', write_file=None, pad_id=-1):
            total_loss = 0.
            total_correct = 0
            total_sample = 0
            total_confusion = np.zeros((CLASSES, CLASSES))
            
            # Loop over all the examples in the evaluation set
            for i, batch in enumerate(tqdm(data_loader)):
                input, label = batch['input_ids'], batch['output']
                input = text2bow(input, vocab_size)
                input[:, pad_id] = 0
                logit = model.predict_log_proba(input)
                label = np.array(label.flatten()) 
                correct_n, sample_n, c_matrix = measure_accuracy(logit, label)
                total_confusion += c_matrix
                total_correct += correct_n
                total_sample += sample_n
            print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
            print(f'*** Confusion matrix:\n{total_confusion}')
            if write_file:
                write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
                write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')
            return total_loss, float(total_correct/total_sample) * 100.


        # Training procedure (for the Naive Bayes models)
        def train_naive_bayes(data_loaders, tokenizer, vocab_size, version='Bernoulli', alpha=1.0, write_file=None, np_filename=None):
            pad_id = tokenizer.encode('[PAD]') # NEW
            print(f'Training a {version} Naive Bayes classifier (with alpha = {alpha})...')
            write_file.write(f'Training a {version} Naive Bayes classifier (with alpha = {alpha})...\n')

            # Bernoulli or Multinomial?
            if version == 'Bernoulli':
                model = BernoulliNB(alpha=alpha) 
            elif version == 'Multinomial':
                model = MultinomialNB(alpha=alpha) 
            
            # Loop over all the examples in the training set
            for i, batch in enumerate(tqdm(data_loaders[0])):
                input, decision = batch['input_ids'], batch['output']
                input = text2bow(input, vocab_size) # change text2bow(input[0], vocab_size)
                input[:, pad_id] = 0 # get rid of the paddings
                label = np.array(decision.flatten())
                # Using "partial fit", instead of "fit", to avoid any potential memory problems
                # model.partial_fit(np.array([input]), np.array([label]), classes=CLASS_NAMES)
                model.partial_fit(input, label, classes=CLASS_NAMES)
            
            print('\n*** Accuracy on the training set ***')
            test_naive_bayes(data_loaders[0], model, vocab_size, 'training', write_file, pad_id)
            print('\n*** Accuracy on the test set ***')
            test_naive_bayes(data_loaders[1], model, vocab_size, 'test', write_file, pad_id)
            
            # Save the log probabilities if np_filename is specified
            if np_filename:
                np.save(f'{np_filename}.npy', np.array(model.feature_log_prob_))
        
        if val and model_path is not None and tokenizer_path is None:
            tokenizer_path = model_path + '_tokenizer'

        filename = filename
        if filename is None:
            if model_name == 'naive_bayes':
                filename = f'./results/{model_name}/{naive_bayes_version}/{cat_label}_{section}.txt'
            else:
                filename = f'./results/{model_name}/{cat_label}_{section}_embdim{embed_dim}_maxlength{max_length}.txt'
        write_file = open(filename, "w")
        
        if model_name == 'naive_bayes':
                batch_size = 1
            
        for name in ['train', 'test']:
            self.dataset_dict[name] = self.dataset_dict[name].map(map_decision_to_string)
            # Remove the pending and CONT-patent applications
            self.dataset_dict[name] = self.dataset_dict[name].filter(lambda e: e['output'] <= 1)
    
        # Create a model and an appropriate tokenizer
        tokenizer, self.dataset_dict, model, vocab_size = create_model_and_tokenizer(
            train_from_scratch = train_from_scratch, 
            model_name = model_name, 
            dataset = self.dataset_dict,
            section = section,
            vocab_size = vocab_size,
            embed_dim = embed_dim,
            n_classes = CLASSES,
            max_length=max_length
            )
        
        print(f'*** CPC Label: {cat_label}') 
        print(f'*** Section: {section}')
        print(f'*** Vocabulary: {vocab_size}')

        if write_file:
            write_file.write(f'*** CPC Label: {cat_label}\n')
            write_file.write(f'*** Section: {section}\n')
            write_file.write(f'*** Vocabulary: {vocab_size}\n')

        # GPU specifications 
        if model_name != 'naive_bayes':
            model.to(device)

        # Load the dataset
        data_loaders = create_dataset(
            tokenizer = tokenizer, 
            section = section
            )
        del self.dataset_dict

        if not test:
            # Print the statistics
            train_label_stats = dataset_statistics(data_loaders[0], tokenizer)
            print(f'*** Training set label statistics: {train_label_stats}')
            val_label_stats = dataset_statistics(data_loaders[1], tokenizer)
            print(f'*** Validation set label statistics: {val_label_stats}')
            if write_file:
                write_file.write(f'*** Training set label statistics: {train_label_stats}\n')
                write_file.write(f'*** Validation set label statistics: {val_label_stats}\n\n')
            

        if model_name == 'naive_bayes': 
            tokenizer.save("multilabel_ipc_nb_abstract.json") ## GET RID OF THIS
            print('Here we are!')
            train_naive_bayes(data_loaders, tokenizer, vocab_size, naive_bayes_version, alpha_smooth_val, write_file, np_filename)
        else:
            # Optimizer
            if model_name in ['logistic_regression', 'cnn', 'big_cnn', 'lstm']:
                optim = torch.optim.Adam(params=model.parameters())
            else:
                optim = torch.optim.AdamW(params=model.parameters(), lr=lr, eps=eps)
                total_steps = len(data_loaders[0]) * epoch_n if not test else 0
            # Scheduler
            scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = 0, num_training_steps = total_steps) if use_scheduler else None
            # Class weights
            if pos_class_weight > 0. and pos_class_weight < 1.:
                class_weights = torch.tensor([pos_class_weight, 1. - pos_class_weight]).to(device)
            else:
                class_weights = None
            # Loss function 
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            
            if write_file:
                write_file.write(f'\nModel:\n {model}\nOptimizer: {optim}\n')
            
            # Train and validate
            train(data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file)

            # Save the model
            if save_path:
                tokenizer.save_pretrained(save_path + '_tokenizer')

        if write_file:
            write_file.close()
    

if __name__ == "__main__":
    db = table_toolkits()
    # db.db_loader('hupd', '2016-2016', False)
    # # pandas_code = "import pandas as pd\naccepted_patents = df[df['decision'] == 'ACCEPTED'].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted"
    # pandas_code = "import pandas as pd\napproval_rates = df.groupby('ipcr_category')['decision'].apply(lambda x: (x == 'ACCEPTED').mean() * 100).reset_index(name='approval_rate')\ntop_categories = approval_rates.nlargest(2, 'approval_rate')['ipcr_category'].tolist()\nans = top_categories"
    # print(db.pandas_interpreter(pandas_code))

    print(db.db_loader('hupd', '2016-2016', True))
    db.classifier('logistic_regression', 'summary', 'decision')
    
    
    
