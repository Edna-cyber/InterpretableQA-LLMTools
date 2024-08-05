import os
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

# Good old Transformer models
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
# from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
# from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import PreTrainedTokenizerFast

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
            return "Error: outcome_col needs to be set to None when test_duration is None."
        
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
                df_raw['patent_number'] = pd.to_numeric(df_raw['patent_number'], errors='coerce').astype('Int64').replace(0, pd.NA) # so that the LLM is aware of which patent numbers are invalid 
                df_raw['examiner_id'] = pd.to_numeric(df_raw['examiner_id'], errors='coerce').astype('Int64') 
                
                df_raw['filing_date'] = pd.to_datetime(df_raw['filing_date'])
                df_raw['patent_issue_date'] = pd.to_datetime(df_raw['patent_issue_date'])
                df_raw['date_published'] = pd.to_datetime(df_raw['date_published'])
                
                df_raw["icpr_category"] = df_raw["main_ipcr_label"].apply(lambda x:x[:3] if isinstance(x, str) else x)
                df_raw["cpc_category"] = df_raw["main_cpc_label"].apply(lambda x:x[:3] if isinstance(x, str) else x)
                df_raw.drop(columns=["main_ipcr_label", "main_cpc_label"], inplace=True)
                column_names = ["'"+x+"'" for x in df_raw.columns.tolist()]
                column_names_str = ', '.join(column_names)
                if outcome_col not in df_raw.columns and outcome_col!="None":
                    return "Error: The outcome_col does not exist in the dataframe. Choose from the following columns: {}.".format(column_names_str)
                # remove rows where the predicted target is NA
                if outcome_col!="None":
                    df_raw.dropna(subset=[outcome_col], inplace=True)
                if outcome_col=="decision": # only use "ACCEPTED" and "REJECTED" column
                    df_raw = df_raw[(df_raw[outcome_col]=="ACCEPTED") | (df_raw[outcome_col]=="REJECTED")]
                # print(df_raw.dtypes)
                # print(df_raw.head())
                df.append(df_raw)
            df = pd.concat(df, ignore_index=True)
            return df
        
        def preprocess_neurips(start_row, end_row):
            if not start_row and not end_row:
                return None
            file_path = "{}/data/external_corpus/neurips/NeurIPS_2023_Papers.csv".format(self.path)
            df = pd.read_csv(file_path)
            # print(df.dtypes)
            df = df.iloc[start_row:end_row+1]
            return df   
           
        if target_db=="hupd":
            if train_end>=2013 and test_duration!="None":
                return "Error: The end year of the training dataframe cannot be later than year 2012."
            train_df = preprocess_hupd(train_start, train_end)
            test_df = preprocess_hupd(test_start, test_end)   
        elif target_db=="neurips":
            if train_end>3583 or test_end>3585:
                return "Error: the dataframe contains 3585 rows in total; the number of rows cannot exceed this limit."
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
            column_names = ["'"+x+"'" for x in self.data.columns.tolist()]
            column_names_str = ', '.join(column_names)
            self.dataset_dict = None
            return "We have successfully loaded the {} dataframe, including the following columns: {}.".format(target_db, column_names_str)+"\nIt has the following structure: {}".format(self.data.head()) 
        else:
            train_dataset = Dataset.from_pandas(train_df)
            if outcome_col!="None":
                self.train_groundtruth = train_df[outcome_col]
                self.test_groundtruth = test_df[outcome_col]
                test_df.drop(columns=[outcome_col], inplace=True)
            test_dataset = Dataset.from_pandas(test_df)
            self.dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
            self.data = None
            return "We have successfully loaded the {} dataset dict that has the following structure: {}".format(target_db, self.dataset_dict)
        
    def pandas_interpreter(self, pandas_code): 
        """
        Executes the provided Pandas code.
        """
        if self.data is None:
            return "Error: Dataframe does not exist. Make sure the dataframe is loaded with LoadDB first."
        else:
            global_var = {"df": self.data.copy()}
            try: 
                exec(pandas_code, global_var)
                variable_values = {}
                for var_name, var_value in locals().items(): 
                    if var_name in ["self", "pandas_code","variable_values"]:
                        continue
                    elif var_name=="global_var":
                        for global_var_name, global_var_value in var_value.items(): 
                            excluded_types = (types.ModuleType, types.FunctionType, type)
                            if global_var_name not in ["df", "__builtins__", "quit", "copyright", "credit", "license", "help"] and not isinstance(global_var_value, excluded_types):
                                pd_types = (pd.DataFrame, pd.Series)
                                if isinstance(global_var_value, pd_types):
                                    variable_values[global_var_name] = global_var_value.head().to_dict()
                                else:
                                    variable_values[global_var_name] = global_var_value
                    elif not var_name.startswith('__') and not isinstance(var_value, excluded_types):
                        if isinstance(var_value, pd_types):
                            variable_values[var_name] = var_value.head().to_dict()
                        else:
                            variable_values[var_name] = var_value
                return variable_values
            except KeyError as e:
                column_names = ["'"+x+"'" for x in self.data.columns.tolist()]
                column_names_str = ', '.join(column_names)
                return "Error: "+str(e)+"column does not exist.\nThe dataframe contains the following columns: "+column_names_str+". It has the following structure: {}".format(self.data.head())
            except NameError as e:
                if "'pd'" in str(e):
                    return "Error: "+str(e)+"\nImport the pandas library using the pandas_interpreter."
            except Exception as e:
                return "Error: "+str(e)
            # other exceptions
            
    def textual_classifier(self, model_name, section, target):
        """
        Runs a classificaiton prediction task given a textual input.
        """
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        combined_series = pd.concat([self.train_groundtruth, self.test_groundtruth])
        value_counts = combined_series.value_counts().to_dict()
        unique_classes = combined_series.unique().tolist()
        num_classes = len(unique_classes)
        CLASSES = num_classes
        CLASS_NAMES = [i for i in range(CLASSES)]
        
        class_weights = []
        for ind in range(num_classes):
            class_weights.append(1/value_counts[unique_classes[ind]])
        class_weights = [x/sum(class_weights) for x in class_weights]
            
        target_to_label = {} 
        for ind in range(num_classes):
            target_to_label[unique_classes[ind]] = ind
        
        # address class imbalance by repeating minority class 5 times
        max_occurrence = max(value_counts.values())
        min_occurrence = min(value_counts.values())
        if max_occurrence / min_occurrence>10:
            for k in range(len(unique_classes)):
                unique_class = unique_classes[k]
                if class_weights[target_to_label[unique_class]]<1/20:
                    train_dataset = self.dataset_dict["train"]
                    df = train_dataset.to_pandas()
                    subset_df = df[df[target] == unique_class]
                    repeated_df = pd.concat([subset_df] * 5, ignore_index=True)
                    class_weights[k] *= 6 
                    repeated_dataset = Dataset.from_pandas(repeated_df)
                    combined_dataset = concatenate_datasets([train_dataset, repeated_dataset])
                    self.dataset_dict["train"] = combined_dataset.shuffle(seed=RANDOM_SEED)
        class_weights = [x/sum(class_weights) for x in class_weights]
        CLASS_WEIGHTS = torch.tensor(np.array(class_weights), dtype=torch.float32)
        
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
        max_length=256
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
                tokenizer.save("/usr/project/xtmp/rz95/InterpretableQA-LLMTools/benchmark/chameleon/run/tools/table/temp_tokenizer.json")  # <YOUR_OWN_PATH>
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
            else:
                raise NotImplementedError()
                    
            if model == 'distilbert-base-uncased':
                print(f'Model name: {model_name} \nModel params: {model.num_parameters()}')
            else:
                print(model)
            return tokenizer, dataset, model, vocab_size

        # Map target2string
        def map_target_to_label(example):
            return {'output': target_to_label[example[target]]}
        
        def map_groundtruth_to_label(lst):
            return [target_to_label[x] for x in lst]
        
        # Create dataset
        def create_dataset(tokenizer, section=section):
            data_loaders = []
            for name in ['train', 'test']:
                # Skip the training set if we are doing only inference
                dataset = self.dataset_dict[name]
                print('*** Tokenizing...')
                # Tokenize the input
                dataset = dataset.map(
                    lambda e: tokenizer(e[section], truncation=True, padding='max_length'),
                    batched=True)
                # Set the dataset format
                if name=="test":
                    gt_list = self.test_groundtruth.to_list()
                    gt_list = map_groundtruth_to_label(gt_list)
                    dataset = dataset.map(lambda example, idx: {'output': torch.tensor(gt_list[idx])}, with_indices=True)
                dataset.set_format(type='torch', 
                    columns=['input_ids', 'attention_mask', 'output'])
                data_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=(name=='train')))
            return data_loaders

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
        def test(test_loader, model, criterion, device, name='test'):
            model.eval()
            total_loss = 0.
            total_correct = 0
            total_sample = 0
            total_confusion = np.zeros((CLASSES, CLASSES))
            predictions = []
            
            # Loop over the examples in the evaluation set
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
            
            return predictions, total_loss, float(total_correct/total_sample) * 100.


        # Training procedure (for the neural models)
        def train(data_loaders, epoch_n, model, optim, criterion, device):
            print('\n>>>Training starts...')
            # Training mode is on
            model.train()
        
            for epoch in range(epoch_n):
                total_train_loss = 0.
                # Loop over the examples in the training set.
                for i, batch in enumerate(tqdm(data_loaders[0])):
                    inputs, decisions = batch['input_ids'], batch['output']
                    inputs = inputs.to(device, non_blocking=True)
                    decisions = decisions.to(device, non_blocking=True)
                    
                    # Forward pass
                    if model_name in ['cnn', 'logistic_regression']:
                        outputs = model (input_ids=inputs)
                    else:
                        outputs = model(input_ids=inputs, labels=decisions).logits
                    loss = criterion(outputs, decisions) #outputs.logits
                    total_train_loss += loss.cpu().item()

                    # Backward pass
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

                    # Print the loss every test_every step
                    if i % test_every == 0:
                        print(f'*** Loss: {loss}')
                        print(f'*** Input: {convert_ids_to_string(tokenizer, inputs[0])}')
                        model.train()

            # Training is complete!
            print(f'\n ~ The End ~')
            
            # Final evaluation on the test set
            predictions, _, _ = test(data_loaders[1], model, criterion, device, name='test')
            
            # Additionally, print the performance of the model on the training set if we were not doing only inference
            if not test:
                test(data_loaders[0], model, criterion, device, name='train')
            return predictions

        # Evaluation procedure (for the Naive Bayes models)
        def test_naive_bayes(data_loader, model, vocab_size, name='test', pad_id=-1):
            total_loss = 0.
            total_correct = 0
            total_sample = 0
            total_confusion = np.zeros((CLASSES, CLASSES))
            predictions = []
            
            # Loop over all the examples in the evaluation set
            for i, batch in enumerate(tqdm(data_loader)):
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
            return predictions, total_loss, float(total_correct/total_sample) * 100.


        # Training procedure (for the Naive Bayes models)
        def train_naive_bayes(data_loaders, tokenizer, vocab_size, version=naive_bayes_version, alpha=1.0):
            pad_id = tokenizer.encode('[PAD]') # NEW
            print(f'Training a {version} Naive Bayes classifier (with alpha = {alpha})...')

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
            test_naive_bayes(data_loaders[0], model, vocab_size, 'training', pad_id)
            print('\n*** Accuracy on the test set ***')
            predictions, _, _ = test_naive_bayes(data_loaders[1], model, vocab_size, 'test', pad_id)
            return predictions
        
        if model_name == 'naive_bayes':
                batch_size = 1
        
        # Remove the rows where the section is None
        self.dataset_dict['train'] = self.dataset_dict['train'].filter(lambda e: e[section] is not None)
        self.dataset_dict['test'] = self.dataset_dict['test'].filter(lambda e: e[section] is not None) ###
        self.dataset_dict['train'] = self.dataset_dict['train'].map(map_target_to_label) 
    
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
        del self.dataset_dict
            
        if model_name == 'naive_bayes': 
            print('Here we are!')
            predictions = train_naive_bayes(data_loaders, tokenizer, vocab_size, naive_bayes_version, alpha_smooth_val)
        else:
            # Optimizer
            if model_name in ['logistic_regression', 'cnn']:
                optim = torch.optim.Adam(params=model.parameters())
            else:
                optim = torch.optim.AdamW(params=model.parameters(), lr=lr, eps=eps)
            # Loss function 
            criterion = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
            
            # Train and validate
            predictions = train(data_loaders, epoch_n, model, optim, criterion, device)
        predictions_to_categories = [unique_classes[x] for x in predictions]
        return {"predictions": predictions_to_categories}

if __name__ == "__main__":
    db = table_toolkits()
    # db.db_loader('hupd', '2016-2016', 'None', 'None')
    # pandas_code = "import pandas as pd\ndf['filing_month'] = df['filing_date'].apply(lambda x:x.month)\nmonth = df['filing_month'].mode()[0]"
    # print(db.pandas_interpreter(pandas_code))

    # print(db.db_loader('hupd', '2004-2006', '2007-2007', 'decision'))
    # db.textual_classifier('cnn', 'full_description', 'decision')
    # print(db.db_loader('neurips', '0-1000', '1001-3583', 'Oral'))
    # db.textual_classifier('cnn', 'Abstract', 'Oral') 
    # logistic_regression, distilbert-base-uncased, cnn, naive_bayes hupd: # title, abstract, summary, claims, background, full_description # decision    
    
