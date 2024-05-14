import os
import pandas as pd
import jsonlines
import json
import re
# Hugging Face datasets
from datasets import load_dataset, DatasetDict


class table_toolkits():
    # init
    def __init__(self):
        self.data = None
        self.dataset_dict = None
        self.path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/" #<YOUR_OWN_PATH>

    def db_loader(self, target_db, duration):
        if target_db == 'hupd':
            df = []
            hyphen_ind = duration.index("-")
            start_year = int(duration[:hyphen_ind])
            end_year = int(duration[hyphen_ind+1:])
            for sub in range(start_year, end_year+1):
                file_path = "{}/data/external_corpus/hupd/hupd_{}.csv".format(self.path, sub)
                df.append(pd.read_csv(file_path))
            self.data = pd.concat(df, ignore_index=True)
        column_names = ', '.join(self.data.columns.tolist())
        contains_na = []
        for col in self.data.columns:
            if self.data[col].isna().any():
                contains_na.append(col)
        contains_na_names = ', '.join(contains_na)
        return "We have successfully loaded the {} database, including the following columns: {}. Among these columns, the following columns have NA values: {}.".format(target_db, column_names, contains_na_names)
    
    def auto_db_loader(self, target_db, train_start='2015-01-01', train_end='2016-12-31', val_start='2017-01-01', val_end='2017-12-31'):
        if target_db == 'hupd':
            self.dataset_dict = load_dataset('HUPD/hupd',
            name='all',
            cache_dir = "/usr/project/xtmp/rz95/.cache/huggingface",
            data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather", 
            icpr_label=None,
            force_extract=True,
            train_filing_start_date=train_start, 
            train_filing_end_date=train_end, 
            val_filing_start_date=val_start,
            val_filing_end_date=val_end,
            )
    
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
    
    # split can be "None" for self.data, "train", "validation"
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
    print(db.db_loader('hupd', '2017-2017'))
    #db.auto_db_loader('hupd')
    db.target_filter("decision", "not NA")
    pandas_code = "import pandas as pd\naccepted_patents = df[df['decision'] == 'ACCEPTED'].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted"
    print(db.pandas_interpreter(pandas_code, "None"))
