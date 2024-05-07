import os
import pandas as pd
import jsonlines
import json
import re

class table_toolkits():
    # init
    def __init__(self):
        self.data = None
        self.path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/" #<YOUR_OWN_PATH>

    def db_loader(self, target_db, range):
        if target_db == 'hupd':
            df = []
            hyphen_ind = range.index("-")
            start_year = range[:hyphen_ind]
            end_year = range[hyphen_ind+1:]
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
        return "We have successfully loaded the {} database, including the following columns: {}. \n \
        Among these columns, the following columns have NA values: {}.".format(target_db, column_names, contains_na_names)
    
    def pandas_interpreter(self, pandas_code): #change
        """
        Returns the path to the python interpreter.
        """
        global_var = {"ans": 0}
        exec(pandas_code, global_var)
        # print(str(global_var))
        return str(global_var['ans'])
    

if __name__ == "__main__":
    db = table_toolkits("/usr/project/xtmp/rz95/InterpretableQA-LLMTools/") #<YOUR_OWN_PATH>
    print(db.db_loader('hupd', '2017-2017'))
    print(db.pandas_interpreter())
    print()
    