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
        return "We have successfully loaded the {} database, including the following columns: {}.\n\
        Among these columns, the following columns have NA values: {}.".format(target_db, column_names, contains_na_names)
    
    def pandas_interpreter(self, pandas_code): 
        """
        Executes the provided Pandas code and updates the 'ans' in global_var from the loaded dataframe.
        """
        global_var = {"df": self.data.copy(), "ans": 0}
        exec(pandas_code, global_var)
        return str(global_var['ans'])

if __name__ == "__main__":
    db = table_toolkits()
    print(db.db_loader('hupd', '2017-2017'))
    pandas_code = "import pandas as pd\naccepted_patents = df[df['decision'] == 1].shape[0]\ntotal_patents = df.shape[0]\npercentage_accepted = (accepted_patents / total_patents) * 100\nans=percentage_accepted"
    print(db.pandas_interpreter(pandas_code))
