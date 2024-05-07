import os
import pandas as pd
import jsonlines
import json
import re

class table_toolkits():
    # init
    def __init__(self, path):
        self.data = None
        self.path = path

    def db_loader(self, target_db):
        if target_db == 'hupd':
            file_path = "{}/data/external_corpus/hupd/hupd_2015-2016.csv".format(self.path) # up for change
            self.data = pd.read_csv(file_path)
        self.data = self.data.astype(str)
        column_names = ', '.join(self.data.columns.tolist())
        return "We have successfully loaded the {} database, including the following columns: {}.".format(target_db, column_names)

if __name__ == "__main__":
    db = table_toolkits("/usr/project/xtmp/rz95/InterpretableQA-LLMTools/")
    print(db.db_loader('flights'))
    print(db.data_filter('IATA_Code_Marketing_Airline=AA, Flight_Number_Marketing_Airline=5647, Origin=BUF, Dest=PHL, FlightDate=2022-04-20'))
    print(db.get_value('DepTime'))