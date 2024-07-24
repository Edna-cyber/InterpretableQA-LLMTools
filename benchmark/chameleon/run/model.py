import os
import sys
# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import *
import jsonlines

class solver:

    def __init__(self, args):
        # arguments
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.args = args
        # external arguments
        self.api_key = openai.api_key
        self.examples, self.pids = self.load_data()
        
    def load_data(self):
        examples = ''
        pids = []
        file_path = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/questions/{}.jsonl".format(self.args.hardness) #<YOUR_OWN_PATH>
        with open(file_path, 'r') as f:
            contents = []
            for item in jsonlines.Reader(f):
                contents.append(item)
                pids.append(item['qid'])
        examples = {item['qid']: item for item in contents}
        return examples, pids
    
    