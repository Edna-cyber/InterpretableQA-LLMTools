import os
import numpy as np
import pandas as pd

neurips_dir = "/usr/project/xtmp/rz95/InterpretableQA-LLMTools/data/external_corpus/neurips/"
df = pd.read_csv(os.path.join(neurips_dir, "NeurIPS_2023_Papers.csv"))
df['Authors'] = df['Authors'].str.split(' · ')
df['Authors_Num'] = df['Authors'].apply(len)

# Save the combined DataFrame to a CSV file
output_file = os.path.join(neurips_dir, "NeurIPS_2023_Newest_Papers.csv")
# print(df.dtypes)
# print(len(df))
df.to_csv(output_file, index=False)
