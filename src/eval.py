import argparse
import warnings
import re

import torch
from torch.nn.functional import cosine_similarity
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration

warnings.filterwarnings("ignore")

class EvalModel():
    def __init__(self, config):
        self.pubmed_datapath = config.pubmed_datapath
        self.subset = config.subset
        self.modelname = config.fine_tuned_model
        self.generate_dir = config.generate_dir
        self.output_dir = config.eval_dir
        self.baseline = config.baseline
        self.colname = config.colname

        self.tokenizer = BartTokenizer.from_pretrained(self.modelname)
        self.model = BartForConditionalGeneration.from_pretrained(self.modelname) 

        # self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.modelname) 

        # Load metric
        self.metric = load_metric('wer')
        
    def calculate_wer(self, col1, col2, path_):
        df = pd.read_csv(path_)
        trans_batch = df[col1].tolist()
        refs_batch = df[col2].tolist()   
        score = self.metric.compute(predictions=refs_batch, references=trans_batch)  
        return score
        
    def generate_results(self,path_):  
        re_path = re.sub('model_outputs/|/|.csv', '', path_)
        baseline_col1 = 'refs'
        baseline_col2 = self.colname

        col1 = 'refs'
        col2 = 'model_corrected'

        out_path = self.output_dir+'/'+re_path+'.txt'
    
        if self.baseline is True:
            baseline_wer = self.calculate_wer(baseline_col1, baseline_col2, path_)

        wer = self.calculate_wer(col1, col2, path_)

        with open(out_path,'w') as text_file:
            text_file.write(re_path+'\n')
            if self.baseline is True:
                text_file.write('baseline wer: %s' % baseline_wer+'\n')
            text_file.write('wer: %s' % wer+'\n')

    def generate_all(self):
        self.generate_results(self.generate_dir+'/'+self.modelname.replace('gayanin/', '')+'-kaggle'+'.csv')
        self.generate_results(self.generate_dir+'/'+self.modelname.replace('gayanin/', '')+'-babylon'+'.csv')
        self.generate_results(self.generate_dir+'/'+self.modelname.replace('gayanin/', '')+'-gcd'+'.csv')
        # self.generate_results(self.generate_dir+'/'+self.modelname.replace('gayanin/', '')+'-pubmed'+'.csv')
        # self.generate_results(self.generate_dir+'/'+self.modelname.replace('gayanin/', '')+'-woz'+'.csv')

    def generate_native(self):
        self.generate_results(self.generate_dir+'/'+self.modelname.replace('gayanin/', '')+'.csv')