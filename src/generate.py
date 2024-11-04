import warnings
import re
import time
import torch

import pandas as pd

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration
from multiprocessing import Pool, cpu_count
from datetime import datetime

warnings.filterwarnings("ignore")

class GenerateModel():
    def __init__(self, config):
        self.pubmed_datapath = config.pubmed_datapath
        self.gcd_datapath = config.gcd_datapath
        self.kaggle_datapath = config.kaggle_datapath
        self.babylon_datapath = config.babylon_datapath
        self.subset = config.subset
        self.modelname = config.fine_tuned_model
        self.generate_dir = config.generate_dir
        self.colname = config.colname
        self.max_length = config.max_length
        self.num_return_sequences = config.num_return_sequences
        self.no_repeat_ngram_size = config.no_repeat_ngram_size
        self.batch_size = config.batch_size  

        self.tokenizer = BartTokenizer.from_pretrained(self.modelname)
        self.model = BartForConditionalGeneration.from_pretrained(self.modelname) 

        # self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.modelname) 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def batch_prediction(self, batch_sentences):

        inputs = self.tokenizer(batch_sentences, 
                                padding=True, 
                                return_tensors="pt", 
                                max_length=self.max_length, 
                                truncation=True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # move input to the correct device

        with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
            outputs = self.model.generate(**inputs, 
                                    max_length=self.max_length, 
                                    num_return_sequences=self.num_return_sequences, 
                                    no_repeat_ngram_size=self.no_repeat_ngram_size, 
                                    do_sample=True, 
                                    early_stopping=True)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return decoded_outputs

    def process_dataframe(self, df):
        start_prompt = 'refine the sentence: '
        end_prompt = ' to: '
        batches = [df[i:i + self.batch_size] for i in range(0, df.shape[0], self.batch_size)]
        results = []
        for batch in batches:
            batch_sentences = batch[self.colname].tolist()
            # print("batch_sentences1: ", batch_sentences)
            batch_sentences = [start_prompt + example + end_prompt for example in batch_sentences]
            # print("batch_sentences2: ", batch_sentences)
            batch_results = self.batch_prediction(batch_sentences)
            results.extend(batch_results)

        return results

    def generate_output(self, dataset_):
        dataset_name = dataset_.split('/')[1].split('-')[0]
        # data = load_dataset(dataset_, self.subset)
        data = load_dataset(dataset_)
        if (dataset_name == "pubmed" or dataset_name == "woz"):
            # data1 = load_dataset(self.pubmed_datapath, 'babylon-01')
            # data2 = load_dataset(self.pubmed_datapath, 'kaggle-01')
            # data3 = load_dataset(self.pubmed_datapath, 'gcd-01')
            # data = concatenate_datasets([data1['test'], data2['test'], data3['test']])
            df = data.to_pandas()
            # combined_dataset = concatenate_datasets([data['train'], data['validation'], data['test']])
            # df = df[:10]
        else: 
            data = load_dataset(dataset_)
            # combined_dataset = concatenate_datasets([data['train'], data['validation'], data['test']])
            df = data['test'].to_pandas()
            # df = df[:10]
        print("df.shape: ", df.shape)
        corrected_sentences = self.process_dataframe(df)
        df['model_corrected'] = corrected_sentences
        df.to_csv(self.generate_dir+'/'+self.modelname.replace('gayanin/', '')+'-'+dataset_name+'.csv')

    def generate_output_native(self, dataset_):
        data = load_dataset(dataset_, self.subset)
        df = data['test'].to_pandas()
        corrected_sentences = self.process_dataframe(df)
        df['model_corrected'] = corrected_sentences
        df.to_csv(self.generate_dir+'/'+self.modelname.replace('gayanin/', '')+'.csv')

    def generate_all_model_outputs(self):
        # print("Start of generating PubMed model outputs: "+ str(datetime.now().strftime("%H:%M:%S")))
        # self.generate_output(self.pubmed_datapath)
        # print("End of generating PubMed model outputs: "+ str(datetime.now().strftime("%H:%M:%S")))
        print("Start of generating GCD model outputs: "+ str(datetime.now().strftime("%H:%M:%S")))
        self.generate_output(self.gcd_datapath)
        print("End of generating GCD model outputs: "+  str(datetime.now().strftime("%H:%M:%S")))
        print("Start of generating Babylon model outputs: "+ str(datetime.now().strftime("%H:%M:%S")))
        self.generate_output(self.babylon_datapath)
        print("End of generating Babylon model outputs: "+ str(datetime.now().strftime("%H:%M:%S")))
        print("Start of generating Kaggle model outputs: "+ str(datetime.now().strftime("%H:%M:%S")))
        self.generate_output(self.kaggle_datapath)
        print("End of generating Kaggle model outputs: "+ str(datetime.now().strftime("%H:%M:%S"))) 

    def generate_native_model_outputs(self):
        print("Start of generating model outputs: "+ str(datetime.now().strftime("%H:%M:%S")))
        self.generate_output_native(self.pubmed_datapath)
        print("End of generating model outputs: "+ str(datetime.now().strftime("%H:%M:%S")))
