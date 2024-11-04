import argparse
import warnings

import wandb

from datasets import load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq

from utils.functions import *
from utils.metrics import compute_metrics

warnings.filterwarnings("ignore")

class FineTuneModel():
    def __init__(self, config):
        self.datapath = config.pubmed_datapath
        self.subset = config.subset
        self.modelname = config.pretrained_modelname
        self.block_size = config.block_size       
        self.colname = config.colname

        self.output_dir = config.checkpoint_dir
        self.overwrite_output_dir = config.overwrite_output_dir
        self.evaluation_strategy = config.evaluation_strategy
        self.save_strategy = config.save_strategy
        self.num_train_epochs = config.num_train_epochs
        self.per_device_train_batch_size = config.per_device_train_batch_size
        self.per_device_eval_batch_size = config.per_device_eval_batch_size
        self.eval_steps = config.eval_steps
        self.save_steps = config.save_steps
        self.warmup_steps = config.warmup_steps
        self.logging_steps = config.logging_steps
        self.logging_dir = config.logging_dir
        self.push_to_hub = config.push_to_hub
        self.hub_model_id = config.fine_tuned_model
        self.fp16 = config.fp16
        self.load_best_model_at_end = config.load_best_model_at_end
        self.save_total_limit = config.save_total_limit

        self.tokenizer = BartTokenizer.from_pretrained(self.modelname)
        self.model = BartForConditionalGeneration.from_pretrained(self.modelname) 
        # self.prefix = "reconstruct "+self.colname+" to refs: "

        self.start_prompt = 'refine the sentence: '
        self.end_prompt = ' to: '
        
        

    def preprocess_function(self, examples):
        # inputs = [self.prefix + example for example in examples[self.colname]]
        inputs = [self.start_prompt + example + self.end_prompt for example in examples[self.colname]]
        targets = [example for example in examples["refs"]]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.block_size, truncation=True)
        return model_inputs

    def finetune_model(self):
        wandb.init(project=self.hub_model_id.replace('gayanin/', ''))

        print('self.datapath '+ self.datapath +'self.subset '+ self.subset)
        
        # data = load_dataset(self.datapath, self.subset)
        # data = load_dataset(self.datapath)

        data1 = load_dataset('gayanin/gcd-native-v8-vocab-noised')
        data2 = load_dataset('gayanin/kaggle-native-v8-vocab-noised')
        data3 = load_dataset('gayanin/babylon-native-v8-vocab-noised')

        train_data = concatenate_datasets([data1['train'], data2['train'], data3['train']])
        eval_data =  concatenate_datasets([data1['validation'], data2['validation'], data3['validation']])

        tokenized_train = train_data.map(self.preprocess_function, batched=True)
        tokenized_eval = eval_data.map(self.preprocess_function, batched=True)

        # tokenized_train = data['train'].map(self.preprocess_function, batched=True)
        # tokenized_eval = data['validation'].map(self.preprocess_function, batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=self.overwrite_output_dir,
            evaluation_strategy = self.evaluation_strategy,
            save_strategy=self.save_strategy,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            logging_dir=self.logging_dir,
            push_to_hub=self.push_to_hub, 
            hub_model_id=self.hub_model_id,
            # fp16=self.fp16,
            load_best_model_at_end=self.load_best_model_at_end,
            save_total_limit=self.save_total_limit
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer,
            # compute_metrics=compute_metrics,
            )
        
        trainer.train()

        if self.push_to_hub == True:
            trainer.push_to_hub()

        wandb.finish()