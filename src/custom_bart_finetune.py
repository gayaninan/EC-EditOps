import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.autograd import Variable


class CustomBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.loss_function = CustomLossWithEditDistance()

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        labels=None,
        **kwargs
    ):
        outputs = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids, 
            decoder_attention_mask=decoder_attention_mask,
            **kwargs
        )

        # Calculate custom loss if labels are provided
        if labels is not None:
            loss = self.loss_function(outputs, labels, attention_mask)
            return loss, outputs.logits  # Return loss and logits

        return outputs  # Return the usual outputs if no labels

# class CustomLoss(nn.Module):
#     def forward(self, output, labels, attention_mask):
#         logits = output.logits
#         loss_fct = nn.CrossEntropyLoss()
#         active_loss = attention_mask.view(-1) == 1
#         active_logits = logits.view(-1, logits.size(-1))
#         active_labels = torch.where(
#             active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
#         )
#         loss = loss_fct(active_logits, active_labels)
#         return loss

# class CustomLoss(nn.Module):
#     def __init__(self, word_addition_penalty=0.1):
#         super().__init__()
#         self.word_addition_penalty = word_addition_penalty

#     def forward(self, output, labels, attention_mask):
#         logits = output.logits
#         loss_fct = nn.CrossEntropyLoss()
#         active_loss = attention_mask.view(-1) == 1
#         active_logits = logits.view(-1, logits.size(-1))
#         active_labels = torch.where(
#             active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
#         )
#         loss = loss_fct(active_logits, active_labels)
        
#         # Calculate the number of added words
#         num_added_words = (active_labels != loss_fct.ignore_index).sum() - active_loss.sum()
        
#         # Add a penalty for added words
#         loss += self.word_addition_penalty * num_added_words
        
#         return loss

import torch
import torch.nn as nn

class CustomLossWithEditDistance(nn.Module):
    def __init__(self, insert_penalty=1.0, substitute_penalty=1.0, delete_penalty=1.0):
        super().__init__()
        # self.insert_penalty = 0.57
        # self.substitute_penalty = 0.22
        # self.delete_penalty = 0.21
        self.insert_penalty = 0.22
        self.substitute_penalty = 0.57
        self.delete_penalty = 0.21
        # self.insert_penalty = insert_penalty
        # self.substitute_penalty = substitute_penalty
        # self.delete_penalty = delete_penalty

    def edit_distance(self, src, tgt):
        m, n = len(src), len(tgt)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif src[i - 1] == tgt[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # Deletion
                        dp[i][j - 1] + 1,  # Insertion
                        dp[i - 1][j - 1] + 1,  # Substitution
                    )

        return dp[m][n]

    def forward(self, output, labels, attention_mask):
        logits = output.logits
        loss_fct = nn.CrossEntropyLoss()
        
        # Standard cross-entropy loss calculation
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, logits.size(-1))
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        ce_loss = loss_fct(active_logits, active_labels)
        
        # Calculate edit distance between predicted output and target label
        predicted_token_ids = torch.argmax(logits, dim=-1)  # Shape: [batch_size, sequence_length]
        predicted_tokens = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in predicted_token_ids]
        target_tokens = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in labels]

        edit_distances = [self.edit_distance(src, tgt) for src, tgt in zip(predicted_tokens, target_tokens)]
        
        # Calculate penalties based on edit distances
        insertions = sum(max(0, d - len(src)) for d, src in zip(edit_distances, predicted_tokens))
        substitutions = sum(d for d in edit_distances)
        deletions = sum(max(0, len(src) - d) for d, src in zip(edit_distances, predicted_tokens))
        
        total_tokens = active_loss.sum().item()
        normalized_insertions = insertions / total_tokens
        normalized_substitutions = substitutions / total_tokens
        normalized_deletions = deletions / total_tokens

        # Add penalties to the loss
        total_loss = ce_loss + self.insert_penalty * normalized_insertions \
                             + self.substitute_penalty * normalized_substitutions \
                             + self.delete_penalty * normalized_deletions

        return total_loss

# Preprocess function
def preprocess_function(examples):
    inputs = examples["trans"]
    targets = examples["refs"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load dataset from Hugging Face
dataset_name = "gayanin/kaggle-native"  # Replace with your dataset name
dataset = load_dataset(dataset_name)


# Define the model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./temp",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    push_to_hub = True,
    hub_model_id = "gayanin/custom4",
    logging_dir='./logs',
    logging_steps=10,
)
tokenizer.save_pretrained(training_args.output_dir)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=None 
)

# Train and save model
trainer.train()

# Push to Hub
trainer.push_to_hub()  # Replace with your desired model name
