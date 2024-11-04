import csv
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from jiwer import wer
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
import torch
import torch.nn as nn

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

model = BartForConditionalGeneration.from_pretrained('gayanin/custom1')
model.eval()

data = load_dataset('gayanin/kaggle-native')
eval_dataset =  data['test']
eval_loader = DataLoader(eval_dataset, batch_size=1)

# Load your tokenizer
tokenizer = BartTokenizer.from_pretrained("gayanin/custom4")

def correct_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


total_wer = 0
total_baseline_wer = 0
results = []
for batch in eval_loader:
    erroneous = batch['trans']  # Replace with the actual column name
    correct = batch['refs']  # Replace with the actual column name
    predicted = correct_text(erroneous[0])
    total_wer += wer(correct[0], predicted)
    total_baseline_wer += wer(correct[0], erroneous)
    results.append([erroneous[0], predicted, correct[0]])

# Calculate average WER
average_wer = total_wer / len(eval_loader)
average_baseline_wer = total_baseline_wer / len(eval_loader)
print(f"Average WER: {average_wer}")
print(f"Average Baseline WER: {average_baseline_wer}")

# Save results to CSV
results_df = pd.DataFrame(results, columns=["trans", "model_out", "refs"])
results_df.to_csv("evaluation_results.csv", index=False)