# Edit Distance Operations-Based Error Correction for Post-ASR Transcriptions

## Experiment Summary

This work presents a comprehensive evaluation of the BART-base model's performance in correcting post-ASR transcription errors using various noising strategies based on three primary edit distance operations: insertions, deletions, and substitutions. The results reveal key insights into the model's strengths and limitations:

- **Synthetic vs. Real-World Transcriptions**: While synthetic noising strategies improved model performance on artificially noised datasets, they struggled to generalize effectively to real-world transcription data. This discrepancy, especially evident in complex datasets like GCD V2, indicates that synthetic noise does not fully emulate the intricacies of actual transcription errors.
  
- **Domain-Specific Challenges**: Vocabulary-specific noising demonstrated potential for handling domain-specific terminology, which is critical in specialized fields like healthcare. However, it also introduced challenges, such as over-correction and hallucination, particularly when applied to complex datasets.

- **Implications for Future Work**: These findings highlight the need for more sophisticated noising strategies that more accurately reflect real-world transcription errors. Future work will focus on refining noise modeling techniques for domain-specific language and enhancing fine-tuning processes to minimize hallucinations and over-correction, ultimately contributing to more robust ASR error correction models in clinical and specialized settings.

## Experiment Setup

To explore edit operations-based error correction, we used [BART-base](https://aclanthology.org/2020.acl-main.703) as our pre-trained language model. The model was fine-tuned using [Hugging Face](https://huggingface.co) and [PyTorch](https://pytorch.org) libraries with default hyperparameters.

## Datasets

- **Training Data**: Consists of two open-source datasets and our internal GCD V2 dataset, with synthetic errors introduced through edit distance operations.
- **Evaluation Data**: Data was split into training and validation sets with varying noise levels to assess model robustness.

The datasets are currently hosted on the [Hugging Face hub](https://huggingface.co), and are also available in our HF collection [EC-EditOps](https://huggingface.co/collections/gayanin/ec-editops-652ea8f41ad13fee8c1f9da3).

## Running the Experiment

To install the necessary dependencies:
```bash
pip install -r requirements.txt
```

To run the experiment, use the `run.sh` script with the desired configuration file. Configuration files are available in `config/`.

### Command to Execute:
```bash
bash run.sh

