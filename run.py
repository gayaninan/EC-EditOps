import yaml
import subprocess

from utils.reproduce import *
from utils.config import *

from src.finetune import *
from src.generate import *
from src.eval import *

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='')
args = parser.parse_args()

def main():

    config_path = args.config_path
    config = load_config(config_path)
    set_seed(config.seed)

    # Fine-tune
    print("Starting the fine-tuning process...")
    fine_tuner = FineTuneModel(config) 
    fine_tuner.finetune_model()  
    print("End of fine-tuning process...")

    # Generate
    print("Generating model outputs on test data...")
    generator = GenerateModel(config) 
    if(config.gcd_datapath != ""):
        generator.generate_all_model_outputs()
    else:
        generator.generate_native_model_outputs()
    print("End of generating model outputs on test data...")

    ## Testing
    print("Evaluating the model's performance...")
    evaluator = EvalModel(config)  
    if(config.gcd_datapath != ""):
        evaluator.generate_all()
    else: 
        evaluator.generate_native()
    print("End of evaluating the model's performance...")

    print("All tasks completed.")

if __name__ == "__main__":
    main()
