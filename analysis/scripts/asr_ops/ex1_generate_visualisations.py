import jiwer
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

# refs = ["short one here", "quite a bit of longer sentence", "cat sat on a mat"]
# trans = ["shoe order one", "quite bit of an even longest sentence here", "cat at on mat"]


def generate_visual(in_path, out_path):
    # df = pd.read_csv(in_path)

    data = load_dataset(in_path)
    print(data)

    # data_ = concatenate_datasets([ data['train'], data['test'],data['validation']])
    # df = data_.to_pandas()

    df = data['validation'].to_pandas()
    df = df.dropna()

    print(df.shape)
    # Kaggle
    
    # df.drop(index=2908, inplace=True)
    print(df.shape)

    refs = df['refs'].tolist()
    trans = df['trans'].tolist()

    out = jiwer.process_words(refs,trans)

    visual = jiwer.visualize_alignment(out)

    with open(out_path, 'w') as file:
        file.write(visual)

# generate_visual('gayanin/gcd-native-v8', 'gcd-native-v8-ops-val.txt')
generate_visual('gayanin/kaggle-native-v8', 'kaggle-native-v8-ops-val.txt')
# generate_visual('gayanin/babylon-native-v8', 'babylon-native-v8-ops-val.txt')