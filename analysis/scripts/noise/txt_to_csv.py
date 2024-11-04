import pandas as pd

# gcd_txt = 'gcd_all.txt' 
# kaggle_txt = 'kaggle_all.txt' 
# babylon_txt = 'babylon_all.txt' 
mixed_txt = 'combined_all.txt'

def txt_csv(input_txt_path, output_csv_path):
    data = []

    with open(input_txt_path, 'r') as file:
        for line in file:
            sub, del_, ins, hit, wer = line.strip().split()
            data.append({
                'sub': int(sub),
                'del': int(del_),
                'ins': int(ins),
                'hit': int(hit),
                'wer': float(wer)
            })

    df = pd.DataFrame(data)

    df.to_csv(output_csv_path, index=False)


# txt_csv(gcd_txt, 'gcd_all.csv')
# txt_csv(kaggle_txt, 'kaggle_all.csv')
# txt_csv(babylon_txt, 'babylon_all.csv')
txt_csv(mixed_txt, 'combined_all.csv')