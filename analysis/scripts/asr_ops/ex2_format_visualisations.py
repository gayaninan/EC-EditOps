import pandas as pd

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    sentence_num = 0
    print(sentence_num)
    for line in lines:
        print(line)
        line = line.strip()
        if line.startswith('sentence'):
            sentence_num += 1
            ref = ''
            hyp = ''
            ops = ''
        elif line.startswith('REF:'):
            ref = line.replace('REF:', '').strip()
        elif line.startswith('HYP:'):
            hyp = line.replace('HYP:', '').strip()
        elif line.startswith('OPS:'):
            ops = line.replace('OPS:', '').strip()
            ops_list = ops.split()
            sub = []
            del_ops = []
            ins = []
            nochange = 0
            for i, op in enumerate(ops_list):
                if op == 'S':
                    sub.append(f"{ref.split()[i]} -> {hyp.split()[i]}")
                elif op == 'D':
                    del_ops.append(ref.split()[i])
                elif op == 'I':
                    ins.append(hyp.split()[i])
                elif op == 'N':
                    nochange += 1
            data.append({
                'sentence_num': sentence_num,
                'ref': ref,
                'hyp': hyp,
                'sub': '; '.join(sub),
                'sub_count': len(sub),
                'del': '; '.join(del_ops),
                'del_count': len(del_ops),
                'ins': '; '.join(ins),
                'ins_count': len(ins),
                'nochange': nochange
            })
    
    return data

def save_to_csv(data, csv_file_path):
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)

def save_counts(in_path, out_path):
    data = process_file(in_path)
    save_to_csv(data, out_path)

# save_counts("gcd-native-v8-ops.txt", "gcd-native-v8-ops.csv")
# save_counts("gcd-native-v8-ops-train.txt", "gcd-native-v8-ops-train.csv")
# save_counts("gcd-native-v8-ops-test.txt", "gcd-native-v8-ops-test.csv")
# save_counts("gcd-native-v8-ops-val.txt", "gcd-native-v8-ops-val.csv")

save_counts("kaggle-native-v8-ops.txt", "kaggle-native-v8-ops.csv")
save_counts("kaggle-native-v8-ops-train.txt", "kaggle-native-v8-ops-train.csv")
save_counts("kaggle-native-v8-ops-test.txt", "kaggle-native-v8-ops-test.csv")
save_counts("kaggle-native-v8-ops-val.txt", "kaggle-native-v8-ops-val.csv")

# save_counts("babylon-native-v8-ops.txt", "babylon-native-v8-ops.csv")
# save_counts("babylon-native-v8-ops-train.txt", "babylon-native-v8-ops-train.csv")
# save_counts("babylon-native-v8-ops-test.txt", "babylon-native-v8-ops-test.csv")
# save_counts("babylon-native-v8-ops-val.txt", "babylon-native-v8-ops-val.csv")