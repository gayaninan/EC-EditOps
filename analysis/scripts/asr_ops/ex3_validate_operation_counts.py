
import pandas as pd

def get_sums(csv_file):
    df = pd.read_csv(csv_file)
    sum_of_sub = df['sub_count'].sum()
    sum_of_del = df['del_count'].sum()
    sum_of_ins = df['ins_count'].sum()

    print('sum_of_sub: ', sum_of_sub)
    print('sum_of_del: ', sum_of_del)
    print('sum_of_ins: ', sum_of_ins)
    print('------------------------')


# get_sums("gcd-native-v8-ops.csv")
# get_sums("gcd-native-v8-ops-train.csv")
# get_sums( "gcd-native-v8-ops-test.csv")
# get_sums( "gcd-native-v8-ops-val.csv")

get_sums("kaggle-native-v8-ops.csv") #check
get_sums("kaggle-native-v8-ops-train.csv") #check
get_sums("kaggle-native-v8-ops-test.csv")
get_sums("kaggle-native-v8-ops-val.csv")

# get_sums("babylon-native-v8-ops.csv")
# get_sums("babylon-native-v8-ops-train.csv")
# get_sums( "babylon-native-v8-ops-test.csv")
# get_sums( "babylon-native-v8-ops-val.csv")
