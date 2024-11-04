# Probabilities from the three datasets
prob_babylon = {'sub': 0.48911222780569513, 'del': 0.17169179229480738, 'ins': 0.3391959798994975}
prob_kaggle = {'sub': 0.6245286112477455, 'del': 0.22429906542056074, 'ins': 0.15117232333169373}
prob_gcd = {'sub': 0.6049743964886612, 'del': 0.2538405267008047, 'ins': 0.14118507681053402}

avg_prob = {op: (prob_babylon[op] + prob_kaggle[op] + prob_gcd[op]) / 3 for op in prob_babylon}

total_prob = sum(avg_prob.values())
normalized_prob = {op: p / total_prob for op, p in avg_prob.items()}

print("Averaged and Normalized Probabilities:", normalized_prob)
#Averaged and Normalized Probabilities: {'sub': 0.5728717451807007, 'del': 0.21661046147205762, 'ins': 0.21051779334724174}

# Weights (with GCD being most significant)
weights = {'babylon': 0.2, 'kaggle': 0.3, 'gcd': 0.5}

# Calculate weighted averages
weighted_avg_prob = {op: (prob_babylon[op] * weights['babylon'] + 
                          prob_kaggle[op] * weights['kaggle'] + 
                          prob_gcd[op] * weights['gcd'])
                     for op in prob_babylon}

print("Weighted Average Probabilities:", weighted_avg_prob)
# Weighted Average Probabilities: {'sub': 0.5876682271797933, 'del': 0.22854834143553204, 'ins': 0.18378343138467462}