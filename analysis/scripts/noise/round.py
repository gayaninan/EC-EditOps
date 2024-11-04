# Original probabilities
# sub = 0.47417582417582416
# del_ = 0.15645604395604396  # Using del_ because 'del' is a keyword in Python
# ins = 0.3693681318681319

# sub = 0.6454197143852316 
# del_ = 0.19522814350400558
# ins = 0.1593521421107628

sub = 0.6142208774583964 
del_ = 0.24130105900151286 
ins = 0.14447806354009077

# Rounding the values to 2 decimal points
sub_rounded = round(sub, 2)
del_rounded = round(del_, 2)
ins_rounded = round(ins, 2)

# Adjusting the rounding to ensure the sum is 1
# This is a simplistic approach to handle rounding errors; for more precise applications, consider a more sophisticated method.
total = sub_rounded + del_rounded + ins_rounded
if total != 1.0:
    # Find which value to adjust to make the sum equal to 1
    # This is a simple heuristic that adds or subtracts the small discrepancy to the largest value
    # It's not the most accurate method but serves for illustrative purposes
    discrepancies = 1.0 - total
    max_val = max(sub_rounded, del_rounded, ins_rounded)
    if max_val == sub_rounded:
        sub_rounded += discrepancies
    elif max_val == del_rounded:
        del_rounded += discrepancies
    else:
        ins_rounded += discrepancies

print(sub_rounded, del_rounded, ins_rounded, sub_rounded + del_rounded + ins_rounded)

