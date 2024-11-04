#!/bin/bash

python scripts/combined.py
python scripts/calculate_wer.py
python scripts/txt_to_df.py
python scripts/distributions.py