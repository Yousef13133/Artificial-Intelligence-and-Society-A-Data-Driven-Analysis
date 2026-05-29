import os
import sys
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

if os.path.exists('data/data_final_with_derived_columns.csv'):
    DATA_FILE = 'data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs'
elif os.path.exists('../data/data_final_with_derived_columns.csv'):
    DATA_FILE = '../data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = '../outputs'
elif os.path.exists('outputs/data_final_with_derived_columns.csv'):
    DATA_FILE = 'outputs/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs'
elif os.path.exists('../outputs/data_final_with_derived_columns.csv'):
    DATA_FILE = '../outputs/data_final_with_derived_columns.csv'
    OUTPUT_DIR = '../outputs'
else:
    DATA_FILE = 'data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs'

CLEAN_DIR = os.path.join(OUTPUT_DIR, '01_data_cleaning')
os.makedirs(CLEAN_DIR, exist_ok=True)

print("DATA CLEANING")

df = pd.read_csv(DATA_FILE, encoding='utf-8')

numeric_cols = [
    'trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score',
    'tech_comfort_level', 'beneficial_score', 'ai_usage_score', 'device_hours_score'
]
categorical_cols = [
    'What is your age range?', 'What is your gender?',
    'What is your education level?', 'What is your employment status?',
    'attitude_category', 'self_awareness_category'
]

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].clip(lower=0, upper=10)
        mean_val = df[c].mean()
        df[c] = df[c].fillna(mean_val)

for c in categorical_cols:
    if c in df.columns:
        mode_val = df[c].mode(dropna=True)
        if len(mode_val) > 0:
            df[c] = df[c].fillna(mode_val.iloc[0])

df = df.drop_duplicates()

out_path = os.path.join(CLEAN_DIR, 'data_cleaned.csv')
df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"Saved cleaned data: {out_path}")
