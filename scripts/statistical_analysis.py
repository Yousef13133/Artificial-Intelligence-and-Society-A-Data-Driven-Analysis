import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

STATS_DIR = os.path.join(OUTPUT_DIR, '03_statistical_tests')
VIZ_DIR = os.path.join(OUTPUT_DIR, '02_visualizations')
os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("STATISTICAL ANALYSIS")
df = pd.read_csv(DATA_FILE, encoding='utf-8')

print("ANOVA: Trust by Age")
model_anova_age = ols('trust_index ~ C(Q("What is your age range?"))', data=df).fit()
anova_table_age = anova_lm(model_anova_age, typ=2)
print(anova_table_age)
anova_table_age.to_csv(os.path.join(STATS_DIR, 'anova_trust_by_age.csv'))

print("Post-hoc Tukey HSD: Trust by Age")
tukey_age = pairwise_tukeyhsd(endog=df['trust_index'].dropna(),
                               groups=df.loc[df['trust_index'].notna(), 'What is your age range?'],
                               alpha=0.05)
tukey_df = pd.DataFrame(data=tukey_age.summary().data[1:], columns=tukey_age.summary().data[0])
tukey_df.to_csv(os.path.join(STATS_DIR, 'posthoc_tukey_age.csv'), index=False)
print(tukey_age)

print("ANOVA: Trust by Education")
model_anova_edu = ols('trust_index ~ C(Q("What is your education level?"))', data=df).fit()
anova_table_edu = anova_lm(model_anova_edu, typ=2)
anova_table_edu.to_csv(os.path.join(STATS_DIR, 'anova_trust_by_education.csv'))
print(anova_table_edu)

print("ANOVA: Anxiety by Age")
model_anova_anx = ols('anxiety_index ~ C(Q("What is your age range?"))', data=df).fit()
anova_table_anx = anova_lm(model_anova_anx, typ=2)
anova_table_anx.to_csv(os.path.join(STATS_DIR, 'anova_anxiety_by_age.csv'))
print(anova_table_anx)

def perform_chi_square(df, var1, var2, output_name):
    contingency_table = pd.crosstab(df[var1], df[var2])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    contingency_table.to_csv(os.path.join(STATS_DIR, f'crosstab_{output_name}.csv'))
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
    plt.title(f'Contingency Table: {var1} vs {var2}')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f'heatmap_{output_name}.png'), dpi=300)
    plt.close()
    return {'chi2': chi2, 'p_value': p_val, 'dof': dof, 'cramers_v': cramers_v}

results = []
results.append({'Test': 'Age vs Attitude', **perform_chi_square(df, 'What is your age range?', 'attitude_category', 'age_attitude')})
results.append({'Test': 'Gender vs Attitude', **perform_chi_square(df, 'What is your gender?', 'attitude_category', 'gender_attitude')})
results.append({'Test': 'Education vs Attitude', **perform_chi_square(df, 'What is your education level?', 'attitude_category', 'education_attitude')})
results.append({'Test': 'Employment vs Attitude', **perform_chi_square(df, 'What is your employment status?', 'attitude_category', 'employment_attitude')})
results.append({'Test': 'Age vs Self-Awareness', **perform_chi_square(df, 'What is your age range?', 'self_awareness_category', 'age_awareness')})

pd.DataFrame(results).to_csv(os.path.join(STATS_DIR, 'chi_square_all_tests.csv'), index=False)
print("Chi-Square tests saved")

males = df[df['What is your gender?'] == 'Male']['trust_index'].dropna()
females = df[df['What is your gender?'] == 'Female']['trust_index'].dropna()
males_anx = df[df['What is your gender?'] == 'Male']['anxiety_index'].dropna()
females_anx = df[df['What is your gender?'] == 'Female']['anxiety_index'].dropna()

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(), g2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled_std

d_gender_trust = cohens_d(males, females)
d_gender_anxiety = cohens_d(males_anx, females_anx)
ss_between = anova_table_age['sum_sq'].iloc[0]
ss_total = anova_table_age['sum_sq'].sum()
eta_squared = ss_between / ss_total

effect_sizes = pd.DataFrame({
    'Comparison': ['Gender on Trust', 'Gender on Anxiety', 'Age on Trust (eta²)'],
    'Effect_Size': [d_gender_trust, d_gender_anxiety, eta_squared]
})
effect_sizes.to_csv(os.path.join(STATS_DIR, 'effect_sizes.csv'), index=False)
print("Effect sizes saved")
