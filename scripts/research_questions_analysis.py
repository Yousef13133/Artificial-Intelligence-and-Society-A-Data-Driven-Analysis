# الغرض العام من هذا السكربت:
# تحليل أسئلة بحثية مرتبطة بالثقة والقلق والمعرفة وسلوكيات المستخدم،
# مع إعداد ملفات جاهزة لـ Power BI، وتنفيذ نماذج انحدار وANOVA،
# وحساب فروقات المعرفة واشتقاق فئات للتجميع البصري والتحليل الإحصائي.
# التعليقات التالية تشرح الخطوات والتعليمات بالتفصيل لضمان فهم شامل.
# افتراضات البيانات:
# - الأعمدة الرقمية (0-10): trust_index, anxiety_index, knowledge_score, ai_literacy_score,
#   tech_comfort_level, beneficial_score, ai_usage_score, device_hours_score.
# - الأعمدة الفئوية: What is your age range?, What is your gender?, What is your education level?,
#   What is your employment status?, attitude_category, self_awareness_category.
# - عند وجود قيم مفقودة، يتم الإسقاط أو التجهيز بحسب نوع التحليل لضمان الاتساق.
# المخرجات:
# - ملفات CSV داخل outputs/04_research_questions جاهزة لـ Power BI لتغذية مصفوفات ومخططات.
# كيفية التشغيل:
# - شغّل الملف؛ سيتعرف تلقائياً على مسار البيانات ويحفظ النتائج منظمة داخل مجلد المخرجات المحدد.

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.mediation import Mediation
import warnings
warnings.filterwarnings('ignore')
import os
import sys

# Fix encoding for Windows Arabic systems
# إصلاح الترميز على أنظمة ويندوز لضمان عرض العربية بشكل صحيح في إخراج الطرفية
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configuration
# Smart path detection - works whether run from main folder or scripts folder
# كشف المسار الذكي لتحديد موقع ملف البيانات ومجلد الإخراج حسب مكان التشغيل
if os.path.exists('data/data_final_with_derived_columns.csv'):
    DATA_FILE = 'data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs/04_research_questions'
elif os.path.exists('../data/data_final_with_derived_columns.csv'):
    DATA_FILE = '../data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = '../outputs/04_research_questions'
elif os.path.exists('outputs/data_final_with_derived_columns.csv'):
    DATA_FILE = 'outputs/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs/04_research_questions'
elif os.path.exists('../outputs/data_final_with_derived_columns.csv'):
    DATA_FILE = '../outputs/data_final_with_derived_columns.csv'
    OUTPUT_DIR = '../outputs/04_research_questions'
else:
    DATA_FILE = 'data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs/04_research_questions'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("RESEARCH QUESTIONS ANALYSIS FOR POWER BI")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv(DATA_FILE, encoding='utf-8')
print(f"✓ Loaded {len(df)} rows")


# ============================================================================
# MAIN RESEARCH QUESTION: Psychological & Behavioral Variables → Trust
# Differences by Age & Gender
# ============================================================================
print("\n[2/5] Analyzing Main Research Question...")
# بدء تحليل السؤال البحثي الرئيسي والمتغيرات المؤثرة على الثقة بحسب العمر والجنس
print("Variables influencing trust, by age and gender\n")

# Prepare comprehensive dataset for Power BI
# تجهيز إطار بيانات شامل يحتوي على المتغيرات الأساسية والمحسوبة لتسهيل التحليل في Power BI
main_analysis = df[[
    'trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score',
    'tech_comfort_level', 'beneficial_score', 'ai_usage_score',
    'device_hours_score', 'future_intent_score',
    'What is your age range?', 'What is your gender?',
    'What is your education level?', 'What is your employment status?',
    'attitude_category', 'self_awareness_category',
    'trust_anxiety_gap', 'literacy_perception_gap', 'knowledge_confidence_gap'
]].copy()

# Add derived grouping variables for Power BI
# إضافة متغيرات فئوية مشتقة (مستويات الثقة/القلق/المعرفة) لاستخدامها في المرئيات
main_analysis['Age_Group'] = main_analysis['What is your age range?']
main_analysis['Gender'] = main_analysis['What is your gender?']
main_analysis['Trust_Level'] = pd.cut(main_analysis['trust_index'], 
                                       bins=[0, 3.33, 6.67, 10], 
                                       labels=['Low Trust', 'Medium Trust', 'High Trust'])
main_analysis['Anxiety_Level'] = pd.cut(main_analysis['anxiety_index'], 
                                         bins=[0, 3.33, 6.67, 10], 
                                         labels=['Low Anxiety', 'Medium Anxiety', 'High Anxiety'])
main_analysis['Knowledge_Level'] = pd.cut(main_analysis['knowledge_score'], 
                                           bins=[0, 3.33, 6.67, 10], 
                                           labels=['Low Knowledge', 'Medium Knowledge', 'High Knowledge'])

# Save main dataset for Power BI
# حفظ مجموعة البيانات المجهزة بصيغة CSV للاستيراد المباشر في Power BI
main_analysis.to_csv(os.path.join(OUTPUT_DIR, 'main_trust_analysis.csv'), index=False)
print("✓ Main analysis dataset saved: main_trust_analysis.csv")

# Multiple Regression: Trust predicted by psychological/behavioral variables
print("\n--- Multiple Regression Analysis ---")
regression_data = df[['trust_index', 'knowledge_score', 'ai_literacy_score', 
                       'tech_comfort_level', 'beneficial_score', 'ai_usage_score',
                       'anxiety_index', 'device_hours_score']].dropna()

X = regression_data.drop('trust_index', axis=1)
X = sm.add_constant(X)
y = regression_data['trust_index']

model = sm.OLS(y, X).fit()
# بناء نموذج الانحدار الخطي العادي (OLS) لقياس تأثير المتغيرات المستقلة على الثقة
print(model.summary())

# Extract regression results for Power BI
regression_results = pd.DataFrame({
    'Variable': model.params.index,
    'Coefficient': model.params.values,
    'Std_Error': model.bse.values,
    'T_Statistic': model.tvalues.values,
    'P_Value': model.pvalues.values,
    'CI_Lower': model.conf_int()[0].values,
    'CI_Upper': model.conf_int()[1].values,
    'Significant': ['Yes' if p < 0.05 else 'No' for p in model.pvalues.values]
})
regression_results['Abs_Coefficient'] = regression_results['Coefficient'].abs()
regression_results = regression_results.sort_values('Abs_Coefficient', ascending=False)
regression_results.to_csv(os.path.join(OUTPUT_DIR, 'trust_predictors_regression.csv'), index=False)
print(f"✓ Regression results saved")
print(f"  R² = {model.rsquared:.4f}")
print(f"  Adj R² = {model.rsquared_adj:.4f}")

# Trust by Age and Gender (for Power BI matrix visual)
trust_by_demographics = df.groupby(['What is your age range?', 'What is your gender?']).agg({
    'trust_index': ['mean', 'std', 'count'],
    'anxiety_index': ['mean', 'std'],
    'knowledge_score': ['mean'],
    'ai_literacy_score': ['mean']
}).reset_index()
trust_by_demographics.columns = ['Age_Range', 'Gender', 
                                  'Trust_Mean', 'Trust_Std', 'Count',
                                  'Anxiety_Mean', 'Anxiety_Std',
                                  'Knowledge_Mean', 'Literacy_Mean']
trust_by_demographics.to_csv(os.path.join(OUTPUT_DIR, 'trust_by_age_gender.csv'), index=False)
print("✓ Trust by age and gender saved")

# Two-way ANOVA: Trust ~ Age + Gender + Age*Gender
print("\n--- Two-Way ANOVA: Trust by Age and Gender ---")
formula = 'trust_index ~ C(Q("What is your age range?")) + C(Q("What is your gender?")) + C(Q("What is your age range?")):C(Q("What is your gender?"))'
model_anova = ols(formula, data=df).fit()
anova_table = anova_lm(model_anova, typ=2)
# استخدام ANOVA من النوع 2 (Type II) الذي يقيم كل عامل بعد التحكم في العوامل الأخرى
print(anova_table)
anova_table.to_csv(os.path.join(OUTPUT_DIR, 'anova_trust_age_gender.csv'))
print("✓ Two-way ANOVA results saved")


# ============================================================================
# SUB-QUESTION 1: Knowledge Discrepancies by Group
# ============================================================================
print("\n[3/5] Analyzing Sub-Question 1...")
print("Knowledge discrepancies (perceived vs actual) by demographic groups\n")

# Calculate knowledge gap for each individual
df['knowledge_gap'] = df['knowledge_score'] - df['ai_literacy_score']
df['gap_category'] = pd.cut(df['knowledge_gap'], 
                             bins=[-np.inf, -2, 2, np.inf],
                             labels=['Underconfident', 'Realistic', 'Overconfident'])

# Knowledge gaps by Age
gaps_by_age = df.groupby('What is your age range?').agg({
    'knowledge_score': 'mean',
    'ai_literacy_score': 'mean',
    'knowledge_gap': 'mean',
    'literacy_perception_gap': 'mean'
}).reset_index()
gaps_by_age.columns = ['Age_Range', 'Perceived_Knowledge', 'Actual_Literacy', 
                        'Knowledge_Gap', 'Perception_Gap']
gaps_by_age['Gap_Direction'] = gaps_by_age['Knowledge_Gap'].apply(
    lambda x: 'Overconfident' if x > 2 else 'Underconfident' if x < -2 else 'Realistic'
)
gaps_by_age.to_csv(os.path.join(OUTPUT_DIR, 'RQ1_knowledge_gaps_by_age.csv'), index=False)
print("✓ Knowledge gaps by age saved")

# Knowledge gaps by Gender
gaps_by_gender = df.groupby('What is your gender?').agg({
    'knowledge_score': 'mean',
    'ai_literacy_score': 'mean',
    'knowledge_gap': 'mean',
    'literacy_perception_gap': 'mean'
}).reset_index()
gaps_by_gender.columns = ['Gender', 'Perceived_Knowledge', 'Actual_Literacy', 
                           'Knowledge_Gap', 'Perception_Gap']
gaps_by_gender.to_csv(os.path.join(OUTPUT_DIR, 'RQ1_knowledge_gaps_by_gender.csv'), index=False)
print("✓ Knowledge gaps by gender saved")

# Knowledge gaps by Education
gaps_by_education = df.groupby('What is your education level?').agg({
    'knowledge_score': 'mean',
    'ai_literacy_score': 'mean',
    'knowledge_gap': 'mean',
    'literacy_perception_gap': 'mean'
}).reset_index()
gaps_by_education.columns = ['Education_Level', 'Perceived_Knowledge', 'Actual_Literacy', 
                              'Knowledge_Gap', 'Perception_Gap']
gaps_by_education = gaps_by_education.sort_values('Knowledge_Gap', ascending=False)
gaps_by_education.to_csv(os.path.join(OUTPUT_DIR, 'RQ1_knowledge_gaps_by_education.csv'), index=False)
print("✓ Knowledge gaps by education saved")

# Gap category distribution by demographics
gap_distribution = pd.crosstab(
    [df['What is your age range?'], df['What is your gender?']], 
    df['gap_category'], 
    normalize='index'
) * 100
gap_distribution = gap_distribution.reset_index()
gap_distribution.to_csv(os.path.join(OUTPUT_DIR, 'RQ1_gap_distribution_by_demographics.csv'), index=False)
print("✓ Gap distribution by demographics saved")

# Statistical test: ANOVA for knowledge gap by age
# اختبار ANOVA لفحص فروقات الفجوة المعرفية بين الفئات العمرية المختلفة
print("\n--- ANOVA: Knowledge Gap by Age ---")
age_groups_gap = [group['knowledge_gap'].dropna() for name, group in df.groupby('What is your age range?')]
# إنشاء قائمة من السلاسل لكل فئة عمرية تحتوي قيم الفجوة المعرفية بدون القيم المفقودة لاستخدامها في ANOVA
f_stat, p_val = f_oneway(*age_groups_gap)
print(f"  F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}")

anova_gap_result = pd.DataFrame({
    'Test': ['Knowledge Gap by Age'],
    'F_Statistic': [f_stat],
    'P_Value': [p_val],
    'Significant': ['Yes' if p_val < 0.05 else 'No']
})
anova_gap_result.to_csv(os.path.join(OUTPUT_DIR, 'RQ1_anova_knowledge_gap.csv'), index=False)
print("✓ ANOVA results saved")

# Detailed breakdown for Power BI
# إنشاء بيانات تفصيلية على مستوى الأفراد لتمكين تحليلات دقيقة في Power BI
individual_gaps = df[['What is your age range?', 'What is your gender?', 
                      'What is your education level?', 'What is your employment status?',
                      'knowledge_score', 'ai_literacy_score', 'knowledge_gap', 
                      'gap_category', 'self_awareness_category']].copy()
individual_gaps.to_csv(os.path.join(OUTPUT_DIR, 'RQ1_individual_knowledge_gaps.csv'), index=False)
print("✓ Individual-level gap data saved")


# ============================================================================
# SUB-QUESTION 2: Actual Knowledge vs Age - Which impacts Trust/Anxiety more?
# ============================================================================
print("\n[4/5] Analyzing Sub-Question 2...")
# مقارنة تأثير المعرفة الفعلية مقابل العمر على الثقة والقلق
print("Comparing impact of actual knowledge vs age on trust and anxiety\n")

# Prepare data for comparison
# تجهيز بيانات المقارنة مع أعمدة مطلوبة وإزالة القيم المفقودة
comparison_data = df[['trust_index', 'anxiety_index', 'ai_literacy_score', 
                      'What is your age range?']].dropna()

# Encode age as numeric for regression
# ترميز الفئة العمرية إلى قيم رقمية لاستخدامها في نماذج الانحدار بسهولة
age_mapping = {
    'Under 18 years old': 1,
    '18-24': 2,
    '25-34': 3,
    '35-44': 4,
    '45 years and older': 5
}
comparison_data['age_numeric'] = comparison_data['What is your age range?'].map(age_mapping)

# Model 1: Trust predicted by Actual Knowledge only
print("\n--- Model 1: Trust ~ Actual Knowledge ---")
X1 = sm.add_constant(comparison_data[['ai_literacy_score']])
y_trust = comparison_data['trust_index']
model1_trust = sm.OLS(y_trust, X1).fit()
print(f"  R² = {model1_trust.rsquared:.4f}")
print(f"  Coefficient (Knowledge): {model1_trust.params['ai_literacy_score']:.4f}")
print(f"  P-value: {model1_trust.pvalues['ai_literacy_score']:.4e}")

# Model 2: Trust predicted by Age only
print("\n--- Model 2: Trust ~ Age ---")
X2 = sm.add_constant(comparison_data[['age_numeric']])
model2_trust = sm.OLS(y_trust, X2).fit()
print(f"  R² = {model2_trust.rsquared:.4f}")
print(f"  Coefficient (Age): {model2_trust.params['age_numeric']:.4f}")
print(f"  P-value: {model2_trust.pvalues['age_numeric']:.4e}")

# Model 3: Trust predicted by Both
print("\n--- Model 3: Trust ~ Knowledge + Age ---")
X3 = sm.add_constant(comparison_data[['ai_literacy_score', 'age_numeric']])
model3_trust = sm.OLS(y_trust, X3).fit()
print(f"  R² = {model3_trust.rsquared:.4f}")
print(f"  Coefficient (Knowledge): {model3_trust.params['ai_literacy_score']:.4f}, p={model3_trust.pvalues['ai_literacy_score']:.4e}")
print(f"  Coefficient (Age): {model3_trust.params['age_numeric']:.4f}, p={model3_trust.pvalues['age_numeric']:.4e}")

# Compare models for TRUST
trust_comparison = pd.DataFrame({
    'Model': ['Knowledge Only', 'Age Only', 'Knowledge + Age'],
    'R_Squared': [model1_trust.rsquared, model2_trust.rsquared, model3_trust.rsquared],
    'Adj_R_Squared': [model1_trust.rsquared_adj, model2_trust.rsquared_adj, model3_trust.rsquared_adj],
    'Knowledge_Coef': [model1_trust.params['ai_literacy_score'], np.nan, model3_trust.params['ai_literacy_score']],
    'Age_Coef': [np.nan, model2_trust.params['age_numeric'], model3_trust.params['age_numeric']],
    'Knowledge_PValue': [model1_trust.pvalues['ai_literacy_score'], np.nan, model3_trust.pvalues['ai_literacy_score']],
    'Age_PValue': [np.nan, model2_trust.pvalues['age_numeric'], model3_trust.pvalues['age_numeric']]
})
trust_comparison.to_csv(os.path.join(OUTPUT_DIR, 'RQ2_trust_model_comparison.csv'), index=False)
print("✓ Trust model comparison saved")

# Same analysis for ANXIETY
print("\n--- Anxiety Models ---")
y_anxiety = comparison_data['anxiety_index']

model1_anx = sm.OLS(y_anxiety, X1).fit()
model2_anx = sm.OLS(y_anxiety, X2).fit()
model3_anx = sm.OLS(y_anxiety, X3).fit()

anxiety_comparison = pd.DataFrame({
    'Model': ['Knowledge Only', 'Age Only', 'Knowledge + Age'],
    'R_Squared': [model1_anx.rsquared, model2_anx.rsquared, model3_anx.rsquared],
    'Adj_R_Squared': [model1_anx.rsquared_adj, model2_anx.rsquared_adj, model3_anx.rsquared_adj],
    'Knowledge_Coef': [model1_anx.params['ai_literacy_score'], np.nan, model3_anx.params['ai_literacy_score']],
    'Age_Coef': [np.nan, model2_anx.params['age_numeric'], model3_anx.params['age_numeric']],
    'Knowledge_PValue': [model1_anx.pvalues['ai_literacy_score'], np.nan, model3_anx.pvalues['ai_literacy_score']],
    'Age_PValue': [np.nan, model2_anx.pvalues['age_numeric'], model3_anx.pvalues['age_numeric']]
})
anxiety_comparison.to_csv(os.path.join(OUTPUT_DIR, 'RQ2_anxiety_model_comparison.csv'), index=False)
print("✓ Anxiety model comparison saved")

# Calculate standardized coefficients (Beta) for direct comparison
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_standardized = scaler.fit_transform(comparison_data[['ai_literacy_score', 'age_numeric']])
X_std = sm.add_constant(X_standardized)

model_std_trust = sm.OLS(y_trust, X_std).fit()
model_std_anxiety = sm.OLS(y_anxiety, X_std).fit()

standardized_effects = pd.DataFrame({
    'Variable': ['Actual Knowledge', 'Age'],
    'Beta_Trust': [model_std_trust.params[1], model_std_trust.params[2]],
    'Beta_Anxiety': [model_std_anxiety.params[1], model_std_anxiety.params[2]],
    'Abs_Beta_Trust': [abs(model_std_trust.params[1]), abs(model_std_trust.params[2])],
    'Abs_Beta_Anxiety': [abs(model_std_anxiety.params[1]), abs(model_std_anxiety.params[2])]
})
standardized_effects['Stronger_Predictor_Trust'] = standardized_effects['Abs_Beta_Trust'].apply(
    lambda x: 'Knowledge' if x == standardized_effects['Abs_Beta_Trust'].max() else 'Age'
)
standardized_effects['Stronger_Predictor_Anxiety'] = standardized_effects['Abs_Beta_Anxiety'].apply(
    lambda x: 'Knowledge' if x == standardized_effects['Abs_Beta_Anxiety'].max() else 'Age'
)
standardized_effects.to_csv(os.path.join(OUTPUT_DIR, 'RQ2_standardized_effects.csv'), index=False)
print("✓ Standardized effects (Beta coefficients) saved")
print(f"\n  Stronger predictor of TRUST: {standardized_effects.loc[standardized_effects['Abs_Beta_Trust'].idxmax(), 'Variable']}")
print(f"  Stronger predictor of ANXIETY: {standardized_effects.loc[standardized_effects['Abs_Beta_Anxiety'].idxmax(), 'Variable']}")


# ============================================================================
# SUB-QUESTION 3: Age → Anxiety, with Knowledge as Mediator
# ============================================================================
print("\n[5/5] Analyzing Sub-Question 3...")
print("Mediation analysis: Age → Anxiety, mediated by Knowledge\n")

# Prepare data for mediation analysis
mediation_data = df[['anxiety_index', 'ai_literacy_score', 'What is your age range?']].dropna()
mediation_data['age_numeric'] = mediation_data['What is your age range?'].map(age_mapping)
mediation_data = mediation_data.dropna()

# Step 1: Total effect (Age → Anxiety)
print("\n--- Step 1: Total Effect (c path) ---")
X_total = sm.add_constant(mediation_data[['age_numeric']])
y_anxiety_med = mediation_data['anxiety_index']
model_total = sm.OLS(y_anxiety_med, X_total).fit()
total_effect = model_total.params['age_numeric']
print(f"  Total effect (c): {total_effect:.4f}")
print(f"  P-value: {model_total.pvalues['age_numeric']:.4e}")
print(f"  R²: {model_total.rsquared:.4f}")

# Step 2: Age → Knowledge (a path)
print("\n--- Step 2: Age → Knowledge (a path) ---")
y_knowledge = mediation_data['ai_literacy_score']
model_a = sm.OLS(y_knowledge, X_total).fit()
a_path = model_a.params['age_numeric']
print(f"  a path coefficient: {a_path:.4f}")
print(f"  P-value: {model_a.pvalues['age_numeric']:.4e}")
print(f"  R²: {model_a.rsquared:.4f}")

# Step 3: Age + Knowledge → Anxiety (b and c' paths)
print("\n--- Step 3: Age + Knowledge → Anxiety (b and c' paths) ---")
X_mediated = sm.add_constant(mediation_data[['age_numeric', 'ai_literacy_score']])
model_mediated = sm.OLS(y_anxiety_med, X_mediated).fit()
b_path = model_mediated.params['ai_literacy_score']
c_prime = model_mediated.params['age_numeric']
print(f"  b path (Knowledge → Anxiety): {b_path:.4f}")
print(f"  P-value: {model_mediated.pvalues['ai_literacy_score']:.4e}")
print(f"  c' path (Direct effect, Age → Anxiety): {c_prime:.4f}")
print(f"  P-value: {model_mediated.pvalues['age_numeric']:.4e}")
print(f"  R²: {model_mediated.rsquared:.4f}")

# Calculate indirect effect and mediation percentage
indirect_effect = a_path * b_path
mediation_percentage = (indirect_effect / total_effect) * 100 if total_effect != 0 else 0

print(f"\n--- Mediation Results ---")
print(f"  Indirect effect (a × b): {indirect_effect:.4f}")
print(f"  Direct effect (c'): {c_prime:.4f}")
print(f"  Total effect (c): {total_effect:.4f}")
print(f"  Mediation percentage: {mediation_percentage:.2f}%")

# Determine mediation type
if abs(c_prime) < 0.001 or model_mediated.pvalues['age_numeric'] > 0.05:
    mediation_type = "Full Mediation"
elif abs(c_prime) < abs(total_effect):
    mediation_type = "Partial Mediation"
else:
    mediation_type = "No Mediation"
print(f"  Mediation type: {mediation_type}")

# Save mediation results
mediation_results = pd.DataFrame({
    'Path': ['Total Effect (c)', 'a path (Age→Knowledge)', 'b path (Knowledge→Anxiety)', 
             'Direct Effect (c\')', 'Indirect Effect (a×b)'],
    'Coefficient': [total_effect, a_path, b_path, c_prime, indirect_effect],
    'Description': [
        'Age → Anxiety (without mediator)',
        'Age → Knowledge',
        'Knowledge → Anxiety (controlling for age)',
        'Age → Anxiety (controlling for knowledge)',
        'Mediated effect through knowledge'
    ]
})
mediation_results.to_csv(os.path.join(OUTPUT_DIR, 'RQ3_mediation_analysis.csv'), index=False)
print("✓ Mediation analysis results saved")

# Summary statistics for Power BI
mediation_summary = pd.DataFrame({
    'Metric': ['Total Effect', 'Direct Effect', 'Indirect Effect', 
               'Mediation Percentage', 'Mediation Type'],
    'Value': [f"{total_effect:.4f}", f"{c_prime:.4f}", f"{indirect_effect:.4f}", 
              f"{mediation_percentage:.2f}%", mediation_type]
})
mediation_summary.to_csv(os.path.join(OUTPUT_DIR, 'RQ3_mediation_summary.csv'), index=False)
print("✓ Mediation summary saved")

# Age-Anxiety relationship by knowledge level (for visualization)
mediation_data['Knowledge_Level'] = pd.cut(mediation_data['ai_literacy_score'], 
                                            bins=[0, 3.33, 6.67, 10],
                                            labels=['Low', 'Medium', 'High'])
age_anxiety_by_knowledge = mediation_data.groupby(['What is your age range?', 'Knowledge_Level']).agg({
    'anxiety_index': ['mean', 'std', 'count']
}).reset_index()
age_anxiety_by_knowledge.columns = ['Age_Range', 'Knowledge_Level', 'Anxiety_Mean', 'Anxiety_Std', 'Count']
age_anxiety_by_knowledge.to_csv(os.path.join(OUTPUT_DIR, 'RQ3_age_anxiety_by_knowledge.csv'), index=False)
print("✓ Age-anxiety by knowledge level saved")

# Correlation matrix for mediation variables
correlation_matrix = mediation_data[['age_numeric', 'ai_literacy_score', 'anxiety_index']].corr()
correlation_matrix.columns = ['Age', 'Knowledge', 'Anxiety']
correlation_matrix.index = ['Age', 'Knowledge', 'Anxiety']
correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, 'RQ3_correlation_matrix.csv'))
print("✓ Correlation matrix saved")


# ============================================================================
# FINAL SUMMARY AND MASTER DATASET
# ============================================================================
print("\n" + "="*80)
print("CREATING MASTER DATASETS FOR POWER BI")
print("="*80)

# Master summary of all research questions
master_summary = pd.DataFrame({
    'Research_Question': [
        'Main: Predictors of Trust',
        'Main: Trust by Age',
        'Main: Trust by Gender',
        'RQ1: Knowledge Gap by Age',
        'RQ1: Knowledge Gap by Gender',
        'RQ2: Knowledge vs Age (Trust)',
        'RQ2: Knowledge vs Age (Anxiety)',
        'RQ3: Age-Anxiety Association',
        'RQ3: Knowledge Mediation'
    ],
    'Key_Finding': [
        f"R² = {model.rsquared:.3f}, Top predictor: {regression_results.iloc[1]['Variable']}",
        f"F = {anova_table.iloc[0]['F']:.2f}, p = {anova_table.iloc[0]['PR(>F)']:.4f}",
        f"F = {anova_table.iloc[1]['F']:.2f}, p = {anova_table.iloc[1]['PR(>F)']:.4f}",
        f"Mean gap = {gaps_by_age['Knowledge_Gap'].mean():.2f}",
        f"Mean gap = {gaps_by_gender['Knowledge_Gap'].mean():.2f}",
        f"Stronger: {standardized_effects.loc[standardized_effects['Abs_Beta_Trust'].idxmax(), 'Variable']}",
        f"Stronger: {standardized_effects.loc[standardized_effects['Abs_Beta_Anxiety'].idxmax(), 'Variable']}",
        f"Total effect = {total_effect:.3f}, p = {model_total.pvalues['age_numeric']:.4f}",
        f"{mediation_type}, {mediation_percentage:.1f}% mediated"
    ],
    'Statistical_Test': [
        'Multiple Regression',
        'Two-way ANOVA',
        'Two-way ANOVA',
        'One-way ANOVA',
        'T-test',
        'Standardized Regression',
        'Standardized Regression',
        'Simple Regression',
        'Mediation Analysis'
    ],
    'Significance': [
        'Yes' if model.f_pvalue < 0.05 else 'No',
        'Yes' if anova_table.iloc[0]['PR(>F)'] < 0.05 else 'No',
        'Yes' if anova_table.iloc[1]['PR(>F)'] < 0.05 else 'No',
        'Yes' if p_val < 0.05 else 'No',
        'TBD',
        'See file',
        'See file',
        'Yes' if model_total.pvalues['age_numeric'] < 0.05 else 'No',
        mediation_type
    ]
})
master_summary.to_csv(os.path.join(OUTPUT_DIR, 'MASTER_SUMMARY.csv'), index=False)
print("\n✓ Master summary created")

# Create a comprehensive dataset with all variables for Power BI
comprehensive_dataset = df[[
    # Demographics
    'What is your age range?', 'What is your gender?', 
    'What is your education level?', 'What is your employment status?',
    # Core variables
    'trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score',
    # Behavioral variables
    'tech_comfort_level', 'ai_usage_score', 'device_hours_score',
    # Psychological variables
    'beneficial_score', 'future_intent_score', 'threat_freedoms_score',
    'job_elimination_score', 'personal_job_threat_score',
    # Derived variables
    'trust_anxiety_gap', 'knowledge_gap', 'literacy_perception_gap',
    'attitude_category', 'self_awareness_category', 'gap_category',
    # Composite scores
    'trust_anxiety_balance', 'response_consistency_score'
]].copy()

# Add numeric age for Power BI calculations
comprehensive_dataset['Age_Numeric'] = comprehensive_dataset['What is your age range?'].map(age_mapping)

# Add categorical levels for filtering
comprehensive_dataset['Trust_Level'] = pd.cut(comprehensive_dataset['trust_index'], 
                                               bins=[0, 3.33, 6.67, 10], 
                                               labels=['Low', 'Medium', 'High'])
comprehensive_dataset['Anxiety_Level'] = pd.cut(comprehensive_dataset['anxiety_index'], 
                                                 bins=[0, 3.33, 6.67, 10], 
                                                 labels=['Low', 'Medium', 'High'])
comprehensive_dataset['Knowledge_Level'] = pd.cut(comprehensive_dataset['ai_literacy_score'], 
                                                   bins=[0, 3.33, 6.67, 10], 
                                                   labels=['Low', 'Medium', 'High'])

comprehensive_dataset.to_csv(os.path.join(OUTPUT_DIR, 'COMPREHENSIVE_DATASET.csv'), index=False)
print("✓ Comprehensive dataset created")

# Create a data dictionary for Power BI users
data_dictionary = pd.DataFrame({
    'Variable_Name': [
        'trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score',
        'knowledge_gap', 'Age_Numeric', 'Trust_Level', 'Anxiety_Level', 'Knowledge_Level'
    ],
    'Description': [
        'Composite trust score (0-10): average of trust, beneficial, and future intent',
        'Composite anxiety score (0-10): average of threat, job concerns, and consciousness fears',
        'Self-reported perceived knowledge about AI (0-10)',
        'Actual knowledge based on quiz questions (0-10)',
        'Difference between perceived and actual knowledge (positive = overconfident)',
        'Age group coded numerically (1=Under 18, 2=18-24, 3=25-34, 4=35-44, 5=45+)',
        'Trust categorized as Low (0-3.33), Medium (3.33-6.67), or High (6.67-10)',
        'Anxiety categorized as Low (0-3.33), Medium (3.33-6.67), or High (6.67-10)',
        'Actual knowledge categorized as Low (0-3.33), Medium (3.33-6.67), or High (6.67-10)'
    ],
    'Type': ['Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 
             'Ordinal', 'Categorical', 'Categorical', 'Categorical'],
    'Range': ['0-10', '0-10', '0-10', '0-10', '-10 to +10', '1-5', 
              'Low/Medium/High', 'Low/Medium/High', 'Low/Medium/High']
})
data_dictionary.to_csv(os.path.join(OUTPUT_DIR, 'DATA_DICTIONARY.csv'), index=False)
print("✓ Data dictionary created")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✅ All files saved in '{OUTPUT_DIR}' folder")
print(f"\n📊 Files created:")
print("   1. main_trust_analysis.csv - Main dataset with all variables")
print("   2. trust_predictors_regression.csv - Regression results")
print("   3. trust_by_age_gender.csv - Trust means by demographics")
print("   4. anova_trust_age_gender.csv - Two-way ANOVA results")
print("   5. RQ1_knowledge_gaps_by_age.csv - Knowledge gaps by age")
print("   6. RQ1_knowledge_gaps_by_gender.csv - Knowledge gaps by gender")
print("   7. RQ1_knowledge_gaps_by_education.csv - Knowledge gaps by education")
print("   8. RQ1_gap_distribution_by_demographics.csv - Gap categories")
print("   9. RQ1_individual_knowledge_gaps.csv - Individual-level data")
print("   10. RQ2_trust_model_comparison.csv - Model comparison for trust")
print("   11. RQ2_anxiety_model_comparison.csv - Model comparison for anxiety")
print("   12. RQ2_standardized_effects.csv - Beta coefficients")
print("   13. RQ3_mediation_analysis.csv - Mediation path coefficients")
print("   14. RQ3_mediation_summary.csv - Mediation summary")
print("   15. RQ3_age_anxiety_by_knowledge.csv - Age-anxiety by knowledge level")
print("   16. RQ3_correlation_matrix.csv - Correlation matrix")
print("   17. MASTER_SUMMARY.csv - Summary of all research questions")
print("   18. COMPREHENSIVE_DATASET.csv - Complete dataset for Power BI")
print("   19. DATA_DICTIONARY.csv - Variable descriptions")
print("\n🎯 Next Steps:")
print("   1. Run this script: python research_questions_analysis.py")
print("   2. Import CSV files into Power BI Desktop")
print("   3. Create visualizations for each research question")
print("   4. Use POWERBI_VISUALIZATION_GUIDE.md for specific instructions")
print("\n" + "="*80)
