# مكتبة Pandas: لمعالجة الجداول (DataFrame)، القراءة/الكتابة لـ CSV، التجميع والإحصاء
import pandas as pd
# مكتبة NumPy: عمليات عددية سريعة، حسابات المصفوفات والمتجهات
import numpy as np
# Matplotlib: إنشاء الرسوم البيانية الثابتة (صور PNG وغيرها)
import matplotlib.pyplot as plt
# Seaborn: تحسين المظهر الافتراضي للرسوم وإضافة مخططات إحصائية سهلة
import seaborn as sns
# SciPy.stats: وظائف إحصائية عامة مثل t-tests، توزيعات واحتمالات
from scipy import stats
# SciPy.stats (اختبارات محددة): كاي-تربيع للجداول، ANOVA أحادي، مان-ويتني، كروسكال-واليس
from scipy.stats import chi2_contingency, f_oneway, mannwhitneyu, kruskal
# warnings: لإخفاء التحذيرات غير المهمة أثناء التنفيذ للحصول على إخراج نظيف
import warnings
# تعطيل التحذيرات العامة التي قد تشتت القارئ
warnings.filterwarnings('ignore')
# sys: معلومات عن النظام وبيئة التشغيل (مثل المنصة)، مفيد لضبط الترميز على ويندوز
import sys
# os: التعامل مع نظام الملفات، المسارات، وإنشاء المجلدات
import os

# Fix encoding for Windows Arabic systems
# إصلاح ترميز الإخراج على ويندوز لدعم العربية في الطرفية بشكل صحيح
if sys.platform == 'win32':
    # على ويندوز: نستخدم io لتغليف تدفقات الإخراج لضمان عرض العربية
    import io
    # ضبط stdout ليستخدم ترميز UTF-8 لتفادي ظهور رموز غير مفهومة
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    # ضبط stderr بنفس الترميز لضمان اتساق رسائل الأخطاء بالعربية
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Statsmodels imports
# statsmodels.api: واجهة منخفضة المستوى لبناء النماذج (OLS وغيرها)
import statsmodels.api as sm
# الصياغة بصيغة المعادلات (R-style) لبناء نماذج OLS بسهولة
from statsmodels.formula.api import ols
# اختبار Tukey HSD للمقارنات بعد ANOVA بين الأزواج من المجموعات
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# تنفيذ ANOVA لقياس تأثير العوامل على المتغير التابع
from statsmodels.stats.anova import anova_lm

# Plotly imports
# plotly.express: واجهة سهلة وسريعة لإنشاء مرئيات تفاعلية
import plotly.express as px
# plotly.graph_objects: بناء أشكال تفاعلية مخصصة بتفاصيل أكبر
import plotly.graph_objects as go
# إنشاء مخططات متعددة ضمن شكل واحد (subplots) لعرض عدة رسومات
from plotly.subplots import make_subplots

# Sklearn imports for enhanced clustering
# KMeans: خوارزمية التجميع الشهيرة لتقسيم البيانات إلى مجموعات بناءً على التشابه
from sklearn.cluster import KMeans
# StandardScaler: تقييس الميزات (متوسط 0، انحراف معياري 1) لتحسين أداء التجميع/النماذج
from sklearn.preprocessing import StandardScaler
# silhouette_score, davies_bouldin_score: تقييم جودة التجميع وعدد المجموعات الأمثل
from sklearn.metrics import silhouette_score, davies_bouldin_score
# LinearRegression, LogisticRegression: نماذج تنبؤية (خطية للثقة، ولوجستية للفئات)
from sklearn.linear_model import LinearRegression, LogisticRegression
# train_test_split: تقسيم البيانات إلى تدريب/اختبار للتقييم العادل
from sklearn.model_selection import train_test_split
# r2_score, mean_squared_error, classification_report, confusion_matrix: مقاييس تقييم النماذج
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix

import os

# Configuration
# Smart path detection - works whether run from main folder or scripts folder
# كشف المسار الذكي لتحديد ملف البيانات ومجلد الإخراج بغض النظر عن موقع التشغيل
if os.path.exists('data/data_final_with_derived_columns.csv'):
    # الحالة 1: التشغيل من المجلد الرئيسي؛ استخدم مسار data/ واحفظ في outputs/
    DATA_FILE = 'data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs'
elif os.path.exists('../data/data_final_with_derived_columns.csv'):
    # الحالة 2: التشغيل من scripts/؛ ملف البيانات في ../data/ والمخرجات في ../outputs/
    DATA_FILE = '../data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = '../outputs'
elif os.path.exists('outputs/data_final_with_derived_columns.csv'):
    # الحالة 3: إذا كان الملف موجوداً داخل outputs/ (سيناريو نقل الملف سابقاً)
    DATA_FILE = 'outputs/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs'
elif os.path.exists('../outputs/data_final_with_derived_columns.csv'):
    # الحالة 4: التشغيل من scripts/ مع وجود الملف في ../outputs/
    DATA_FILE = '../outputs/data_final_with_derived_columns.csv'
    OUTPUT_DIR = '../outputs'
else:
    # افتراضي: نفترض وجود الملف تحت data/ وننشئ outputs/ إن لزم
    DATA_FILE = 'data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs'

# Create organized output subdirectories
STATS_DIR = os.path.join(OUTPUT_DIR, '03_statistical_tests')
VIZ_DIR = os.path.join(OUTPUT_DIR, '02_visualizations')
PLOTLY_DIR = os.path.join(OUTPUT_DIR, '07_interactive_plots')
CLUSTER_DIR = os.path.join(OUTPUT_DIR, '05_clustering')
PREDICT_DIR = os.path.join(OUTPUT_DIR, '06_predictive_models')
POWERBI_DIR = os.path.join(OUTPUT_DIR, '08_powerbi_export')

# Create output directories
# التأكد من وجود جميع مجلدات الإخراج وإنشاؤها عند الحاجة لتجنب أخطاء الحفظ
for dir_path in [OUTPUT_DIR, STATS_DIR, VIZ_DIR, PLOTLY_DIR, CLUSTER_DIR, PREDICT_DIR, POWERBI_DIR]:
    # إنشاء المجلد إن لم يكن موجوداً لمنع أخطاء الحفظ
    os.makedirs(dir_path, exist_ok=True)

print("="*80)
print("COMPLETE ADVANCED ANALYSIS - AI IMPACT STUDY")
print("="*80)

# Load data
print("\n[1/10] Loading data...")
# قراءة ملف CSV إلى DataFrame مع ترميز UTF-8 لضمان دعم العربية
df = pd.read_csv(DATA_FILE, encoding='utf-8')
# طباعة عدد الصفوف والأعمدة لتأكيد التحميل الصحيح وإعطاء فكرة عن حجم البيانات
print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")


# ============================================================================
# PART 1: STATSMODELS REGRESSION ANALYSIS
# ============================================================================
print("\n[2/10] Performing Statsmodels Regression Analysis...")
# بدء تنفيذ نماذج الانحدار الإحصائي لشرح العوامل المؤثرة على الثقة

# Multiple Linear Regression: Predict trust_index
print("\n--- Multiple Linear Regression: Predicting Trust Index ---")

# Prepare data for regression
regression_data = df[['trust_index', 'knowledge_score', 'ai_literacy_score', 
                       'tech_comfort_level', 'device_hours_score']].dropna()
# إسقاط القيم المفقودة لضمان أن النموذج يتعامل مع بيانات مكتملة فقط

# Build formula for OLS
# صياغة معادلة OLS: الثقة ~ المعرفة + المهارة الفعلية + الراحة التقنية + ساعات الأجهزة
formula = 'trust_index ~ knowledge_score + ai_literacy_score + tech_comfort_level + device_hours_score'
# ملاءمة النموذج باستخدام المعادلة المحددة على بيانات الانحدار
model_trust = ols(formula, data=regression_data).fit()

print(model_trust.summary())
# عرض جدول الملخص الذي يشمل معاملات المتغيرات وإحصاءات الأهمية ومدى جودة الملاءمة
# عرض ملخص النموذج: يشمل معاملات المتغيرات، الأخطاء المعيارية، إحصائيات T وP،
# مقاييس الملاءمة مثل R² وAdj R²، مما يساعد على تقييم قوة النموذج وتفسير النتائج.

# Save regression results
# تحويل نتائج النموذج إلى جدول منظم يُسهّل التحليل والعرض في أدوات خارجية
regression_results = pd.DataFrame({
    'Variable': model_trust.params.index,
    'Coefficient': model_trust.params.values,
    'Std_Error': model_trust.bse.values,
    'T_Statistic': model_trust.tvalues.values,
    'P_Value': model_trust.pvalues.values,
    'CI_Lower': model_trust.conf_int()[0].values,
    'CI_Upper': model_trust.conf_int()[1].values
})
# حفظ النتائج كملف CSV داخل مجلد الاختبارات الإحصائية
regression_results.to_csv(os.path.join(STATS_DIR, 'regression_trust_results.csv'), index=False)
print(f"✓ Regression results saved")
print(f"  R-squared: {model_trust.rsquared:.4f}")
print(f"  Adj R-squared: {model_trust.rsquared_adj:.4f}")
print(f"  F-statistic: {model_trust.fvalue:.4f}, p-value: {model_trust.f_pvalue:.4e}")

# Regression with demographic variables
print("\n--- Regression with Demographics ---")
# إضافة العوامل الفئوية العمر والجنس بصيغة C() مع Q() لحماية أسماء الأعمدة ذات المسافات
formula_demo = 'trust_index ~ knowledge_score + ai_literacy_score + C(Q("What is your gender?")) + C(Q("What is your age range?"))'
# ملاءمة نموذج OLS مع العوامل الفئوية لاختبار تأثيرها على الثقة
model_trust_demo = ols(formula_demo, data=df).fit()
print(model_trust_demo.summary())

# Save demographic regression results
# استخراج معاملات النموذج والقيم الاحتمالية للعوامل الفئوية لحفظها وتحليلها لاحقاً
demo_results = pd.DataFrame({
    'Variable': model_trust_demo.params.index,
    'Coefficient': model_trust_demo.params.values,
    'P_Value': model_trust_demo.pvalues.values
})
demo_results.to_csv(os.path.join(STATS_DIR, 'regression_trust_demographics.csv'), index=False)
print(f"✓ Demographic regression results saved")


# ============================================================================
# PART 2: ANOVA AND POST-HOC TESTS
# ============================================================================
print("\n[3/10] Performing ANOVA and Post-hoc Tests...")
# بدء اختبارات ANOVA أحادي الاتجاه لفحص الفروق بين المجموعات، يتلوها اختبار Tukey للمقارنات الزوجية

# One-way ANOVA: Trust by Age Group
print("\n--- ANOVA: Trust Index by Age Group ---")
# صياغة نموذج OLS حيث المتغير التابع هو trust_index والعامل الفئوي هو الفئة العمرية
formula_anova = 'trust_index ~ C(Q("What is your age range?"))'
# ملاءمة النموذج ثم حساب جدول ANOVA من النوع 2
model_anova_age = ols(formula_anova, data=df).fit()
anova_table_age = anova_lm(model_anova_age, typ=2)
# anova_lm بنوع 2 (Type II) يقيم تأثير كل عامل بعد التحكم بالعوامل الأخرى (إن وجدت)
print(anova_table_age)

# Save ANOVA results
# حفظ جدول ANOVA الذي يحتوي على مجاميع المربعات، درجات الحرية، قيمة F والقيمة الاحتمالية
anova_table_age.to_csv(os.path.join(STATS_DIR, 'anova_trust_by_age.csv'))
print(f"✓ ANOVA results saved")
print(f"  F-statistic: {anova_table_age['F'].iloc[0]:.4f}")
print(f"  P-value: {anova_table_age['PR(>F)'].iloc[0]:.4e}")

# Post-hoc Tukey HSD test
print("\n--- Post-hoc Tukey HSD Test ---")
# بعد وجود فروق عامة من ANOVA، يسمح Tukey HSD بفحص الأزواج لمعرفة أي المجموعات تختلف تحديداً
tukey_age = pairwise_tukeyhsd(endog=df['trust_index'].dropna(), 
                               groups=df.loc[df['trust_index'].notna(), 'What is your age range?'], 
                               alpha=0.05)
# عرض جدول المقارنات الزوجية بما في ذلك الفروق المتوسطة وحدود الثقة والدلالة
print(tukey_age)

# Save Tukey results
# حفظ نتائج Tukey في CSV لتسهيل عرض الأزواج المهمة إحصائياً وحدود الفروق
tukey_df = pd.DataFrame(data=tukey_age.summary().data[1:], columns=tukey_age.summary().data[0])
tukey_df.to_csv(os.path.join(STATS_DIR, 'posthoc_tukey_age.csv'), index=False)
print(f"✓ Tukey HSD results saved")

# ANOVA: Trust by Education Level
print("\n--- ANOVA: Trust Index by Education Level ---")
# نموذج ANOVA آخر يفحص تأثير المستوى التعليمي على مؤشر الثقة
formula_anova_edu = 'trust_index ~ C(Q("What is your education level?"))'
model_anova_edu = ols(formula_anova_edu, data=df).fit()
anova_table_edu = anova_lm(model_anova_edu, typ=2)
print(anova_table_edu)
anova_table_edu.to_csv(os.path.join(STATS_DIR, 'anova_trust_by_education.csv'))

# ANOVA: Anxiety by Age Group
print("\n--- ANOVA: Anxiety Index by Age Group ---")
# فحص تأثير العمر على مؤشر القلق باستخدام ANOVA أحادي الاتجاه
formula_anova_anx = 'anxiety_index ~ C(Q("What is your age range?"))'
model_anova_anx = ols(formula_anova_anx, data=df).fit()
anova_table_anx = anova_lm(model_anova_anx, typ=2)
print(anova_table_anx)
anova_table_anx.to_csv(os.path.join(STATS_DIR, 'anova_anxiety_by_age.csv'))


# ============================================================================
# PART 3: CHI-SQUARE TESTS FOR CATEGORICAL RELATIONSHIPS
# ============================================================================
print("\n[4/10] Performing Chi-Square Tests...")
# تنفيذ اختبارات كاي-تربيع للعلاقات بين المتغيرات الفئوية (العمر/الجنس/التعليم/الوظيفة مقابل الفئات)

def perform_chi_square(df, var1, var2, output_name):
    """Perform Chi-Square test and calculate Cramér's V"""
    # دالة تنفّذ اختبار كاي-تربيع بين متغيرين فئويين وتحسب Cramér's V كقياس قوة علاقة
    # بناء جدول تقاطع (contingency) يحصي تكرارات كل تركيبة من الفئتين
    contingency_table = pd.crosstab(df[var1], df[var2])
    # تطبيق اختبار كاي-تربيع لاكتشاف وجود علاقة ذات دلالة بين المتغيرين الفئويين
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    # expected: القيم المتوقعة نظرياً في كل خلية إذا لم تكن هناك علاقة؛ تُقارن بالقيم المرصودة
    
    # Calculate Cramér's V
    # n: إجمالي عدد الملاحظات في الجدول؛ min_dim: أصغر بُعد ناقص واحد لتطبيع القياس
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
    # حساب Cramér's V من قيمة كاي-تربيع بعد التطبيع بحجم العينة وأبعاد الجدول
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    # Cramér's V يقيس قوة العلاقة بين متغيرين فئويين (0: لا علاقة، 1: علاقة قوية)
    
    print(f"\n--- Chi-Square: {var1} vs {var2} ---")
    print(f"  Chi-Square: {chi2:.4f}")
    print(f"  P-value: {p_val:.4e}")
    print(f"  Degrees of Freedom: {dof}")
    print(f"  Cramér's V: {cramers_v:.4f}")
    
    # Save contingency table
    # حفظ جدول التقاطع لتقييم التوزيع الفعلي للفئات ومراجعته بصرياً/كامياً
    contingency_table.to_csv(os.path.join(STATS_DIR, f'crosstab_{output_name}.csv'))
    
    # Create heatmap
    # إنشاء خريطة حرارية لعرض جدول التقاطع بشكل بصري واضح
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
    plt.title(f'Contingency Table: {var1} vs {var2}')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f'heatmap_{output_name}.png'), dpi=300)
    plt.close()
    
    return {'chi2': chi2, 'p_value': p_val, 'dof': dof, 'cramers_v': cramers_v}

# Perform multiple Chi-Square tests
chi_square_results = []

# Age vs Attitude Category
result1 = perform_chi_square(df, 'What is your age range?', 'attitude_category', 'age_attitude')
chi_square_results.append({'Test': 'Age vs Attitude', **result1})

# Gender vs Attitude Category
result2 = perform_chi_square(df, 'What is your gender?', 'attitude_category', 'gender_attitude')
chi_square_results.append({'Test': 'Gender vs Attitude', **result2})

# Education vs Attitude Category
result3 = perform_chi_square(df, 'What is your education level?', 'attitude_category', 'education_attitude')
chi_square_results.append({'Test': 'Education vs Attitude', **result3})

# Employment vs Attitude Category
result4 = perform_chi_square(df, 'What is your employment status?', 'attitude_category', 'employment_attitude')
chi_square_results.append({'Test': 'Employment vs Attitude', **result4})

# Age vs Self-Awareness Category
result5 = perform_chi_square(df, 'What is your age range?', 'self_awareness_category', 'age_awareness')
chi_square_results.append({'Test': 'Age vs Self-Awareness', **result5})

# Save all Chi-Square results
chi_square_df = pd.DataFrame(chi_square_results)
chi_square_df.to_csv(os.path.join(STATS_DIR, 'chi_square_all_tests.csv'), index=False)
print(f"\n✓ All Chi-Square tests completed and saved")
# تأكيد اكتمال جميع اختبارات كاي-تربيع وحفظ النتائج


# ============================================================================
# PART 4: EFFECT SIZE CALCULATIONS
# ============================================================================
print("\n[5/10] Calculating Effect Sizes...")
# حساب أحجام الأثر لفهم حجم الاختلافات والعلاقات إحصائياً

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    # دالة لحساب حجم الأثر Cohen's d بين مجموعتين مستقلتين
    # n1/n2: أحجام العينات؛ var1/var2: التباين داخل كل مجموعة
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    # pooled_std: الانحراف المعياري المُجمّع الذي يجمع تباين المجموعتين بطريقة موزونة
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    # الفرق في المتوسطات مقسوماً على الانحراف المعياري المُجمّع يعطي حجم الأثر d
    return (group1.mean() - group2.mean()) / pooled_std
    # تفسير: القيم ~0.2 صغيرة، ~0.5 متوسطة، ~0.8 كبيرة (تقريباً)

# Gender effect on trust
males = df[df['What is your gender?'] == 'Male']['trust_index'].dropna()
females = df[df['What is your gender?'] == 'Female']['trust_index'].dropna()
d_gender_trust = cohens_d(males, females)
print(f"\n--- Effect Sizes ---")
print(f"  Cohen's d (Gender on Trust): {d_gender_trust:.4f}")

# Gender effect on anxiety
males_anx = df[df['What is your gender?'] == 'Male']['anxiety_index'].dropna()
females_anx = df[df['What is your gender?'] == 'Female']['anxiety_index'].dropna()
d_gender_anxiety = cohens_d(males_anx, females_anx)
print(f"  Cohen's d (Gender on Anxiety): {d_gender_anxiety:.4f}")

# Calculate eta-squared for ANOVA (Age on Trust)
ss_between = anova_table_age['sum_sq'].iloc[0]
ss_total = anova_table_age['sum_sq'].sum()
eta_squared = ss_between / ss_total
print(f"  Eta-squared (Age on Trust): {eta_squared:.4f}")
# eta²: نسبة التباين المفسَّر بالعامل (العمر) إلى التباين الكلي؛ يعطى حجم الأثر في ANOVA

# Save effect sizes
# إنشاء جدول مُفسَّر يضم مقارنة/حجم الأثر/تفسير وصفي للحجم
effect_sizes = pd.DataFrame({
    'Comparison': ['Gender on Trust', 'Gender on Anxiety', 'Age on Trust (eta²)'],
    'Effect_Size': [d_gender_trust, d_gender_anxiety, eta_squared],
    'Interpretation': [
        'Small' if abs(d_gender_trust) < 0.5 else 'Medium' if abs(d_gender_trust) < 0.8 else 'Large',
        'Small' if abs(d_gender_anxiety) < 0.5 else 'Medium' if abs(d_gender_anxiety) < 0.8 else 'Large',
        'Small' if eta_squared < 0.06 else 'Medium' if eta_squared < 0.14 else 'Large'
    ]
})
effect_sizes.to_csv(os.path.join(STATS_DIR, 'effect_sizes.csv'), index=False)
print(f"✓ Effect sizes saved")


# ============================================================================
# PART 5: RESEARCH QUESTION ANALYSIS
# ============================================================================
print("\n[6/10] Analyzing Research Questions...")
# تحليل أسئلة بحثية محددة وإنتاج جداول مهيأة لـ Power BI للإجابة عنها بصرياً وإحصائياً

# Research Question 1: Trust related to age groups?
print("\n--- RQ1: Trust and Age Relationship ---")
# تجميع الثقة حسب الفئة العمرية وقياس المتوسط والانحراف المعياري والعدد لكل مجموعة
trust_by_age = df.groupby('What is your age range?')['trust_index'].agg(['mean', 'std', 'count']).reset_index()
# حساب متوسط الثقة والانحراف المعياري وعدد العينات لكل فئة عمرية، ثم ترتيب النتائج تنازلياً بحسب المتوسط
trust_by_age.columns = ['Age_Range', 'Mean_Trust', 'Std_Trust', 'Count']
trust_by_age = trust_by_age.sort_values('Mean_Trust', ascending=False)
print(trust_by_age)
trust_by_age.to_csv(os.path.join(POWERBI_DIR, 'RQ1_trust_by_age.csv'), index=False)
print(f"✓ RQ1 analysis saved")

# Research Question 2: Knowledge and Trust relationship?
print("\n--- RQ2: Knowledge and Trust Relationship ---")
# حساب معاملات الارتباط بين الثقة والمعرفة المتصورة والثقة والمعرفة الفعلية (literacy)
corr_knowledge_trust = df['knowledge_score'].corr(df['trust_index'])
corr_literacy_trust = df['ai_literacy_score'].corr(df['trust_index'])
print(f"  Correlation (Perceived Knowledge × Trust): {corr_knowledge_trust:.4f}")
print(f"  Correlation (Actual Literacy × Trust): {corr_literacy_trust:.4f}")

rq2_results = pd.DataFrame({
    'Relationship': ['Perceived Knowledge → Trust', 'Actual Literacy → Trust'],
    'Correlation': [corr_knowledge_trust, corr_literacy_trust],
    'Interpretation': [
        'Weak positive' if corr_knowledge_trust < 0.3 else 'Moderate positive' if corr_knowledge_trust < 0.7 else 'Strong positive',
        'Weak positive' if corr_literacy_trust < 0.3 else 'Moderate positive' if corr_literacy_trust < 0.7 else 'Strong positive'
    ]
})
# إطار نتائج يوضّح قوة الارتباط واتجاهه مع تفسير وصفي (ضعيف/متوسط/قوي)
rq2_results.to_csv(os.path.join(POWERBI_DIR, 'RQ2_knowledge_trust.csv'), index=False)
print(f"✓ RQ2 analysis saved")

# Research Question 3: Barriers limiting AI usage?
print("\n--- RQ3: Barriers to AI Usage ---")
barrier_cols = {
    'threat_freedoms_score': 'Threat to Individual Freedoms',
    'job_elimination_score': 'Job Elimination Concerns',
    'personal_job_threat_score': 'Personal Job Threat',
    'ethical_constraints_score': 'Need for Ethical Constraints',
    'anxiety_index': 'Overall Anxiety Index'
}
# خريطة أسماء الأعمدة إلى تسميات مفهومة للمستخدم النهائي لعرضها في الجداول/المرئيات

barriers = df[list(barrier_cols.keys())].mean().sort_values(ascending=False)
# حساب متوسطات العوائق وترتيبها تنازلياً لتحديد أهم العوائق إدراكاً
barriers_df = pd.DataFrame({
    'Barrier': [barrier_cols[col] for col in barriers.index],
    'Mean_Score': barriers.values,
    'Percentage_High_Concern': [(df[col] >= 7).sum() / len(df) * 100 for col in barriers.index]
})
print(barriers_df)
barriers_df.to_csv(os.path.join(POWERBI_DIR, 'RQ3_barriers_analysis.csv'), index=False)
print(f"✓ RQ3 analysis saved")
print(f"  Top Barrier: {barriers_df.iloc[0]['Barrier']} (Mean: {barriers_df.iloc[0]['Mean_Score']:.2f}/10)")

# Research Question 4: Most common concerns?
print("\n--- RQ4: Most Common Concerns ---")
concern_cols = {
    'threat_freedoms_score': 'Threat to Freedoms',
    'job_elimination_score': 'Job Elimination',
    'personal_job_threat_score': 'Personal Job Threat',
    'consciousness_score': 'AI Consciousness'
}

concerns_high = {}
for col, name in concern_cols.items():
    high_count = (df[col] >= 7).sum()
    percentage = (high_count / len(df)) * 100
    concerns_high[name] = {'Count': high_count, 'Percentage': percentage}
# نحسب عدد من لديهم مستوى مرتفع (>=7) ونسبة هؤلاء من إجمالي العينة لكل نوع من المخاوف

concerns_df = pd.DataFrame(concerns_high).T.reset_index()
concerns_df.columns = ['Concern', 'Count_High', 'Percentage_High']
concerns_df = concerns_df.sort_values('Percentage_High', ascending=False)
# إنشاء جدول يلخّص أكثر المخاوف شيوعاً مع عدد ونسبة من لديهم مستوى قلق مرتفع لكل بند
print(concerns_df)
concerns_df.to_csv(os.path.join(POWERBI_DIR, 'RQ4_common_concerns.csv'), index=False)
print(f"✓ RQ4 analysis saved")
print(f"  Most Common Concern: {concerns_df.iloc[0]['Concern']} ({concerns_df.iloc[0]['Percentage_High']:.1f}% high concern)")

# Research Question 5: Concerns by age?
print("\n--- RQ5: Concerns by Age Group ---")
concerns_by_age = df.groupby('What is your age range?')[list(concern_cols.keys())].mean()
concerns_by_age.columns = [concern_cols[col] for col in concerns_by_age.columns]
print(concerns_by_age)
concerns_by_age.to_csv(os.path.join(POWERBI_DIR, 'RQ5_concerns_by_age.csv'))
print(f"✓ RQ5 analysis saved")


# ============================================================================
# PART 6: ENHANCED CLUSTERING WITH VALIDATION
# ============================================================================
print("\n[7/10] Performing Enhanced Clustering Analysis...")
# بدء التحليل المحسّن للتجميع مع تقييم العدد الأمثل للمجموعات

# Prepare features for clustering
features = ['trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score']
# السمات الأساسية المستخدمة في التجميع: الثقة، القلق، المعرفة، والمهارة الفعلية
df_cluster = df[features].dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cluster)

# Elbow Method
print("\n--- Elbow Method for Optimal K ---")
# تطبيق طريقة الكوع لتقدير العدد الأمثل للمجموعات ومقارنته بدرجات السيلويت
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_scores.append(sil_score)
    # في كل قيمة k: نسجل Inertia (مجموع مربعات المسافات داخل المجموعات)
    # ودرجة Silhouette (مدى انفصال/تمايز المجموعات). القيم الأفضل: Inertia منخفض وSilhouette مرتفع.
    print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}")

# Save elbow data
elbow_data = pd.DataFrame({
    'K': list(k_range),
    'Inertia': inertias,
    'Silhouette_Score': silhouette_scores
})
elbow_data.to_csv(os.path.join(CLUSTER_DIR, 'clustering_elbow_analysis.csv'), index=False)

# Plot Elbow Method
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# إنشاء شكلين فرعيين لعرض منحنى الكوع ومنحنى السيلويت جنباً إلى جنب

ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True)

ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(CLUSTER_DIR, 'clustering_validation.png'), dpi=300)
plt.close()
print(f"✓ Clustering validation plots saved")

# Optimal clustering (k=3 or k=4 based on silhouette)
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
# اختيار k الأمثل بناءً على أعلى درجة سيلويت. نضيف 2 لأن نطاق k يبدأ من 2.
print(f"\n--- Optimal K = {optimal_k} (highest silhouette score) ---")

kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_optimal.fit_predict(scaled_features)

# Add cluster labels to dataframe
df_cluster['Cluster'] = clusters

# Detailed cluster profiles
print("\n--- Cluster Profiles ---")
cluster_profiles = df_cluster.groupby('Cluster')[features].agg(['mean', 'std', 'count'])
print(cluster_profiles)
# عرض ملف تعريف كل مجموعة: المتوسط والانحراف المعياري وعدد العناصر لكل سمة،
# مما يساعد في فهم خصائص كل مجموعة بوضوح وربطها بسلوك المستخدمين.
cluster_profiles.to_csv(os.path.join(CLUSTER_DIR, 'cluster_profiles_detailed.csv'))

# Assign cluster names based on characteristics
cluster_centers = scaler.inverse_transform(kmeans_optimal.cluster_centers_)
cluster_names = []
for i, center in enumerate(cluster_centers):
    trust, anxiety, knowledge, literacy = center
    if trust > 6 and anxiety < 5:
        name = f"Cluster {i}: Optimists"
    elif trust < 5 and anxiety > 6:
        name = f"Cluster {i}: Skeptics"
    elif trust > 5 and anxiety > 5:
        name = f"Cluster {i}: Cautious"
    else:
        name = f"Cluster {i}: Neutral"
    cluster_names.append(name)
    print(f"  {name}")
    print(f"    Trust: {trust:.2f}, Anxiety: {anxiety:.2f}, Knowledge: {knowledge:.2f}, Literacy: {literacy:.2f}")

# Save cluster assignments with original indices
cluster_assignments = df_cluster.copy()
cluster_assignments['Cluster_Name'] = cluster_assignments['Cluster'].map(dict(enumerate(cluster_names)))
cluster_assignments.to_csv(os.path.join(CLUSTER_DIR, 'cluster_assignments_enhanced.csv'))
print(f"✓ Enhanced clustering results saved")


# ============================================================================
# PART 7: PREDICTIVE MODELING
# ============================================================================
print("\n[8/10] Building Predictive Models...")
# بناء وتقييم نماذج تنبؤية: انحدار خطي للتنبؤ بالثقة، ولوجستي للتنبؤ بفئة الموقف

# Linear Regression Model for Trust Prediction
print("\n--- Linear Regression: Predicting Trust Index ---")
X = df[['knowledge_score', 'ai_literacy_score', 'tech_comfort_level', 
        'device_hours_score', 'ai_usage_score']].dropna()
# اختيار الميزات ذات الصلة وإسقاط القيم المفقودة لتوحيد مجموعة التدريب/الاختبار
y = df.loc[X.index, 'trust_index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# تقسيم البيانات بنسبة 20% للاختبار مع بذرة ثابتة لضمان إعادة إنتاج النتائج

lr_model = LinearRegression()
# تهيئة نموذج الانحدار الخطي
lr_model.fit(X_train, y_train)
# تدريب نموذج الانحدار الخطي على بيانات التدريب لاستخلاص العلاقات الخطية بين الميزات والثقة

y_pred = lr_model.predict(X_test)
# حساب التنبؤات لقيم الثقة على مجموعة الاختبار
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# تقييم النموذج على بيانات الاختبار باستخدام R² (جودة الملاءمة) وRMSE (متوسط الخطأ الجذري)

print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")

# Feature importance
# استخراج معاملات النموذج كقياسات تقريبية لأهمية الميزات (قيمة أكبر → تأثير أكبر)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)
print("\n  Feature Importance:")
print(feature_importance)
feature_importance.to_csv(os.path.join(PREDICT_DIR, 'feature_importance_trust.csv'), index=False)

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Trust Index')
plt.ylabel('Predicted Trust Index')
plt.title(f'Trust Prediction Model (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PREDICT_DIR, 'prediction_trust_actual_vs_predicted.png'), dpi=300)
plt.close()
print(f"✓ Trust prediction model saved")

# Logistic Regression for Attitude Category
print("\n--- Logistic Regression: Predicting Attitude Category ---")
from sklearn.preprocessing import LabelEncoder

# Prepare data
X_class = df[['knowledge_score', 'ai_literacy_score', 'trust_index', 'anxiety_index']].dropna()
y_class = df.loc[X_class.index, 'attitude_category']
# تجهيز بيانات التصنيف مع إسقاط القيم المفقودة ومواءمة الفهارس مع y

# Encode labels
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)
# ترميز الفئات النصية إلى أرقام (0..n-1) لتناسب نموذج الانحدار اللوجستي متعدد الفئات

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded
)
# stratify: الحفاظ على نفس توزيع الفئات بين التدريب والاختبار لمنع الانحياز

log_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
log_model.fit(X_train_c, y_train_c)
# multi_class='multinomial' لتعامل النموذج مع تعدد الفئات بشكل صحيح؛ زيادة max_iter لتأكيد التقارب

y_pred_c = log_model.predict(X_test_c)
accuracy = (y_pred_c == y_test_c).mean()
# حساب الدقة كنسبة التوقعات الصحيحة إلى الإجمالي

print(f"  Accuracy: {accuracy:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test_c, y_pred_c, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
# عرض مصفوفة الالتباس مع تسميات الفئات لفهم نقاط القوة والضعف في التنبؤ
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Attitude Category Prediction')
plt.tight_layout()
plt.savefig(os.path.join(PREDICT_DIR, 'confusion_matrix_attitude.png'), dpi=300)
plt.close()
print(f"✓ Attitude prediction model saved")


# ============================================================================
# PART 8: PLOTLY INTERACTIVE VISUALIZATIONS
# ============================================================================
print("\n[9/10] Creating Plotly Interactive Visualizations...")

# 1. Interactive Scatter: Trust vs Anxiety
print("  Creating interactive scatter plot...")
# هذا الرسم يوضّح كيف تتغير الثقة مع القلق عبر الجنس، مع ترميز حجم النقطة بمستوى المعرفة
# يقدّم سياقاً غنيّاً عبر معلومات إضافية عند تمرير المؤشر على النقاط
# استخدام DataFrame df، وتحديد المحور السيني (anxiety_index) والصادي (trust_index)
fig1 = px.scatter(df, x='anxiety_index', y='trust_index',
                  # color: التلوين حسب الجنس لتفريق المجموعات بصرياً
                  color='What is your gender?',
                  # size: حجم النقطة يمثّل مستوى المعرفة المتصوّرة (قيمة أكبر → نقطة أكبر)
                  size='knowledge_score',
                  # hover_data: معلومات تظهر عند تمرير المؤشر لتقديم تفاصيل مفيدة للمستخدم
                  hover_data=['What is your age range?', 'attitude_category', 
                             'ai_literacy_score', 'tech_comfort_level'],
                  # title: عنوان تفسيري للرسم يوضّح الرسالة العامة
                  title='Trust vs Anxiety by Gender (Interactive)',
                  # labels: تسميات واضحة للمحاور، مع توضيح نطاق القيم (0-10)
                  labels={'anxiety_index': 'Anxiety Index (0-10)',
                         'trust_index': 'Trust Index (0-10)'},
                  # color_discrete_map: تثبيت ألوان محددة للجنسين لضمان الاتساق البصري
                  color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'})
# ضبط أبعاد الشكل لضمان وضوح النقاط والعناصر النصية
fig1.update_layout(height=600, width=900)
# حفظ الرسم كملف HTML تفاعلي يمكن فتحه مباشرة في المتصفح أو تضمينه في تقارير
fig1.write_html(os.path.join(PLOTLY_DIR, 'interactive_trust_vs_anxiety.html'))
print(f"    ✓ Saved: interactive_trust_vs_anxiety.html")

# 2. Interactive Correlation Heatmap
print("  Creating interactive correlation heatmap...")
# تحديد الأعمدة الرقمية ذات الصلة لبناء مصفوفة الارتباط التفاعلية
corr_cols = ['trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score', 
             'tech_comfort_level', 'beneficial_score', 'ai_usage_score']
# حساب مصفوفة الارتباط بين الأعمدة المحددة لتحديد العلاقات الخطية (من -1 إلى 1)
corr_matrix = df[corr_cols].corr()

fig2 = px.imshow(corr_matrix, 
                 # text_auto: عرض قيم الارتباط داخل الخلايا بنمط ثلاث منازل عشرية
                 text_auto='.3f',
                 # aspect="auto": ضبط الأبعاد تلقائياً لتناسب عدد الصفوف/الأعمدة
                 aspect="auto",
                 # color_continuous_scale: استخدام تدرج ألوان متباين (أحمر-أزرق) للتفريق بين السالب والموجب
                 color_continuous_scale='RdBu_r',
                 # title: عنوان يوضّح المحتوى والغرض من الخريطة الحرارية
                 title='Interactive Correlation Matrix',
                 # labels: تسمية مقياس الألوان إلى "Correlation" لسهولة الفهم
                 labels=dict(color="Correlation"))
# ضبط أبعاد الشكل لعرض جميع الخلايا بوضوح
fig2.update_layout(height=700, width=800)
# حفظ الخريطة الحرارية كملف HTML تفاعلي
fig2.write_html(os.path.join(PLOTLY_DIR, 'interactive_correlation_heatmap.html'))
print(f"    ✓ Saved: interactive_correlation_heatmap.html")

# 3. Interactive Box Plot: Trust by Age Group
print("  Creating interactive box plots...")
# هذا الرسم يوضّح توزيع مؤشر الثقة لكل فئة عمرية، مع إظهار الفروق حسب الجنس
fig3 = px.box(df, x='What is your age range?', y='trust_index',
              # x: المحور الأفقي يمثل الفئة العمرية
              color='What is your gender?',
              # y: المحور الرأسي يمثل مؤشر الثقة (يظهر الوسيط والمجال بين ربعيين)
              title='Trust Index Distribution by Age Group and Gender',
              # labels: تسميات واضحة للمحاور لتسهيل القراءة
              labels={'trust_index': 'Trust Index (0-10)',
                     'What is your age range?': 'Age Range'})
# ضبط ارتفاع/عرض الشكل لمنع ازدحام التسميات والفئات
fig3.update_layout(height=600, width=1000)
# حفظ الرسم كملف HTML تفاعلي
fig3.write_html(os.path.join(PLOTLY_DIR, 'interactive_trust_by_age.html'))
print(f"    ✓ Saved: interactive_trust_by_age.html")

# 4. Interactive 3D Scatter: Clustering Visualization
print("  Creating 3D cluster visualization...")
# التحقق من توفر عمود التجميع (Cluster) قبل إنشاء الرسم ثلاثي الأبعاد
if 'Cluster' in df_cluster.columns:
    fig4 = px.scatter_3d(df_cluster, x='trust_index', y='anxiety_index', z='knowledge_score',
                         # color: ترميز اللون حسب رقم المجموعة لتمييز العناقيد
                         color='Cluster',
                         # title: عنوان يوضّح أنه تم استخدام ثلاث سمات لرسم الفضاء ثلاثي الأبعاد
                         title='3D Cluster Visualization',
                         # labels: تسمية كل محور بما يعكس السمة الممثّلة
                         labels={'trust_index': 'Trust Index',
                                'anxiety_index': 'Anxiety Index',
                                'knowledge_score': 'Knowledge Score'},
                         # color_continuous_scale: تدرّج ألوان مناسب لعرض القيم العددية للمجموعات
                         color_continuous_scale='Viridis')
    # ضبط أبعاد الشكل لضمان سهولة دوران/تفاعل المستخدم مع الرسم
    fig4.update_layout(height=700, width=900)
    # حفظ الرسم كملف HTML تفاعلي
    fig4.write_html(os.path.join(PLOTLY_DIR, 'interactive_3d_clusters.html'))
    print(f"    ✓ Saved: interactive_3d_clusters.html")

# 5. Interactive Bar Chart: Barriers Analysis
print("  Creating interactive barriers chart...")
# إنشاء نسخة من جدول العوائق لاستخدامها في الرسم دون تعديل الجدول الأصلي
barriers_df_plot = barriers_df.copy()
fig5 = px.bar(barriers_df_plot, x='Barrier', y='Mean_Score',
              # color: تلوين الأعمدة حسب نسبة من لديهم قلق مرتفع لزيادة وضوح التأثير
              color='Percentage_High_Concern',
              # title: عنوان يوضّح أن الرسم يعرض متوسطات العوائق بشكل تفاعلي
              title='Barriers to AI Usage (Interactive)',
              # labels: تسمية المحاور والقيم بوضوح، مع الإشارة إلى نطاق 0-10 للمتوسط
              labels={'Mean_Score': 'Mean Score (0-10)',
                     'Percentage_High_Concern': '% High Concern'},
              # color_continuous_scale: استخدام تدرج أحمر لربط اللون بمستوى القلق
              color_continuous_scale='Reds')
# ضبط الأبعاد وزاوية تسميات المحور X لتحسين القراءة عند كثرة الفئات
fig5.update_layout(height=600, width=1000, xaxis_tickangle=-45)
# حفظ الرسم كملف HTML تفاعلي
fig5.write_html(os.path.join(PLOTLY_DIR, 'interactive_barriers_analysis.html'))
print(f"    ✓ Saved: interactive_barriers_analysis.html")

# 6. Interactive Sunburst: Demographics and Attitudes
print("  Creating interactive sunburst chart...")
# إعداد بيانات هرمية تربط العمر → الجنس → فئة الموقف، مع عدّ عنصر لكل سجل
sunburst_data = df[['What is your age range?', 'What is your gender?', 
                     'attitude_category']].dropna()
sunburst_data['count'] = 1

fig6 = px.sunburst(sunburst_data, 
                   # path: ترتيب المستويات الهرمية (العمر ثم الجنس ثم الفئة)
                   path=['What is your age range?', 'What is your gender?', 'attitude_category'],
                   # values: القيم المستخدمة لحجم الشرائح (عدد السجلات لكل مسار)
                   values='count',
                   # title: عنوان يصف الهرمية الديموغرافية وعلاقتها بالمواقف
                   title='Demographics and Attitudes Hierarchy (Interactive)')
# ضبط الأبعاد لعرض المخطط الهرمي بشكل مريح وواضح
fig6.update_layout(height=700, width=700)
# حفظ الرسم كملف HTML تفاعلي
fig6.write_html(os.path.join(PLOTLY_DIR, 'interactive_demographics_sunburst.html'))
print(f"    ✓ Saved: interactive_demographics_sunburst.html")

# 7. Interactive Parallel Coordinates
print("  Creating parallel coordinates plot...")
# إعداد بيانات متعددة الأبعاد تمثل مؤشرات الثقة/القلق/المعرفة/المهارة،
# مع ترميز فئة الموقف إلى قيمة عددية لاستخدامها في التلوين
parallel_data = df[['trust_index', 'anxiety_index', 'knowledge_score', 
                     'ai_literacy_score', 'attitude_category']].dropna()
parallel_data['attitude_code'] = pd.Categorical(parallel_data['attitude_category']).codes

fig7 = px.parallel_coordinates(parallel_data,
                               # dimensions: قائمة الأبعاد التي ستظهر كمحاور متوازية
                               dimensions=['trust_index', 'anxiety_index', 
                                         'knowledge_score', 'ai_literacy_score'],
                               # color: تلوين الخطوط بناء على الكود العددي لفئة الموقف
                               color='attitude_code',
                               # labels: تسمية المحاور بأسماء مختصرة وواضحة
                               labels={'trust_index': 'Trust',
                                      'anxiety_index': 'Anxiety',
                                      'knowledge_score': 'Knowledge',
                                      'ai_literacy_score': 'Literacy'},
                               # title: عنوان يصف الملف الشخصي الإدراكي عبر عدة مؤشرات
                               title='Parallel Coordinates: AI Perception Profiles')
# ضبط أبعاد الشكل لعرض عدد كبير من الخطوط دون ازدحام
fig7.update_layout(height=600, width=1000)
# حفظ الرسم كملف HTML تفاعلي
fig7.write_html(os.path.join(PLOTLY_DIR, 'interactive_parallel_coordinates.html'))
print(f"    ✓ Saved: interactive_parallel_coordinates.html")

print(f"\n✓ All {7} interactive visualizations created in '{PLOTLY_DIR}' folder")

