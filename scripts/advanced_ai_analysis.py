
# الغرض العام من هذا السكربت:
# هذا السكربت ينفّذ تحليلاً متقدماً لتأثيرات الذكاء الاصطناعي على المشاركين في الدراسة.
# يوفر إعدادات المسارات، تحميل البيانات مع معالجة الترميزات، استخراج علاقات الارتباط،
# إجراء تحليل الفروق الديموغرافية، تنفيذ تجميع K-Means لتحديد الشخصيات، وإنشاء رسومات ثابتة،
# ثم حفظ نواتج بصيغة CSV وصور جاهزة للاستخدام في Power BI. التعليقات هنا تشرح كل سطر وخطوة بالتفصيل.
# افتراضات البيانات المستخدمة:
# - الأعمدة الرقمية بمقياس 0-10: trust_index, anxiety_index, knowledge_score, ai_literacy_score, tech_comfort_level
# - أعمدة فئوية للنمذجة الديموغرافية: What is your gender?, What is your age range?, What is your education level?, What is your employment status?
# - قد توجد أعمدة إضافية مثل trust_score؛ يتم التحقق من وجود الأعمدة قبل استخدامها.
# - أي قيم مفقودة تُستبعد من عمليات تتطلب اكتمال البيانات (مثل التجميع).
# مخرجات السكربت:
# - CSV جاهزة لـ Power BI: correlation_analysis_powerbi.csv, gender_impact_summary.csv, cluster_profiles.csv, user_clusters.csv
# - صور في OUTPUT_DIR: correlation_heatmap.png, trust_vs_anxiety.png
# كيفية التشغيل:
# - تأكد من وجود ملف البيانات في أحد المسارات المتوقعة أو عدّل DATA_FILE يدوياً.
# - شغّل السكربت مباشرة ليتم حفظ النتائج تلقائياً في المجلد المحدد.

import pandas as pd
# استيراد مكتبة pandas للتعامل مع جداول البيانات وقراءة/كتابة CSV
import numpy as np
# استيراد NumPy للعمليات العددية والمصفوفات
import matplotlib.pyplot as plt
# استيراد matplotlib لإنشاء الرسوم البيانية الثابتة
import seaborn as sns
# استيراد seaborn لتحسين مظهر الرسومات وتقديم مخططات إحصائية
from scipy import stats
# استيراد أدوات إحصائية مثل t-test من SciPy
import os
# التعامل مع نظام الملفات والمسارات
import sys
# معلومات عن النظام وبيئة التشغيل

# Set plotting style
# ضبط نمط الرسوم البيانية ليكون متناسقاً وسهل القراءة
plt.style.use('ggplot')
# استخدام نمط ggplot المعروف بألوانه وتخطيطه المناسب
sns.set_theme(style="whitegrid")
# تعيين نمط Seaborn بخلفية شبكية بيضاء لتحسين القراءة

# --- Configuration ---
# Smart path detection - works whether run from main folder or scripts folder
# كشف المسار الذكي: يحدد موقع ملف البيانات ومجلد الإخراج بحسب مكان التشغيل
if os.path.exists('data/data_final_with_derived_columns.csv'):
    DATA_FILE = 'data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs/08_powerbi_export'
elif os.path.exists('../data/data_final_with_derived_columns.csv'):
    DATA_FILE = '../data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = '../outputs/08_powerbi_export'
elif os.path.exists('outputs/data_final_with_derived_columns.csv'):
    DATA_FILE = 'outputs/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs/08_powerbi_export'
elif os.path.exists('../outputs/data_final_with_derived_columns.csv'):
    DATA_FILE = '../outputs/data_final_with_derived_columns.csv'
    OUTPUT_DIR = '../outputs/08_powerbi_export'
else:
    # Fallback for Power BI or other contexts
    DATA_FILE = 'data/data_final_with_derived_columns.csv'
    OUTPUT_DIR = 'outputs/08_powerbi_export'

# دالة: تحميل البيانات من ملف CSV مع معالجة مشاكل الترميز إن ظهرت
def load_data(filepath):
    """
    Loads the dataset. Handles potential encoding issues.
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    return df

# دالة: حفظ أي DataFrame بصيغة CSV في مجلد الإخراج ليُستخدم في Power BI
def save_for_power_bi(df, filename):
    """
    Saves a DataFrame to CSV in the output directory for Power BI ingestion.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel/Power BI compatibility
    print(f"Saved {filename} for Power BI.")

# دالة: تحليل التوزيع الديموغرافي (مثل الفرق في الثقة بين الجنسين) وإرجاع ملخصات
def analyze_demographics(df):
    """
    Analyzes demographic distributions.
    """
    # 1. Gender vs Trust (T-Test)
    males = df[df['What is your gender?'] == 'Male']['trust_index']
    females = df[df['What is your gender?'] == 'Female']['trust_index']
    
    t_stat, p_val = stats.ttest_ind(males, females, nan_policy='omit')
    
    gender_stats = df.groupby('What is your gender?')['trust_index'].agg(['mean', 'count', 'std']).reset_index()
    gender_stats['p_value_vs_others'] = p_val # Simplified
    
    return gender_stats

# دالة: حساب مصفوفة الارتباطات بين المتغيرات الرقمية وإرجاع نسخة مهيأة لـ Power BI
def analyze_correlations(df):
    """
    Calculates correlations between numerical scores.
    """
    cols = [
        'knowledge_score', 'trust_score', 'anxiety_index', 
        'trust_index', 'ai_literacy_score', 'tech_comfort_level'
    ]
    # Filter only columns that exist
    valid_cols = [c for c in cols if c in df.columns]
    
    corr_matrix = df[valid_cols].corr()
    
    # Unpivot for Power BI (Source, Target, Correlation)
    corr_unpivoted = corr_matrix.stack().reset_index()
    corr_unpivoted.columns = ['Variable_1', 'Variable_2', 'Correlation']
    
    return corr_matrix, corr_unpivoted

# دالة: تنفيذ تجميع K-Means لتحديد شخصيات المستخدمين بناءً على السمات المحددة
def perform_clustering(df):
    """
    Performs K-Means clustering to identify user personas.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # السمات المستخدمة في التجميع: الثقة، القلق، المعرفة
        features = ['trust_index', 'anxiety_index', 'knowledge_score']
        # Drop rows with missing values for clustering
        # حذف الصفوف ذات القيم المفقودة لتحسين جودة التجميع
        df_cluster = df[features].dropna()
        
        # تقييس السمات لتكون بمقياس موحّد (متوسط 0، انحراف معياري 1)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_cluster)
        
        # Assume 3 clusters (e.g., Optimists, Skeptics, Neutrals)
        # افتراض وجود 3 مجموعات (متفائلون، مشككون، محايدون)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # إضافة رقم المجموعة لكل صف بعد التجميع
        df_cluster['Cluster'] = clusters
        
        # Name the clusters based on centers
        # استرجاع مراكز المجموعات وإعادة تحويلها للمقياس الأصلي لفهم خصائصها
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        center_df = pd.DataFrame(centers, columns=features)
        center_df['Cluster'] = range(3)
        
        # Join back to original dataframe (simplified)
        # We return the centers and the assigned cluster for the rows we used
        return df_cluster, center_df
        
    except ImportError:
        print("scikit-learn not installed. Skipping clustering.")
        return None, None

# دالة: إنشاء الرسومات البيانية (خريطة حرارية ومبعثر) وحفظها كصور
def generate_visualizations(df, corr_matrix):
    """
    Generates and saves static visualizations.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Correlation Heatmap
    # خريطة حرارية لمصفوفة الارتباطات لإظهار قوة واتجاه العلاقة بين المتغيرات
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of AI Perception Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    plt.close()

    # 2. Trust vs Anxiety Scatter
    # رسم مبعثر يوضح العلاقة بين القلق والثقة، ملوّن بحسب الجنس لسهولة المقارنة
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='anxiety_index', y='trust_index', hue='What is your gender?', alpha=0.7)
    plt.title('Trust vs Anxiety by Gender')
    plt.savefig(os.path.join(OUTPUT_DIR, 'trust_vs_anxiety.png'))
    plt.close()

# الدالة الرئيسية: تنسّق جميع خطوات التحليل من التحقق من الملف وحتى الحفظ والرسومات
def main():
    print("--- Starting Advanced AI Impact Analysis ---")
    
    # Check if file exists
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    # تحميل البيانات إلى DataFrame مع معالجة الترميز إن لزم
    df = load_data(DATA_FILE)
    print(f"Loaded data: {df.shape}")

    # --- 1. Correlation Analysis ---
    # حساب الارتباطات بين المتغيرات الرقمية وحفظ جدول مهيأ لـ Power BI
    corr_matrix, corr_powerbi = analyze_correlations(df)
    save_for_power_bi(corr_powerbi, 'correlation_analysis_powerbi.csv')
    print("Correlation analysis done.")

    # --- 2. Demographic Analysis ---
    # تحليل الفروق الديموغرافية (الجنس مقابل الثقة) وحفظ النتائج
    gender_impact = analyze_demographics(df)
    save_for_power_bi(gender_impact, 'gender_impact_summary.csv')
    print("Demographic analysis done.")

    # --- 3. Advanced Clustering (Personas) ---
    # تنفيذ التجميع للحصول على شخصيات المستخدمين وحفظ خصائص المجموعات والصفوف المصنفة
    df_clustered, cluster_centers = perform_clustering(df)
    if df_clustered is not None:
        save_for_power_bi(cluster_centers, 'cluster_profiles.csv')
        # Merge cluster ID back to main data for export
        # Note: This only merges for rows that weren't dropped. 
        # Ideally, we'd fill missing before clustering or handle indices carefully.
        # For now, we export the subset.
        save_for_power_bi(df_clustered, 'user_clusters.csv')
        print("Clustering analysis done.")

    # --- 4. Visualizations ---
    # إنشاء الرسومات (الخريطة الحرارية والمبعثر) وحفظها
    generate_visualizations(df, corr_matrix)
    print("Visualizations saved.")

    print("\n--- Analysis Complete ---")
    print(f"Results are saved in the '{OUTPUT_DIR}' folder.")
    print("You can import these CSV files directly into Power BI Desktop.")

if __name__ == "__main__":
    # نقطة الدخول القياسية: تنفيذ الدالة الرئيسية عند تشغيل الملف مباشرة
    main()
