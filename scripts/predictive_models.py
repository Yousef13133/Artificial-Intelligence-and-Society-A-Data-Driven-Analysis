import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

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

PREDICT_DIR = os.path.join(OUTPUT_DIR, '06_predictive_models')
os.makedirs(PREDICT_DIR, exist_ok=True)

print("PREDICTIVE MODELS")
df = pd.read_csv(DATA_FILE, encoding='utf-8')

print("Linear Regression: Predicting Trust Index")
X = df[['knowledge_score', 'ai_literacy_score', 'tech_comfort_level', 'device_hours_score', 'ai_usage_score']].dropna()
y = df.loc[X.index, 'trust_index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R2={r2:.4f}, RMSE={rmse:.4f}")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)
feature_importance.to_csv(os.path.join(PREDICT_DIR, 'feature_importance_trust.csv'), index=False)

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

print("Logistic Regression: Predicting Attitude Category")
X_class = df[['knowledge_score', 'ai_literacy_score', 'trust_index', 'anxiety_index']].dropna()
y_class = df.loc[X_class.index, 'attitude_category']
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded
)
log_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
log_model.fit(X_train_c, y_train_c)
y_pred_c = log_model.predict(X_test_c)
accuracy = (y_pred_c == y_test_c).mean()
print(f"Accuracy={accuracy:.4f}")

report = classification_report(y_test_c, y_pred_c, target_names=le.classes_)
with open(os.path.join(PREDICT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Attitude Category Prediction')
plt.tight_layout()
plt.savefig(os.path.join(PREDICT_DIR, 'confusion_matrix_attitude.png'), dpi=300)
plt.close()
