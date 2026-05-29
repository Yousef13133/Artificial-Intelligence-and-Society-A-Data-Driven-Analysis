import os
import sys
import pandas as pd
import plotly.express as px

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

PLOTLY_DIR = os.path.join(OUTPUT_DIR, '07_interactive_plots')
os.makedirs(PLOTLY_DIR, exist_ok=True)

print("INTERACTIVE PLOTS")
df = pd.read_csv(DATA_FILE, encoding='utf-8')

print("Scatter: Trust vs Anxiety")
fig1 = px.scatter(df, x='anxiety_index', y='trust_index',
                  color='What is your gender?',
                  size='knowledge_score',
                  hover_data=['What is your age range?', 'attitude_category', 'ai_literacy_score', 'tech_comfort_level'],
                  title='Trust vs Anxiety by Gender (Interactive)',
                  labels={'anxiety_index': 'Anxiety Index (0-10)', 'trust_index': 'Trust Index (0-10)'},
                  color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'})
fig1.update_layout(height=600, width=900)
fig1.write_html(os.path.join(PLOTLY_DIR, 'interactive_trust_vs_anxiety.html'))

print("Correlation Matrix")
corr_cols = ['trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score', 'tech_comfort_level', 'beneficial_score', 'ai_usage_score']
corr_matrix = df[corr_cols].corr()
fig2 = px.imshow(corr_matrix, text_auto='.3f', aspect="auto", color_continuous_scale='RdBu_r',
                 title='Interactive Correlation Matrix', labels=dict(color="Correlation"))
fig2.update_layout(height=700, width=800)
fig2.write_html(os.path.join(PLOTLY_DIR, 'interactive_correlation_heatmap.html'))

print("Box: Trust by Age and Gender")
fig3 = px.box(df, x='What is your age range?', y='trust_index',
              color='What is your gender?',
              title='Trust Index Distribution by Age Group and Gender',
              labels={'trust_index': 'Trust Index (0-10)', 'What is your age range?': 'Age Range'})
fig3.update_layout(height=600, width=1000)
fig3.write_html(os.path.join(PLOTLY_DIR, 'interactive_trust_by_age.html'))

print("Sunburst: Demographics and Attitudes")
sunburst_data = df[['What is your age range?', 'What is your gender?', 'attitude_category']].dropna()
sunburst_data['count'] = 1
fig6 = px.sunburst(sunburst_data, path=['What is your age range?', 'What is your gender?', 'attitude_category'],
                   values='count', title='Demographics and Attitudes Hierarchy (Interactive)')
fig6.update_layout(height=700, width=700)
fig6.write_html(os.path.join(PLOTLY_DIR, 'interactive_demographics_sunburst.html'))

print("Parallel Coordinates")
parallel_data = df[['trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score', 'attitude_category']].dropna()
parallel_data['attitude_code'] = pd.Categorical(parallel_data['attitude_category']).codes
fig7 = px.parallel_coordinates(parallel_data,
                               dimensions=['trust_index', 'anxiety_index', 'knowledge_score', 'ai_literacy_score'],
                               color='attitude_code',
                               labels={'trust_index': 'Trust','anxiety_index': 'Anxiety','knowledge_score': 'Knowledge','ai_literacy_score': 'Literacy'},
                               title='Parallel Coordinates: AI Perception Profiles')
fig7.update_layout(height=600, width=1000)
fig7.write_html(os.path.join(PLOTLY_DIR, 'interactive_parallel_coordinates.html'))

print("Done")
