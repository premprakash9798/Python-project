"""" Student Academic Performance Analysis System
 + From: Minor 1 – Python, Pandas, EDA
 + Extension: Analyze marks dataset → clean data → visualize subject-wise performance → identify weak areas.
 + Add-ons: Seaborn heatmap, statistical summary, Streamlit dashboard.
 + Difficulty: Medium–Medium
"""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# --- 1. DATA GENERATION & LOADING ---
@st.cache_data
def load_data():
    """
    Generates a dummy dataset simulating student marks.
    In a real scenario, you would replace this with pd.read_csv('marks.csv').
    """
    np.random.seed(42)
    num_students = 100
    data = {
        'Student_ID': [f'STU{i:03d}' for i in range(1, num_students + 1)],
        'Gender': np.random.choice(['Male', 'Female'], num_students),
        'Math': np.random.randint(35, 100, num_students),
        'Science': np.random.randint(30, 95, num_students),
        'English': np.random.randint(40, 98, num_students),
        'History': np.random.randint(30, 90, num_students),
        'Attendance_Pct': np.random.randint(60, 100, num_students)
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing data to simulate "Cleaning" requirement
    df.loc[5:8, 'Math'] = np.nan 
    
    return df

# --- 2. DATA CLEANING & PROCESSING ---
def process_data(df):
    # Fill missing values with the median of the column
    df['Math'] = df['Math'].fillna(df['Math'].median())
    
    # Calculate Total and Average
    subjects = ['Math', 'Science', 'English', 'History']
    df['Total_Marks'] = df[subjects].sum(axis=1)
    df['Average_Score'] = df[subjects].mean(axis=1)
    
    # Assign Grades
    def get_grade(avg):
        if avg >= 90: return 'A'
        elif avg >= 75: return 'B'
        elif avg >= 50: return 'C'
        else: return 'D'
    
    df['Grade'] = df['Average_Score'].apply(get_grade)
    return df, subjects

# --- MAIN APP FLOW ---

# Load and Process Data
raw_df = load_data()
df, subject_list = process_data(raw_df)

# --- DASHBOARD UI ---
st.title("📊 Student Academic Performance Analysis System")
st.markdown("Analyze marks, visualize trends, and identify weak areas.")

# Sidebar Filters
st.sidebar.header("Filters")
gender_filter = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
grade_filter = st.sidebar.multiselect("Select Grade", options=sorted(df['Grade'].unique()), default=sorted(df['Grade'].unique()))

# Apply Filters
filtered_df = df[(df['Gender'].isin(gender_filter)) & (df['Grade'].isin(grade_filter))]

# --- SECTION 1: DATASET OVERVIEW ---
with st.expander("📂 View Raw Dataset & Stats"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Student Data")
        st.dataframe(filtered_df.style.highlight_max(axis=0, color='lightgreen'))
    with col2:
        st.subheader("Statistical Summary")
        st.write(filtered_df[subject_list].describe())

# --- SECTION 2: SUBJECT-WISE PERFORMANCE ---
st.divider()
st.header("📈 Subject-Wise Performance Analysis")

col1, col2 = st.columns(2)

# Visualization 1: Average Marks per Subject (Bar Chart)
with col1:
    st.subheader("Average Marks by Subject")
    avg_marks = filtered_df[subject_list].mean().reset_index()
    avg_marks.columns = ['Subject', 'Average']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Subject', y='Average', data=avg_marks, palette='viridis', ax=ax)
    ax.set_ylim(0, 100)
    st.pyplot(fig)

# Visualization 2: Correlation Heatmap
with col2:
    st.subheader("Subject Correlation Heatmap")
    st.caption("Do students who are good at Math also score well in Science?")
    corr = filtered_df[subject_list].corr()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- SECTION 3: WEAK AREA IDENTIFICATION ---
st.divider()
st.header("⚠️ Identify Weak Areas")

threshold = st.slider("Select Failing Threshold (Marks)", 0, 100, 40)

# Logic to find students struggling
weak_students = []
for index, row in filtered_df.iterrows():
    failed_subjects = []
    for sub in subject_list:
        if row[sub] < threshold:
            failed_subjects.append(sub)
    
    if failed_subjects:
        weak_students.append({
            'Student_ID': row['Student_ID'],
            'Overall_Avg': row['Average_Score'],
            'Failed_Subjects': ", ".join(failed_subjects),
            'Count': len(failed_subjects)
        })

weak_df = pd.DataFrame(weak_students)

if not weak_df.empty:
    st.error(f"Found {len(weak_df)} students scoring below {threshold} in at least one subject.")
    st.dataframe(weak_df.sort_values(by='Count', ascending=False))
else:
    st.success(f"No students found scoring below {threshold} marks!")

# --- SECTION 4: COMPARATIVE ANALYSIS ---
st.divider()
st.subheader("Gender Performance Comparison")
fig, ax = plt.subplots(figsize=(8, 4))
df_melted = filtered_df.melt(id_vars=['Gender'], value_vars=subject_list, var_name='Subject', value_name='Score')
sns.boxplot(x='Subject', y='Score', hue='Gender', data=df_melted, ax=ax)
st.pyplot(fig)