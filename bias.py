import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chisquare
from io import BytesIO
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bias Checker Validation Tool", layout="wide")
st.title("Dataset Bias Checker & Validation Report")

st.markdown("""
    <style>
    .dataframe th, .dataframe td {
        font-size: 12px;
        padding: 4px 6px;
    }
    .element-container:has(.stDataFrame) {
        max-width: 450px !important;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Dataset")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.sidebar.markdown("---")
    sampling_method = st.sidebar.selectbox("Select Sampling Method", ["Biased (Lower Half)", "Random", "Stratified by Category"])
    stratify_col = st.sidebar.selectbox("Stratify by (if applicable)", cat_cols) if sampling_method == "Stratified by Category" else None

    report_lines = []

    if numeric_cols and cat_cols:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Numeric Feature Bias Check")
            num_col = st.selectbox("Select a numeric column", numeric_cols)

            pop_mean = df[num_col].mean()
            pop_median = df[num_col].median()
            pop_std = df[num_col].std()

            if sampling_method == "Biased (Lower Half)":
                sample = df[df[num_col] < df[num_col].quantile(0.5)].sample(n=200 if len(df) > 200 else int(len(df)*0.2), random_state=42)
            elif sampling_method == "Random":
                sample = df.sample(n=200 if len(df) > 200 else int(len(df)*0.2), random_state=42)
            elif sampling_method == "Stratified by Category" and stratify_col:
                _, sample = train_test_split(df, test_size=0.2, stratify=df[stratify_col], random_state=42)
            else:
                sample = df.sample(n=200 if len(df) > 200 else int(len(df)*0.2), random_state=42)

            sample_mean = sample[num_col].mean()
            sample_median = sample[num_col].median()
            sample_std = sample[num_col].std()

            ks_result = ks_2samp(df[num_col].dropna(), sample[num_col].dropna())

            summary_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Standard Deviation', 'KS Test p-value'],
                'Population': [round(pop_mean,2), round(pop_median,2), round(pop_std,2), ''],
                'Sample': [round(sample_mean,2), round(sample_median,2), round(sample_std,2), round(ks_result.pvalue, 4)]
            })
            st.dataframe(summary_df)

            fig, ax = plt.subplots(figsize=(4, 2))
            sns.histplot(df[num_col], label='Population', color='#63993D', kde=True, stat="density", bins=25, alpha=0.6)
            sns.histplot(sample[num_col], label='Sample', color='#ede15b', kde=True, stat="density", bins=25, alpha=0.6)
            ax.set_title(f'{num_col}: Population vs Sample', fontsize=10)
            ax.set_xlabel(num_col)
            ax.set_ylabel('Density')
            ax.legend()
            st.pyplot(fig)

            report_lines.append(f"Numeric Feature: {num_col}\n")
            report_lines.append(f"Population Mean: {pop_mean:.2f}, Sample Mean: {sample_mean:.2f}\n")
            report_lines.append(f"Population Median: {pop_median:.2f}, Sample Median: {sample_median:.2f}\n")
            report_lines.append(f"KS Test p-value: {ks_result.pvalue:.4f}\n")
            report_lines.append("\n")

        with col2:
            st.markdown("### Categorical Feature Bias Check")
            cat_col = st.selectbox("Select a categorical column", cat_cols)

            pop_counts = df[cat_col].value_counts()
            sample_counts = sample[cat_col].value_counts().reindex(pop_counts.index).fillna(0)

            expected = (pop_counts / pop_counts.sum()) * sample_counts.sum()
            chi_result = chisquare(f_obs=sample_counts, f_exp=expected)

            st.write(f"**Chi-Square Test p-value:** {round(chi_result.pvalue, 4)}")

            fig2, ax2 = plt.subplots(figsize=(4, 2))
            width = 0.35
            index = np.arange(len(pop_counts))
            ax2.bar(index - width/2, pop_counts.values, width, label='Population', color='#63993D')
            ax2.bar(index + width/2, sample_counts.values, width, label='Sample', color='#ede15b')
            ax2.set_title(f"{cat_col}: Population vs Sample", fontsize=10)
            ax2.set_ylabel("Count")
            ax2.set_xticks(index)
            ax2.set_xticklabels(pop_counts.index, rotation=30)
            ax2.legend()
            st.pyplot(fig2)

            report_lines.append(f"Categorical Feature: {cat_col}\n")
            report_lines.append(f"Chi-Square Test p-value: {chi_result.pvalue:.4f}\n")
            report_lines.append("\n")

    if report_lines:
        report_text = "".join(report_lines)
        report_bytes = BytesIO()
        report_bytes.write(report_text.encode())
        report_bytes.seek(0)
        st.download_button("Download Bias Report", data=report_bytes, file_name="bias_report.txt", mime="text/plain")

else:
    st.info("Please upload a CSV file to begin.")