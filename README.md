# Bias Checker Dashboard

This interactive Streamlit app allows users to upload a dataset and visualize sampling bias between population and selected samples. It provides statistical insights, side-by-side graphs, and downloadable reports to help validate fairness in data selection.

---

## Features

- Upload a `.csv` file and preview the dataset.
- Select sampling strategies:
  - Biased (lower half of a numeric feature)
  - Random sampling
  - Stratified sampling (by categorical column)
- Analyze:
  - **Numeric features** (histogram, KDE plot, KS Test)
  - **Categorical features** (bar plots, Chi-Square Test)
- Download a summary report with key metrics.
- Clean, side-by-side visual layout for better comparison.

---

##  Installation

1. Clone this repository or copy the code.
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```
4. Run the application:
 ``` bash
 streamlit run app.py
 ```

## Sampling Methods Explained

1. Min-Max Bias Sampling
- Selects only the lower half of the values from a numeric column.
- Useful for simulating biased or unbalanced data.

2. Random Sampling
- Randomly picks a sample of rows from the full dataset.
- Good for general testing or baseline comparisons.

3. Stratified Sampling
- Ensures each category (e.g., gender or department) is proportionally represented in the sample.
- Great for fairness-focused evaluations.

## Statistical Tests Used

- KS Test (Kolmogorov-Smirnov): Compares distributions of numeric values.
- Chi-Square Test: Checks frequency distribution differences between population and sample.

## Output
 Displays sample vs population plots.
 Downloadable .txt report with key metrics and test results.

Feedback
For suggestions or improvements, feel free to open an issue or submit a pull request.
