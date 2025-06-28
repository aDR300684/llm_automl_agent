# prompts.py

def build_prompt_dataset_analysis(column_names, n_rows, n_cols, nan_summary, dtypes, target_col):
    return f"""
You are a senior Machine Learning engineer. Analyze the following tabular dataset and propose a clean and general preprocessing strategy suitable for supervised learning.

--- DATASET OVERVIEW ---
- Number of rows: {n_rows}
- Number of columns: {n_cols}
- Target column to predict: **{target_col}**
- Column names: {', '.join(column_names)}
- Missing values:
{nan_summary}
- Data types:
{dtypes}

--- TASK ---
1. Identify common issues in tabular datasets such as:
   - Missing values
   - Mixed data types (categorical, numerical, text)
   - Identifier-like or high-cardinality columns (e.g., IDs, codes, free text)
   - Columns that are likely irrelevant or misleading for predicting `{target_col}`
   - Skewed numerical features or outliers

2. Propose a **general-purpose preprocessing pipeline** including:
   - Handling of missing values
   - Selection and encoding of features
   - Proper treatment of categorical and numerical columns
   - Scaling and transformation (if needed)
   - Outlier detection and correction (optional)
   - Any feature removal if it could hurt model performance (e.g.,Id columns or other irrelevant columns)

⚠️ Do NOT include any dataset-specific logic or domain knowledge.

Return a clean, structured, and readable preprocessing strategy. No code.
""".strip()

def build_prompt_feature_engineering(column_names, target_col, input_csv_path):
    return f"""
You are a senior ML engineer. Your task is to write a clean and runnable Python script that performs **general-purpose feature engineering** on a tabular dataset.

--- INPUT ---
- Input CSV path: {input_csv_path}
- Target column: '{target_col}'
- Input columns: {', '.join(column_names)}

--- GOALS ---
1. **Drop irrelevant columns**:
   - Identifier-like (e.g., IDs, codes, names, free-text)
   - Columns with more than 50% missing values
   - Columns with only 1 unique value

2. **Create new features**:
   - at most 3 meaningful interaction terms between numeric columns. Avoid noisy or redundant combinations."
   - Boolean flags from numeric columns (e.g., is zero)
   - Quantile binning of wide-range numeric columns
   - Count missing values per row (`MissingCount`)
   - Ensure no inf or -inf values: replace them with NaN after ratio computations

3. **Preserve original data types**:
   - Do NOT encode categorical variables
   - Do NOT impute missing values
   - Do NOT apply transformations (e.g., log, scale)

4. **Save output**:
   - Save the processed dataset to: `dataset_FE.csv` (in the same folder)
   - Clearly log all feature creation steps to the console

--- CONSTRAINTS ---
- Use only pandas and numpy (no scikit-learn required here)
- Do NOT use `sparse=False`. Always use `sparse_output=False` to ensure compatibility with modern scikit-learn.
- Do NOT use OneHotEncoder or imputation
- Do NOT apply transformations (e.g., log1p, StandardScaler, etc.)
- Do NOT include any markdown, explanations, or dataset-specific logic
- After feature creation, replace any `inf` or `-inf` values with `np.nan` using `df.replace([np.inf, -np.inf], np.nan, inplace=True)`

Return only runnable Python code.
""".strip()

def build_prompt_generate_code(strategy_description, target_col, csv_path):
    return f"""
You are a senior machine learning engineer assistant.

Your task is to generate a complete, clean, and executable Python script that implements the preprocessing and modeling pipeline described below. The output must be directly runnable end-to-end with no manual editing.

--- STRATEGY TO IMPLEMENT ---
{strategy_description}

--- DATASET DETAILS ---
- Input CSV path: dataset_FE.csv
- Target column: '{target_col}'

--- TECHNICAL REQUIREMENTS ---
- Use pandas, numpy, and scikit-learn (version ≥1.2)
- Use ColumnTransformer and Pipeline to modularize preprocessing
- ⚠️ Do NOT define any custom transformation functions (e.g., `def cap_outliers(...)` or `def handle_skew(...)`) outside of pipelines
- If outlier capping or skew handling is required, use only built-in NumPy or scikit-learn functions inside `FunctionTransformer`
- If only some numerical columns require transformation (e.g., skewed columns), split them into subgroups using logical column selections (e.g., by column index)
- ⚠️ When using `FunctionTransformer`, do NOT rely on column names inside the function unless explicitly passed. Use only NumPy operations on the input array
- ⚠️ Avoid lambda functions inside pipelines. Use only named NumPy functions (e.g., `np.log1p`, `np.clip`) or scikit-learn transformers
- For OneHotEncoder, use: `sparse_output=False` (not `sparse=False`)
- Include robust handling of missing values, skewed distributions, and column type inference
- Ensure the script splits the data, trains a `RandomForestClassifier`, and prints accuracy and classification report
- Name the final trained pipeline variable `clf`
- Train the model using `clf.fit(...)` on the training set
- The script must run end-to-end without any shape mismatches in transformers

--- OUTPUT FORMAT ---
- Return only valid, clean Python code (no explanations, no markdown, no backticks)
- The script must run end-to-end without any manual fix
""".strip()

def build_prompt_summarize_results(target_col, full_output):
    return f"""
You are a senior machine learning assistant. Your task is to summarize the output of an automated ML pipeline that has trained a classifier to predict the target column: **'{target_col}'**.

--- RAW OUTPUT ---
{full_output}

--- INSTRUCTIONS ---
- Analyze the classification output and feature importances.
- Identify the most important features that the model uses for prediction.
- Rephrase technical or transformed feature names into natural language:
  - Remove prefixes like "skewed_num__", "low_card_cat__", etc.
  - Clarify interaction terms (e.g., "X_x_Y" → "interaction between X and Y")
  - Reword binned or flagged features (e.g., suffixes like "_qbin", "_iszero") into simple descriptions.
- Do not assume any domain, dataset, or meaning for the target values.
- Write for a non-technical audience using clear, accessible language.
- End with a concise, 1–2 sentence summary of which original features are most important in predicting '{target_col}'.
""".strip()
