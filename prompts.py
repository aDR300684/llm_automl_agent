# prompts.py

def build_prompt_dataset_analysis(summary, target_col):
    return f"""
You are a senior machine learning engineer. Based on the dataset summary below, propose a preprocessing strategy that is tailored to this dataset but follows general best practices for supervised learning on tabular data.

--- DATASET OVERVIEW ---
- Number of rows: {summary['n_rows']}
- Number of columns: {summary['n_cols']}
- Target column: {target_col}
- Column names: {', '.join(summary['column_names'])}

Missing values:
{summary['nan_summary']}

Data types:
{summary['dtypes']}

Unique value counts:
{summary['unique_counts']}

Example values from selected object columns:
{summary['example_values']}

--- TASK ---
1. Identify common data quality issues such as:
   - Missing values
   - Mixed data types (categorical, numerical, text)
   - High-cardinality or identifier-like columns (e.g. IDs, codes)
   - Irrelevant or misleading features
   - Skewed distributions or outliers

2. Propose a clean, structured preprocessing strategy that includes:
   - Feature selection and handling of irrelevant columns
   - Missing value treatment : if any numerical column has missing values, propose grouped imputation when group context is predictive. Avoid default global imputation in such cases.
   - Encoding for categorical variables
   - Scaling for numerical variables
   - Transformation for skewed data (if needed)
   - Handling of outliers
   - Encoding of the target variable (if needed)
   - Feature engineering based on available columns

3. For categorical variables with many unique values, suggest general strategies.

⚠️ At the end of preprocessing, there must be no remaining missing values in any categorical or numerical column used for modeling. All imputations must be explicitly applied.

Important constraints:
- Base your strategy on the dataset details provided above.
- Refer to specific columns when proposing preprocessing steps.
- DO NOT use vague phrases like “if used”, “optionally”, “consider”, or “you may”.
- Avoid long prose : give clear and concise instructions.
- Make final decisions. For each column, clearly state whether to drop, transform, or keep.
- This strategy will be used to auto-generate executable code, so clarity and determinism are critical.
- Do not generate Python code. Return only a structured and readable strategy description.

""".strip()

def build_prompt_feature_engineering(strategy_text: str, input_csv_path: str) -> str:
    return f"""
You are a senior ML engineer. Based on the following preprocessing strategy, write a Python script that performs the recommended preproccessing and feature engineering steps on the dataset.

--- INPUT ---
- Dataset CSV path: {input_csv_path}

--- STRATEGY TO IMPLEMENT ---
{strategy_text}

--- GOALS ---
1. **Drop irrelevant columns**:
   - Identifier-like (e.g., IDs, codes, names, free-text)
   - Columns with more than 50% missing values
   - Columns with only 1 unique value

2. **Create new features**:
   - at most 3 meaningful interaction terms between numeric columns. Avoid noisy or redundant combinations."
   - Quantile binning of wide-range numeric columns
   - Count missing values per row (`MissingCount`)
   - Avoid generating features that are constant or mostly zero
   - Ensure no inf or -inf values: replace them with NaN after ratio computations

3. **Preserve original data types**:
   - Do NOT use the target column in any transformations or new features. No leakage allowed.
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
- After all transformations, replace any `inf` or `-inf` values with `np.nan` using:  
  `df = df.replace([np.inf, -np.inf], np.nan)`.
- Return only runnable Python code.

""".strip()


def build_prompt_generate_code(target_col, csv_path, strategy_text, summary_dict):
    return f"""
You are a senior machine learning engineer assistant.

Your task is to generate a complete and clean Python script that performs **only the remaining preprocessing needed for modeling**, using the dataset that already includes manually engineered features and cleaned columns.

- Strategy was generated earlier
- Feature engineering script was applied next, creating derived features
- Now: You MUST fully implement the imputation, encoding, and transformations exactly as specified in the strategy below
- Do not fall back to default SimpleImputer if a grouped logic was specified.

--- PRIOR STRATEGY ---
{strategy_text}

--- CURRENT DATASET SUMMARY ({csv_path}) ---
- Target column: '{target_col}'
- Number of rows: {summary_dict['n_rows']}
- Number of columns: {summary_dict['n_cols']}
- Column names: {', '.join(summary_dict['column_names'])}

Missing values:
{summary_dict['nan_summary']}

Data types:
{summary_dict['dtypes']}

Unique value counts:
{summary_dict['unique_counts']}

Example object values:
{summary_dict['example_values']}

--- OBJECTIVE ---
- Evaluate multiple classification models using LazyPredict
- Identify the best-performing **actual classifier model** (by default: highest accuracy on test set)
- After fitting LazyClassifier, print the full `results_df` showing all tested models, sorted by accuracy (or f1).
- Retrain the top model on the same train/test split
- Report its accuracy, classification report, and actual vs predicted values (head of DataFrame)
- If the top model is linear (e.g. LogisticRegression, LinearSVC), print the absolute value of model.coef_ as a proxy for feature importance.
- Always print top 10 feature importances using either `feature_importances_` or `coef_` (if available).
- When printing feature importances, always check whether the model has `.feature_importances_` or `.coef_` using `hasattr()` before accessing them.
- Only print importances if one of those attributes exists. Otherwise, print `"Feature importances not available for this model."`
- Always use `X_train.columns` to get `feature_names`, and avoid hardcoded indexing that could cause an `IndexError`..

--- TECHNICAL REQUIREMENTS ---
- Use ColumnTransformer and Pipeline to modularize preprocessing
- ⚠️ If grouped imputation is required (e.g., median by group), it must be implemented using pandas logic **before** any pipeline or ColumnTransformer.
DO NOT fallback to SimpleImputer or generic logic.
This grouped imputation MUST be performed as a preprocessing step using DataFrame manipulation.
- ⚠️ Do NOT define any custom transformation functions (e.g., `def cap_outliers(...)` or `def handle_skew(...)`) outside of pipelines
- To retrieve final feature names, loop over preprocessor.transformers_, which yields (name, transformer, columns) tuples (for name, transformer, columns in preprocessor.transformers_:
    # logic to extract feature names per transformer)
- If outlier capping or skew handling is required, use only built-in NumPy or scikit-learn functions inside `FunctionTransformer`
- If only some numerical columns require transformation (e.g., skewed columns), split them into subgroups using logical column selections (e.g., by column index)
- ⚠️ When using FunctionTransformer, do NOT rely on column names inside the function unless explicitly passed. Use only NumPy operations on the input array
- ⚠️ If the input to a FunctionTransformer is a DataFrame, and column access is needed (e.g., for binary encoding), use `X.iloc[:, 0]` instead of `.ravel()` or `.values.ravel()`, which may break. Always return a reshaped NumPy array with `.to_numpy().reshape(-1, 1)` if needed.
- ⚠️ Avoid lambda functions inside pipelines. Use only named NumPy functions (e.g., `np.log1p`, `np.clip`) or scikit-learn transformers
- For OneHotEncoder, use: `sparse_output=False` (not `sparse=False`)
- Include robust handling of missing values, skewed distributions, and column type inference
- Use pandas, numpy, scikit-learn, and lazypredict
- Use train_test_split from scikit-learn (80/20 split, random_state=42)
- Ensure that target leakage is avoided throughout. Never use the target column in preprocessing steps.
- Ensure LazyPredict `LazyClassifier` uses `verbose=0` and `ignore_warnings=True`
- ⚠️ Do NOT select models like `Pipeline()` or `DummyClassifier()` from LazyPredict results – only select real classifiers like RandomForest, XGBoost, etc.
- After LazyPredict, retrain the **top real classifier model class** using the same data split
- Print classification accuracy, classification report, and head of a DataFrame showing actual vs predicted labels
- If applicable, print top 10 most important features with their importance scores
- The script must run end-to-end without any shape mismatches in transformers

--- MODEL SAFETY REQUIREMENTS ---
If the selected top model from LazyPredict is not available in your `models` list or causes any error:
- Print a warning: `[INFO] Model 'top_model_name' not supported. Falling back to RidgeClassifier.`
- Fallback to RidgeClassifier and continue the training pipeline
- Do NOT raise errors or halt execution
- Only allow the following model classes for retraining:
  - RidgeClassifier
  - LogisticRegression
  - RandomForestClassifier
  - XGBClassifier
  - LGBMClassifier

--- IMPLEMENTATION TIP ---
When looping through models, always define the list as tuples of the form (name, estimator), and use:
for name, est in models:
    if name == top_model_name:
        ...

--- OUTPUT FORMAT ---
- Output must be clean Python code only (no markdown, no explanations)
- Must be runnable end-to-end with no manual edits
""".strip()


def build_prompt_summarize_results(target_col, full_output):
    return f"""
You are a senior machine learning assistant. Your task is to summarize the output of an automated ML pipeline that has trained a classifier to predict the target column: **'{target_col}'**.

--- RAW OUTPUT ---
{full_output}

--- INSTRUCTIONS ---
1. Focus only on the final selected and retrained model. Ignore fallback notes, warnings, or LazyPredict baseline models like DummyClassifier.
2. Identify the most important features used by that model (based on printed feature importances or coefficients).
3. Rephrase technical feature names into natural language:
   - Remove prefixes like 'skewed_num__', 'low_card_cat__', etc.
   - Clarify interaction terms (e.g., 'X_x_Y' → 'interaction between X and Y')
   - Reword binned or flagged features (e.g., '_qbin', '_iszero') into readable phrases
4. Try to infer **directional influence** of each top feature, if possible:
   - Does a higher or lower value increase the predicted likelihood of the positive class?
   - Base this only on visible `coef_` or `feature_importances_` values (if available)
5. Ignore irrelevant output sections (e.g., raw training logs, fallback warnings, or diagnostics not related to the final model)
6. Write clearly for a non-technical audience — your summary should be intuitive and easy to understand.
7. End with a concise summary (1–2 sentences) describing which types of features most strongly influence predictions for '{target_col}'.
""".strip()