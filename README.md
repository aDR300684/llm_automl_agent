# 🤖 LLM-powered AutoML Agent

This project defines an autonomous agent that uses GPT-4 to analyze a raw CSV dataset and build a complete ML classification pipeline with zero manual intervention.

### 🔁 End-to-End Workflow

1. **Load Dataset + Select Target Column**  
   - A user provides a local CSV file and specifies which column is the target to predict.

2. **Step 1 – Dataset Analysis via LLM**  
   - GPT-4 analyzes the structure of the dataset (columns, missing values, data types).
   - It generates a general-purpose preprocessing strategy suitable for supervised learning.

3. **Step 2 – Feature Engineering (Code Generation + Execution)**  
   - GPT-4 writes Python code (`feature_engineering.py`) to:
     - Drop irrelevant or ID-like columns
     - Create basic interactions and boolean/binned features
     - Handle missing values and infinities
   - The script is executed, producing a cleaned file: `dataset_FE.csv`.

4. **Step 3 – Training Pipeline (Code Generation + Execution)**  
   - GPT-4 generates a full scikit-learn pipeline (`generated_code.py`) that:
     - Implements preprocessing via `ColumnTransformer` and `Pipeline`
     - Trains a `RandomForestClassifier`
     - Outputs model performance (accuracy, classification report)
     - Prints top feature importances (sorted)

5. **Step 4 – Summary Generation**  
   - GPT-4 interprets the full training output and generates a plain-English summary:
     - Highlights the most important features
     - Removes technical prefixes and clarifies engineered features
     - Explains the model’s prediction logic clearly and generically

### ✅ Outputs
- Cleaned dataset: `dataset_FE.csv`  
- Saved model: `pipeline_model.pkl`  
- Generated Python scripts: `feature_engineering.py`, `generated_code.py`  
- Final human-readable summary: printed to console  
- Compatible with any tabular classification dataset  
