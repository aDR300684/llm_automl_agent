# 🤖 LLM-powered AutoML Agent

This project defines a fully autonomous agent that takes a raw CSV file and a target column name, then automatically builds a complete ML classification pipeline using GPT-4. The workflow consists of:

1. **Dataset Analysis:** Uses the OpenAI API to detect data types, missing values, and potential preprocessing strategies.
2. **Feature Engineering:** Dynamically generates a Python script that cleans the data, encodes categorical variables, handles skewness, and creates interaction terms.
3. **Pipeline Generation:** Creates a training pipeline with scikit-learn (e.g., preprocessing + classifier), saves the model, evaluates performance, and prints accuracy, classification report, and feature importances.
4. **Summary Generation:** Produces a plain-language explanation of the most important features influencing the prediction — without assuming any domain-specific knowledge.

All logic is LLM-driven, adaptable to any tabular classification dataset, and executes end-to-end without manual intervention. Outputs include the transformed dataset, saved model, full logs, and an interpretable model summary.
