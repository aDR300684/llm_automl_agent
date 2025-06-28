# llm_agent.py

import textwrap
import os
import sys
import pandas as pd
from dotenv import load_dotenv
import subprocess
from openai import OpenAI
from prompts import (
    build_prompt_dataset_analysis,
    build_prompt_feature_engineering,
    build_prompt_generate_code,
    build_prompt_summarize_results
)

load_dotenv()
client = OpenAI()


def call_openai(prompt: str, model="gpt-4.1-2025-04-14", temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a helpful and precise ML assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.strip().splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def summarize_dataset(df: pd.DataFrame, target_col: str):
    n_rows, n_cols = df.shape
    column_names = df.columns.tolist()

    nan_summary = "\n".join([
        f"- {col}: {df[col].isna().sum()} missing ({df[col].isna().mean()*100:.1f}%)"
        for col in df.columns if df[col].isna().sum() > 0
    ]) or "None"

    type_groups = df.dtypes.groupby(df.dtypes).groups
    dtypes = "\n".join([
        f"- {dtype}: {list(cols)}" for dtype, cols in type_groups.items()
    ])

    return column_names, n_rows, n_cols, nan_summary, dtypes


def run_llm_pipeline(csv_path: str, target_col: str) -> str:
    log = []
    df = pd.read_csv(csv_path)
    column_names, n_rows, n_cols, nan_summary, dtypes = summarize_dataset(df, target_col)

    # Step 1: Strategy generation
    log.append("[LLM_AGENT] Step 1: Analyzing dataset...")
    prompt1 = build_prompt_dataset_analysis(column_names, n_rows, n_cols, nan_summary, dtypes, target_col)
    strategy_text = call_openai(prompt1)
    log.append("[LLM_AGENT] ✅ Strategy received:")
    log.append(strategy_text)

    # Step 2: Feature engineering
    log.append("\n[LLM_AGENT] Step 2: Generating feature engineering script...")
    fe_prompt = build_prompt_feature_engineering(column_names, target_col, csv_path)
    fe_code = strip_code_fences(call_openai(fe_prompt))

    with open("feature_engineering.py", "w", encoding="utf-8") as f:
        f.write(fe_code)

    log.append("[LLM_AGENT] ✅ Saved feature_engineering.py. Running it now...")

    try:
        fe_result = subprocess.run(
            [sys.executable, "feature_engineering.py"],
            capture_output=True,
            text=True,
            check=True
        )
        log.append("[LLM_AGENT] ✅ Feature engineering completed:")
        log.append(fe_result.stdout)
        if fe_result.stderr:
            log.append("[LLM_AGENT] ⚠️ FE Warnings:")
            log.append(fe_result.stderr)
    except subprocess.CalledProcessError as e:
        log.append("[LLM_AGENT] ❌ Feature engineering failed:")
        log.append(e.stdout)
        log.append(e.stderr)
        return "\n".join(log)

    # Step 3: Model code generation (now using dataset_FE.csv)
    log.append("\n[LLM_AGENT] Step 3: Generating training pipeline...")
    prompt3 = build_prompt_generate_code(strategy_text, target_col, "dataset_FE.csv")
    train_code = strip_code_fences(call_openai(prompt3))

    with open("generated_code.py", "w", encoding="utf-8") as f:
        f.write(train_code)

    # Append model saving line
    with open("generated_code.py", "a", encoding="utf-8") as f:
        f.write("\n\nimport joblib\njoblib.dump(clf, 'pipeline_model.pkl')")

    # Append actual vs predicted + feature importances (safe version, no pandas re-import)
    with open("generated_code.py", "a", encoding="utf-8") as f:
        f.write(textwrap.dedent("""
            # Postprocessing: Actual vs Predicted + Feature Importances

            # Show a preview of y_true vs y_pred
            preview = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred
            }).reset_index(drop=True)
            print("y_true vs y_pred (first 10 rows):")
            print(preview.head(10).to_string(index=False))

            # Feature importances if supported
            if hasattr(clf.named_steps["classifier"], "feature_importances_"):
                importances = clf.named_steps["classifier"].feature_importances_
                feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
                sorted_idx = np.argsort(importances)[::-1]
                print("\\nFeature Importances (sorted):")
                for idx in sorted_idx:
                    print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        """))

    log.append("[LLM_AGENT] ✅ Code saved to generated_code.py. Running it now...")

    try:
        train_result = subprocess.run(
            [sys.executable, "generated_code.py"],
            capture_output=True,
            text=True,
            check=True
        )
        log.append("[LLM_AGENT] ✅ Model training completed:")
        log.append(train_result.stdout)
        if train_result.stderr:
            log.append("[LLM_AGENT] ⚠️ Training Warnings:")
            log.append(train_result.stderr)
    except subprocess.CalledProcessError as e:
        log.append("[LLM_AGENT] ❌ Training failed:")
        log.append(e.stdout)
        log.append(e.stderr)
        return "\n".join(log)

    # Step 4: Generate final plain-language summary
    log.append("\n[LLM_AGENT] Step 4: Generating human-readable summary...")
    summary_prompt = build_prompt_summarize_results(train_result.stdout, target_col)
    final_summary = call_openai(summary_prompt)
    log.append("[LLM_AGENT] ✅ Summary:")
    log.append(final_summary)
               
    return "\n\n".join(log)