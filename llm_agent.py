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


def call_openai(prompt: str, model="gpt-4.1-2025-04-14", temperature=0.2):
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            timeout=60,  # Prevent freeze if API stalls
            messages=[
                {"role": "system", "content": "You are a helpful and precise ML assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] OpenAI API failed: {str(e)}"


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

    # Missing values
    nan_summary = "\n".join([
        f"- {col}: {df[col].isna().sum()} missing ({df[col].isna().mean()*100:.1f}%)"
        for col in df.columns if df[col].isna().sum() > 0
    ]) or "None"

    # Data types grouped
    type_groups = df.dtypes.groupby(df.dtypes).groups
    dtypes = "\n".join([
        f"- {dtype}: {list(cols)}" for dtype, cols in type_groups.items()
    ])

    # Unique value counts
    unique_counts = "\n".join([
        f"- {col}: {df[col].nunique()} unique"
        for col in df.columns if col != target_col
    ])

    # Optional: Example values for top 3 object columns
    example_values = "\n".join([
        f"- {col}: {df[col].dropna().unique()[:5].tolist()}"
        for col in df.select_dtypes(include="object").columns[:3]
    ]) or "None"

    return {
        "column_names": column_names,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nan_summary": nan_summary,
        "dtypes": dtypes,
        "unique_counts": unique_counts,
        "example_values": example_values
    }



def run_llm_pipeline(csv_path: str, target_col: str) -> str:
    log = []
    df = pd.read_csv(csv_path)
    summary = summarize_dataset(df, target_col)

    # Step 1: Strategy generation
    log.append("[LLM_AGENT] Step 1: Analyzing dataset...")
    prompt1 = build_prompt_dataset_analysis(summary, target_col)
    log.append("[LLM_AGENT] Step 1: Calling OpenAI to generate strategy...")
    strategy_text = call_openai(prompt1)

    if strategy_text.startswith("[ERROR]"):
        log.append(strategy_text)
        return "\n".join(log)

    log.append("[LLM_AGENT] ✅ Strategy received:")
    log.append(strategy_text)

    # Step 2: Feature engineering
    log.append("\n[LLM_AGENT] Step 2: Generating feature engineering script...")
    fe_prompt = build_prompt_feature_engineering(strategy_text, csv_path)
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

    # Step 3: Model code generation (dataset_FE.csv structure + strategy)
    log.append("\n[LLM_AGENT] Step 3: Summarizing dataset_FE.csv for modeling...")
    df_fe = pd.read_csv("dataset_FE.csv")
    summary_fe = summarize_dataset(df_fe, target_col)

    log.append("[LLM_AGENT] ✅ Summary of dataset_FE.csv completed.")

    log.append("[LLM_AGENT] Step 3: Generating modeling script based on FE dataset and strategy...")
    prompt3 = build_prompt_generate_code(
        target_col=target_col,
        csv_path="dataset_FE.csv",
        strategy_text=strategy_text,
            summary_dict=summary_fe
    )
    train_code = strip_code_fences(call_openai(prompt3))

    with open("generated_code.py", "w", encoding="utf-8") as f:
        f.write(train_code)


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