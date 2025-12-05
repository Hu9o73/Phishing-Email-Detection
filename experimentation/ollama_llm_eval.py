#!/usr/bin/env python3
"""Run a lightweight DeepSeek model via Ollama on a balanced phishing/ham sample."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple

import json
import kagglehub
import pandas as pd
from tqdm import tqdm


DEFAULT_MODEL = "deepseek-r1:1.5b"
TARGET_ROWS = 10_000
RESULTS_PATH = Path(__file__).resolve().parent / "llm_results.csv"


def is_phishing_label(value) -> bool:
    """Detect phishing labels across common formats (0/1, strings, booleans)."""
    if pd.isna(value):
        return False

    if isinstance(value, (int, float)):
        return value > 0

    val = str(value).strip().lower()
    phishing_tokens = {"phishing", "fraud", "spam", "malicious", "1", "true", "yes"}
    ham_tokens = {"ham", "legit", "legitimate", "safe", "0", "false", "no"}

    if val in phishing_tokens or "phish" in val:
        return True
    if val in ham_tokens:
        return False
    return False


def download_dataset(local_csv: Path | None = None) -> Path:
    """Return a CSV path, preferring a user-supplied file and falling back to Kaggle."""
    if local_csv:
        if not local_csv.exists():
            raise RuntimeError(f"Provided CSV does not exist: {local_csv}")
        return local_csv.resolve()

    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    if not creds_path.exists():
        raise RuntimeError(
            "Kaggle credentials missing. Create ~/.kaggle/kaggle.json with your username/key and chmod 600."
        )
    try:
        json.loads(creds_path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to read Kaggle creds at {creds_path}: {exc}") from exc

    bar = tqdm(total=1, desc="Downloading dataset", unit="file", ncols=100, leave=False)
    try:
        file_path = kagglehub.dataset_download("advaithsrao/enron-fraud-email-dataset")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Kaggle download failed (403 usually means terms not accepted). "
            "Ensure: 1) you clicked 'Download/Accept' on the dataset page while logged in, "
            "2) ~/.kaggle/kaggle.json exists with username/key and perms 600, "
            "3) ~/.kaggle has perms 700, "
            "or pass --csv-path to use a local CSV."
        ) from exc
    finally:
        bar.update(1)
        bar.close()

    return Path(file_path) / "enron_data_fraud_labeled.csv"


def load_sample(csv_path: Path, target_rows: int, seed: int) -> Tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(csv_path)
    if "Label" not in df.columns:
        raise ValueError("Dataset must contain a 'Label' column")

    phishing_df = df[df["Label"].apply(is_phishing_label)]
    ham_df = df[~df["Label"].apply(is_phishing_label)]

    normal_needed = max(0, target_rows - len(phishing_df))
    normal_needed = min(normal_needed, len(ham_df))
    ham_sample = ham_df.sample(normal_needed, random_state=seed) if normal_needed else ham_df.iloc[0:0]

    combined = pd.concat([phishing_df, ham_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(
        f"Prepared sample with {len(phishing_df)} phishing and {len(ham_sample)} normal emails (total={len(combined)})."
    )
    if len(combined) < target_rows:
        print(f"Warning: only {len(combined)} rows available (requested {target_rows}).")

    needed_cols = ["Subject", "Body", "X-From", "X-To", "Label"]
    for col in needed_cols:
        if col not in combined.columns:
            combined[col] = ""
    return combined[needed_cols], len(phishing_df), len(ham_sample)


def pull_model(model: str) -> None:
    print(f"Pulling model '{model}' via Ollama (this may take a while)...")
    bar = tqdm(total=1, desc="Downloading model", ncols=100, unit="model", leave=False)
    try:
        subprocess.run(["ollama", "pull", model], check=True)
    finally:
        bar.update(1)
        bar.close()


def remove_model(model: str) -> None:
    print(f"Removing model '{model}' from Ollama cache...")
    subprocess.run(["ollama", "rm", model], check=False)


def build_prompt(subject: str, sender: str, recipient: str, body: str) -> str:
    body_short = body.strip()[:800]
    return (
        "You are a security classifier. Decide if this email is phishing or legitimate. "
        "Answer with a single word: 'phishing' or 'legitimate'.\n"
        f"Subject: {subject}\n"
        f"From: {sender}\n"
        f"To: {recipient}\n"
        "Body:\n"
        f"{body_short}\n"
        "Answer:"
    )


def query_llm(model: str, prompt: str, timeout: int) -> bool:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )
    text = result.stdout.decode("utf-8", errors="ignore").strip().lower()
    if "phish" in text:
        return True
    if "legit" in text or "not phishing" in text or "benign" in text:
        return False
    return "phish" in text


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def evaluate(df: pd.DataFrame, model: str, timeout: int) -> dict:
    tp = fp = tn = fn = 0
    pbar = tqdm(total=len(df), desc="Classifying emails", ncols=120)

    for _, row in df.iterrows():
        prompt = build_prompt(
            subject=str(row.get("Subject", "")),
            sender=str(row.get("X-From", "")),
            recipient=str(row.get("X-To", "")),
            body=str(row.get("Body", "")),
        )

        predicted_phish = query_llm(model, prompt, timeout)
        actual_phish = is_phishing_label(row.get("Label"))

        if predicted_phish and actual_phish:
            tp += 1
        elif predicted_phish and not actual_phish:
            fp += 1
        elif not predicted_phish and not actual_phish:
            tn += 1
        else:
            fn += 1

        metrics = compute_metrics(tp, fp, tn, fn)
        pbar.set_postfix(
            tp=metrics["tp"],
            fp=metrics["fp"],
            tn=metrics["tn"],
            fn=metrics["fn"],
            prec=f"{metrics['precision']:.3f}",
            rec=f"{metrics['recall']:.3f}",
            acc=f"{metrics['accuracy']:.3f}",
            f1=f"{metrics['f1']:.3f}",
        )
        pbar.update(1)

    pbar.close()
    return compute_metrics(tp, fp, tn, fn)


def save_results(metrics: dict, model: str, sample_size: int, phish_count: int, ham_count: int) -> None:
    row = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "sample_size": sample_size,
        "phishing_rows": phish_count,
        "normal_rows": ham_count,
        **metrics,
    }

    exists = RESULTS_PATH.exists()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(RESULTS_PATH, mode="a", header=not exists, index=False)
    print(f"Saved final metrics to {RESULTS_PATH}.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek via Ollama on phishing emails.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name (default: %(default)s)")
    parser.add_argument("--target-rows", type=int, default=TARGET_ROWS, help="Total rows to evaluate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--timeout", type=int, default=45, help="Per-request timeout in seconds")
    parser.add_argument("--skip-pull", action="store_true", help="Skip pulling the model if already cached")
    parser.add_argument("--keep-model", action="store_true", help="Do not delete the Ollama model after run")
    parser.add_argument(
        "--csv-path",
        type=Path,
        help="Optional local CSV path (e.g., experimentation/data/enron_data_fraud_labeled.csv).",
    )
    args = parser.parse_args()

    csv_path = download_dataset(args.csv_path)
    sample_df, phish_count, ham_count = load_sample(csv_path, args.target_rows, args.seed)

    if not args.skip_pull:
        pull_model(args.model)

    try:
        metrics = evaluate(sample_df, args.model, args.timeout)
        print(
            "Final confusion matrix and scores: "
            f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']} "
            f"Precision={metrics['precision']:.3f} Recall={metrics['recall']:.3f} "
            f"Accuracy={metrics['accuracy']:.3f} F1={metrics['f1']:.3f}"
        )
        save_results(metrics, args.model, len(sample_df), phish_count, ham_count)
    finally:
        if not args.keep_model:
            remove_model(args.model)


if __name__ == "__main__":
    main()
