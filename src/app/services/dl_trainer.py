
import json
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm


class EmailDataset(Dataset):
    """Simple text dataset that defers tokenization to the collate_fn."""

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class DLTrainer:
    def __init__(
        self,
        datamanager,
        model_checkpoint: str = "distilbert-base-uncased",
        save_dir: str | None = None
    ):
        self.dm = datamanager
        self.model_checkpoint = model_checkpoint
        self.save_dir = save_dir or os.path.join("models", "distilbert_phishing")
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Data preparation helpers
    # -----------------------------
    def _ensure_text_column(self):
        if self.dm.df is None:
            raise ValueError("Please load the dataset first.")

        df = self.dm.df
        if "text_combined" not in df.columns:
            subject_series = df["Subject"].fillna("") if "Subject" in df.columns else pd.Series([""] * len(df))
            body_series = df["Body"].fillna("") if "Body" in df.columns else pd.Series([""] * len(df))
            df["text_combined"] = (subject_series.astype(str) + " " + body_series.astype(str)).str.strip()
        return df

    def _encode_labels(self, labels):
        unique_labels = sorted(set([str(label) for label in labels]))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = [label2id[str(label)] for label in labels]
        id2label = {idx: label for label, idx in label2id.items()}
        return encoded, label2id, id2label

    def _build_dataloaders(
        self,
        texts,
        labels,
        tokenizer,
        batch_size: int,
        max_length: int,
        val_size: float = 0.2
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=val_size,
            random_state=42,
            stratify=labels if len(set(labels)) > 1 else None
        )

        train_dataset = EmailDataset(X_train, y_train)
        val_dataset = EmailDataset(X_val, y_val)

        class_counts = np.bincount(y_train)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        collate_fn = self._build_collate_fn(tokenizer, max_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        return train_loader, val_loader

    @staticmethod
    def _build_collate_fn(tokenizer, max_length: int) -> Callable:
        def collate(batch):
            texts, labels = zip(*batch)
            encodings = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            encodings["labels"] = torch.tensor(labels, dtype=torch.long)
            return encodings
        return collate

    # -----------------------------
    # Training and evaluation
    # -----------------------------
    def fine_tune_distilbert(
        self,
        epochs: int = 2,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        max_length: int = 256
    ):
        df = self._ensure_text_column()

        labels_raw = df["Label"]
        if labels_raw.isnull().any():
            df = df.dropna(subset=["Label"])
            labels_raw = df["Label"]

        texts = df["text_combined"].fillna("").astype(str).tolist()
        encoded_labels, label2id, id2label = self._encode_labels(labels_raw.tolist())

        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        config = AutoConfig.from_pretrained(
            self.model_checkpoint,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, config=config)
        model.to(self.device)

        train_loader, val_loader = self._build_dataloaders(
            texts, encoded_labels, tokenizer, batch_size, max_length
        )

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps
        )

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch in progress:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
            print(f"Epoch {epoch + 1} - average training loss: {epoch_loss / max(len(train_loader), 1):.4f}")

        metrics = self.evaluate(model, val_loader, id2label)
        self._print_metrics(metrics)
        self._save_model(model, tokenizer, label2id)
        return metrics

    def evaluate(self, model, dataloader, id2label):
        model.eval()
        all_labels, all_preds, prob_batches = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_labels.extend(batch["labels"].cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                prob_batches.append(probs.cpu().numpy())

        prob_matrix = np.concatenate(prob_batches, axis=0) if prob_batches else np.empty((0, len(id2label)))
        metrics = self._compute_metrics(all_labels, all_preds, prob_matrix, id2label)
        return metrics

    @staticmethod
    def _compute_metrics(labels, preds, prob_matrix, id2label):
        average = "binary" if len(id2label) == 2 else "weighted"
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=average, zero_division=0
        )

        roc_auc = None
        try:
            if len(id2label) == 2 and prob_matrix.shape[1] >= 2:
                roc_auc = roc_auc_score(labels, prob_matrix[:, 1])
            elif prob_matrix.size > 0:
                roc_auc = roc_auc_score(labels, prob_matrix, multi_class="ovr")
        except Exception:
            roc_auc = None

        target_names = [id2label[idx] for idx in sorted(id2label)]
        report_text = classification_report(
            labels,
            preds,
            target_names=target_names,
            zero_division=0
        )

        support_per_class = {
            id2label[idx]: int(sum(1 for label in labels if label == idx))
            for idx in sorted(id2label)
        }

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "support": support_per_class,
            "report": report_text
        }

    @staticmethod
    def _print_metrics(metrics):
        print("--- Evaluation Metrics (DL) ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        roc = metrics.get("roc_auc")
        if roc is not None:
            print(f"ROC-AUC:   {roc:.4f}")
        else:
            print("ROC-AUC:   not available for this run")
        print("Per-class support:")
        for label, count in metrics["support"].items():
            print(f"  {label}: {count}")
        print("Detailed classification report:")
        print(metrics["report"])

    # -----------------------------
    # Persistence
    # -----------------------------
    def _save_model(self, model, tokenizer, label2id):
        model.save_pretrained(self.save_dir)
        tokenizer.save_pretrained(self.save_dir)
        with open(os.path.join(self.save_dir, "label_mapping.json"), "w", encoding="utf-8") as fp:
            json.dump(label2id, fp, indent=2)
        print(f"Model and tokenizer saved to {self.save_dir}")

    def load_model(self):
        if not os.path.exists(self.save_dir):
            print(f"No saved DL model found in {self.save_dir}")
            return None, None, None

        try:
            with open(os.path.join(self.save_dir, "label_mapping.json"), "r", encoding="utf-8") as fp:
                label2id = json.load(fp)
        except FileNotFoundError:
            label2id = {}

        tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        config = AutoConfig.from_pretrained(self.save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.save_dir, config=config)
        model.to(self.device)
        id2label = {int(v): k for k, v in label2id.items()} if label2id else config.id2label
        print(f"Loaded DL model from {self.save_dir} (device: {self.device})")
        return model, tokenizer, id2label

    def evaluate_saved_model(self, batch_size: int = 8, max_length: int = 256):
        model, tokenizer, id2label = self.load_model()
        if model is None or tokenizer is None or id2label is None:
            return

        df = self._ensure_text_column()
        labels_raw = df["Label"]
        if labels_raw.isnull().any():
            df = df.dropna(subset=["Label"])
            labels_raw = df["Label"]

        texts = df["text_combined"].fillna("").astype(str).tolist()
        label2id = {label: idx for idx, label in id2label.items()} if isinstance(id2label, dict) else {}
        encoded_labels = [label2id.get(str(label), 0) for label in labels_raw.tolist()]

        _, val_loader = self._build_dataloaders(
            texts,
            encoded_labels,
            tokenizer,
            batch_size,
            max_length
        )

        metrics = self.evaluate(model, val_loader, id2label)
        self._print_metrics(metrics)
        return metrics
