
# Deep Learning Classifier

This project now includes a lightweight transformer fine-tuned to detect phishing emails. The model uses DistilBERT on the combined `text_combined` field (Subject + Body) to complement the classical ML baselines.

## Implementation overview
- Dataset: uses `datamanager.df`; if `text_combined` is missing it is built from Subject and Body.
- Split: train/validation split (80/20, stratified when possible).
- Tokenization: DistilBERT tokenizer with padding/truncation to a configurable max length (default 256).
- Imbalance handling: sample weights via `WeightedRandomSampler` on the training split.
- Training: AdamW + linear warmup/decay; defaults lr=5e-5, batch_size=8, epochs=2.
- Metrics: accuracy, precision, recall, F1, ROC-AUC (when available), per-class support + a full classification report.
- Device: automatically uses CUDA when available, otherwise CPU.
- Persistence: saves model, tokenizer, and `label_mapping.json` to `models/distilbert_phishing/`.

## How to train
1. Install dependencies: `pip install -r ./src/requirements.txt`.
2. Run the CLI: `python ./src`.
3. Load the dataset, then choose `Train a DL model` > `Fine-tune DistilBERT`.
4. Accept defaults or enter custom epochs/batch size/lr/max length. Training prints epoch losses and validation metrics.

## How to evaluate
- From the CLI, choose `Train a DL model` > `Evaluate saved DistilBERT model`. This reloads the artifacts from `models/distilbert_phishing/` and reports the metric set above on the current validation split.

## Repro tips
- Ensure `Label` is present and non-null before training; rows with missing labels are dropped.
- Larger max lengths capture more context but consume more memory; consider lowering to 128 on CPU-only setups.
- A GPU accelerates fine-tuning but is not required; expect slower epochs on CPU.
