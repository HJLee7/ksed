# from google.colab import drive
# drive.mount('/content/drive')

import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# ==============================================================================
# 1. Load data and build single-label targets (all essays, no downsampling)
# ==============================================================================
WORKING_DIR = Path("/content/drive/MyDrive/KSED")
essay = pd.read_csv(WORKING_DIR / "essay_truth.csv")

TEXT_COL = "og_text"
ID_COL = "id"
LABELS = ["shame", "guilt", "auth", "hubris"]
Y_COLS = ["truth_shame", "truth_guilt", "truth_auth", "truth_hubris"]  # same order as LABELS

Y = essay[Y_COLS].values.argmax(axis=1)   # one class id per essay (0..3)
essay["label"] = Y
SEED = 20260928

# ==============================================================================
# 2. Settings and folds (all essays, stratified)
# ==============================================================================
OUT = WORKING_DIR / "Validation 1 - ESSAY OOF"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "kcbert":    "beomi/kcbert-base",
    "klue-bert": "klue/bert-base",
}

MAX_LEN = 300   # kcbert-base cannot go above 300 tokens; the other two could use 512
N_FOLDS = 5
MAX_EPOCHS = 5
LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5]
WEIGHT_DECAY = 0.01

set_seed(SEED)

folds = list(
    StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    .split(essay, Y)
)


# ==============================================================================
# 3. Helpers
# ==============================================================================
class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.enc = tokenizer(df[TEXT_COL].tolist(), truncation=True, max_length=MAX_LEN)
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def macro_scores(y_true, probs):
    """Macro one-vs-rest AUC / PR-AUC, macro F1 and accuracy (argmax prediction)."""
    aucs, praucs = [], []
    for i in range(len(LABELS)):
        yi = (y_true == i).astype(int)
        if len(np.unique(yi)) < 2:
            continue  # AUC undefined when only one class is present
        aucs.append(roc_auc_score(yi, probs[:, i]))
        praucs.append(average_precision_score(yi, probs[:, i]))
    y_pred = probs.argmax(axis=1)
    return {
        "auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
        "prauc_macro": float(np.mean(praucs)) if praucs else float("nan"),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": float((y_true == y_pred).mean()),
    }


def make_args(name, lr, run_name):
    return TrainingArguments(
        output_dir=f"runs/{name}/{run_name}",
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        seed=SEED,
        report_to="none",
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
    )


def fresh_trainer(model_dir, tokenizer, args, train_ds, eval_ds, compute_metrics):
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(LABELS),
        problem_type="single_label_classification",
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def progress_dir(name, model_dir):
    """Checkpoint folder for one model. The name includes a hash of the settings,
    so changing any of them starts fresh instead of reusing stale checkpoints."""
    cfg = {"model_dir": model_dir, "seed": SEED, "n_folds": N_FOLDS,
           "max_epochs": MAX_EPOCHS, "max_len": MAX_LEN,
           "weight_decay": WEIGHT_DECAY, "n": len(essay),
           "text_col": TEXT_COL}
    sig = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    d = OUT / "progress" / f"{name}_{sig}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ==============================================================================
# 4. One model: out-of-fold probabilities for every (lr, epoch), then pick the best
# ==============================================================================
def run_model(name, model_dir):
    pred_path = OUT / f"bert_{name}_oof_predictions.csv"
    if pred_path.exists():
        print(f"{name}: already finished, skipping")
        return

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prog = progress_dir(name, model_dir)

    # oof[(lr, epoch)][i] = probabilities for essay i from the model that did NOT train on it
    oof = {(lr, ep): np.full((len(essay), len(LABELS)), np.nan)
           for lr in LEARNING_RATES for ep in range(1, MAX_EPOCHS + 1)}

    for lr in LEARNING_RATES:
        for k, (tr_idx, va_idx) in enumerate(folds):
            ck = prog / f"lr{lr}_f{k}.npy"

            # ---- pick up where we left off: reuse a finished run ----
            if ck.exists():
                saved = np.load(ck)              # shape (MAX_EPOCHS, len(va_idx), n_labels)
                for ep in range(1, MAX_EPOCHS + 1):
                    oof[(lr, ep)][va_idx] = saved[ep - 1]
                print(f"{name}  lr={lr}  fold {k + 1}/{N_FOLDS}: loaded from checkpoint")
                continue

            print(f"\n### {name}  lr={lr}  fold {k + 1}/{N_FOLDS}")
            per_epoch = []   # filled by compute_metrics: one probability array per epoch

            def compute_metrics(eval_pred, store=per_epoch):
                logits, _ = eval_pred
                store.append(softmax(np.asarray(logits)))
                return {"n_eval": float(len(logits))}

            trainer = fresh_trainer(
                model_dir, tokenizer, make_args(name, lr, f"lr{lr}_f{k}"),
                EssayDataset(essay.iloc[tr_idx], tokenizer),
                EssayDataset(essay.iloc[va_idx], tokenizer),
                compute_metrics,
            )
            trainer.train()
            for ep, p in enumerate(per_epoch, start=1):
                oof[(lr, ep)][va_idx] = p

            # ---- checkpoint this run (write to a temp file, then rename) ----
            tmp = prog / f"lr{lr}_f{k}.tmp.npy"
            np.save(tmp, np.stack(per_epoch))
            os.replace(tmp, ck)

            del trainer
            free_memory()

    # Score every (lr, epoch) on all out-of-fold probabilities
    rows = []
    for (lr, ep), p in oof.items():
        fold_auc = [macro_scores(Y[va], p[va])["auc_macro"] for _, va in folds]
        rows.append({"lr": lr, "epoch": ep, **macro_scores(Y, p),
                     "fold_auc_sd": float(np.nanstd(fold_auc, ddof=1))})
    grid = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    grid_out = grid.copy()
    grid_out.insert(0, "model", name)
    grid_out.to_csv(OUT / f"bert_{name}_grid.csv", index=False)

    best = grid.iloc[0]
    best_lr, best_ep = float(best["lr"]), int(best["epoch"])
    print(f"\n=== {name}: all settings (pooled OOF over {len(essay)} essays) ===")
    print(grid.round(3).to_string(index=False))
    print(f"Best (by macro F1): lr={best_lr}, epoch={best_ep} | "
          f"AUC={best['auc_macro']:.3f}, PR-AUC={best['prauc_macro']:.3f}, "
          f"F1={best['f1_macro']:.3f}, fold AUC SD={best['fold_auc_sd']:.3f}")
    print(f"Grid spread: median F1 = {grid['f1_macro'].median():.3f}, "
          f"best - median = {best['f1_macro'] - grid['f1_macro'].median():.3f}")

    best_probs = oof[(best_lr, best_ep)]
    y_pred = best_probs.argmax(axis=1)
    per_label = {}
    for i, l in enumerate(LABELS):
        yi = (Y == i).astype(int)
        per_label[f"auc_{l}"] = roc_auc_score(yi, best_probs[:, i])
        per_label[f"prauc_{l}"] = average_precision_score(yi, best_probs[:, i])
        per_label[f"f1_{l}"] = f1_score(yi, (y_pred == i).astype(int), zero_division=0)
    print(f"\n=== {name}: per-label results at the best setting ===")
    print(pd.Series(per_label).round(3))

    # Save predictions LAST, so the file's existence means the model is finished
    out_df = essay[[ID_COL, TEXT_COL]].copy()   # id + text, then the prob_ and pred_ columns
    for i, l in enumerate(LABELS):
        out_df[f"prob_{l}"] = best_probs[:, i]
        out_df[f"pred_{l}"] = (y_pred == i).astype(int)   # 1 for the predicted class, else 0
    out_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    shutil.rmtree(prog, ignore_errors=True)   # checkpoints no longer needed


# ==============================================================================
# 5. Run all models
# ==============================================================================
for name, path in MODELS.items():
    run_model(name, path)

print(f"\nPredictions and grids saved in: {OUT}")