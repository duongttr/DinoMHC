import argparse
import json, yaml
import tempfile
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from safetensors.torch import load_file
from pathlib import Path
from src.model import DinoMHC, DinoMHC_ProtTrans
import torch
from src.data_module import MHCPeptideDataModule
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

torch._dynamo.config.capture_scalar_outputs = True


def centroid_trim(peptide: str, target_length: int = 9, n_anchors: int = 4) -> str:
    """
    Trims a peptide to a target length by removing residues from the middle.

    Keeps the first `n_anchors` residues from the N-terminus and the remaining
    `target_length - n_anchors` residues from the C-terminus, deleting everything
    in between.
    """
    current_length = len(peptide)

    # If it's already the right length (or shorter), return as-is
    if current_length <= target_length:
        return peptide

    c_length = target_length - n_anchors

    if c_length <= 0:
        return peptide[:target_length]

    # Keep n_anchors from N-term and c_length from C-term
    trimmed_peptide = peptide[:n_anchors] + peptide[-c_length:]
    
    return trimmed_peptide


def build_inference_df(original_df, max_supported_length, centroid_target_length=9, n_anchors=4):
    if "peptide" not in original_df.columns:
        raise ValueError("Input dataset must contain column 'peptide'")

    base_df = original_df.reset_index(drop=True).copy()
    base_df["__orig_index"] = np.arange(len(base_df))

    expanded_records = []
    for _, row in base_df.iterrows():
        peptide = str(row["peptide"]).strip().upper()

        if len(peptide) <= max_supported_length:
            rec = row.to_dict()
            rec["peptide"] = peptide
            expanded_records.append(rec)
        else:
            # Peptide is longer than supported MAX — trim it
            trimmed = centroid_trim(peptide, target_length=centroid_target_length, n_anchors=n_anchors)
            rec = row.to_dict()
            rec["peptide"] = trimmed
            expanded_records.append(rec)

    expanded_df = pd.DataFrame.from_records(expanded_records)
    if expanded_df.empty:
        raise ValueError("Expanded inference dataset is empty")

    return base_df, expanded_df


def aggregate_window_predictions(expanded_combined_df, base_df, prediction_columns):
    # Since we use centroid_trim, it's a 1-to-1 mapping. No aggregation needed, just merge predictions.
    final_df = base_df.copy()
    
    final_df["__n_windows"] = 1

    for col in prediction_columns + ["prediction"]:
        final_df[col] = expanded_combined_df[col].values

    return final_df

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_checkpoint(model, checkpoint_path, device="cuda"):
    model = model.to(device=device)
    
    if str(checkpoint_path).endswith(".safetensors"):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
    
    if all(key.startswith("model.") for key in state_dict.keys()):
        state_dict = {key[len("model."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.to(dtype=torch.bfloat16)
    model.eval()

def load_data_module(benchmark_dataset_path, 
                     batch_size, 
                     num_workers, 
                     model_config,
                     device):
    
    # Set tokenizer type
    if model_config.get("encoder_type", "esm2").startswith("esm"):
        tokenizer_type = "esm2"
    elif model_config.get("encoder_type", "esm2").startswith("prottrans"):
        tokenizer_type = "prottrans"
    else:
        tokenizer_type = "embedding"
    
    data_module = MHCPeptideDataModule(
        data_dir=".", # Not used
        fold=0, # Not used
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_type=tokenizer_type,
        esm_model_name=model_config.get("esm_model_name", None),
        prottrans_model_name=model_config.get("prottrans_model_name", None),
        max_peptide_length=model_config["max_peptide_length"],
        max_mhc_length=model_config["max_mhc_length"],
        use_flanks=model_config["use_flanks"],
        flank_length=model_config["flank_length"],
        flank_mask_prob=0,
        binarize_labels=True,
        label_threshold=0.5,
        test_files=[benchmark_dataset_path],
        pin_memory=device.type == "cuda"
    )
    data_module.setup(stage="test")
    
    return data_module

def evaluate_predictions(combined_df, threshold=0.5):
    y_true = combined_df["label"].values
    y_scores = combined_df["prediction"].values
    y_pred = (y_scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    results = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "auroc": float(auroc),
        "auprc": float(auprc)
    }
    
    return results

def evaluate_predictions_grouped_by_allele(combined_df, threshold=0.5):
    # Grouped by mhc allele
    combined_df["predicted_label"] = (combined_df["prediction"] >= threshold).astype(int)
    results = {}
    
    for allele, group in combined_df.groupby("mhc"):
        y_true = group["label"].values
        y_pred = group["predicted_label"].values
        y_scores = group["prediction"].values
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_scores, )
        auprc = average_precision_score(y_true, y_scores)
        
        # Special metric: PPV@k
        def ppv_at_k(y_true, y_scores, k):
            sorted_indices = np.argsort(y_scores)[::-1]
            top_k_indices = sorted_indices[:k]
            return np.sum(y_true[top_k_indices]) / k

        ppv_pos = ppv_at_k(y_true, y_scores, k=np.sum(y_true))
        
        results[str(allele)] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_acc),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "mcc": float(mcc),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "ppv_pos": float(ppv_pos)
        }
        
    # Calculate average metrics across all alleles
    results["average"] = {
        "tn": sum(r["tn"] for r in results.values()) / len(results),
        "fp": sum(r["fp"] for r in results.values()) / len(results),
        "fn": sum(r["fn"] for r in results.values()) / len(results),
        "tp": sum(r["tp"] for r in results.values()) / len(results),
        "accuracy": sum(r["accuracy"] for r in results.values()) / len(results),
        "balanced_accuracy": sum(r["balanced_accuracy"] for r in results.values()) / len(results),
        "precision": sum(r["precision"] for r in results.values()) / len(results),
        "recall": sum(r["recall"] for r in results.values()) / len(results),
        "specificity": sum(r["specificity"] for r in results.values()) / len(results),
        "f1_score": sum(r["f1_score"] for r in results.values()) / len(results),
        "mcc": sum(r["mcc"] for r in results.values()) / len(results),
        "auroc": sum(r["auroc"] for r in results.values()) / len(results),
        "auprc": sum(r["auprc"] for r in results.values()) / len(results),
        "ppv_pos": sum(r["ppv_pos"] for r in results.values()) / len(results)
    }
    
    return results

def main(args):
    # Verify existence of paths
    assert len(args.checkpoints) > 0, "At least one model checkpoint must be provided"
    assert Path(args.benchmark_dataset).exists(), "Benchmark dataset path does not exist"
    assert Path(args.model_config).exists(), "Model configuration file does not exist"
    
    # Init device
    device = torch.device(f"{args.device}:{args.device_id}" if args.device == "cuda" else args.device)
    
    # Load Model configuration
    model_config = load_config(args.model_config)
    print(json.dumps(model_config, indent=4))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(args.benchmark_dataset)
    max_supported_length = int(model_config["max_peptide_length"])

    base_df, expanded_df = build_inference_df(
        original_df=original_df,
        max_supported_length=max_supported_length,
        centroid_target_length=args.centroid_target_length,
        n_anchors=args.n_anchors
    )

    print(f"Original rows: {len(base_df)} | Inference rows: {len(expanded_df)}")
    
    if model_config.get("encoder_type", "esm2").startswith("esm"):
        model = DinoMHC(model_config)
    else:
        model = DinoMHC_ProtTrans(model_config)
    
    probs_csv_files = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=True) as tmp_file:
        expanded_df.to_csv(tmp_file.name, index=False)

        for idx, checkpoint_path in enumerate(args.checkpoints):
            print(f"🔄 Evaluating checkpoint {checkpoint_path} ({idx+1}/{len(args.checkpoints)})")

            load_model_checkpoint(model, checkpoint_path, device)

            data_module = load_data_module(tmp_file.name, args.batch_size, args.num_workers, model_config, device)

            probs_csv_file = output_dir / f"probs_{idx}.csv"
            with open(probs_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"prediction_{checkpoint_path}"])

                total_batches = len(data_module.test_dataloader())

                for batch in tqdm(data_module.test_dataloader(), desc="Evaluating", total=total_batches, unit="batch"):
                    with torch.no_grad():
                        outputs = model(
                            peptide_tokens=batch["peptide_tokens"].to(device),
                            mhc_tokens=batch["mhc_tokens"].to(device),
                            peptide_mask=batch["peptide_mask"].to(device),
                            mhc_mask=batch["mhc_mask"].to(device),
                            nflank_len=batch["nflank_len"].to(device),
                            cflank_len=batch["cflank_len"].to(device),
                            original_peptide_len=batch["original_peptide_len"].to(device),
                            return_attention=False
                        )

                    predictions = outputs[f"prediction"].flatten().cpu().float().numpy()
                    for pred in predictions:
                        writer.writerow([pred])

            probs_csv_files.append(probs_csv_file)
        
    # Combine predictions with original dataset for easier evaluation
    
    predictions_dfs = [pd.read_csv(probs_file) for probs_file in probs_csv_files]
    expanded_combined_df = pd.concat([expanded_df.reset_index(drop=True)] + predictions_dfs, axis=1)
    prediction_columns = [f"prediction_{cp}" for cp in args.checkpoints]
    expanded_combined_df["prediction"] = expanded_combined_df[prediction_columns].mean(axis=1)

    combined_df = aggregate_window_predictions(
        expanded_combined_df=expanded_combined_df,
        base_df=base_df,
        prediction_columns=prediction_columns
    )

    combined_csv_file = output_dir / "full_predictions.csv"
    combined_df.to_csv(combined_csv_file, index=False)
    print(f"Saved aggregated predictions to {combined_csv_file}")
    
    # Evaluate predictions
    if args.calculate_by_allele:
        evaluation_results = evaluate_predictions_grouped_by_allele(combined_df, args.threshold)
    else:
        evaluation_results = evaluate_predictions(combined_df, args.threshold)
        
    evaluation_results_xlsx_file = output_dir / "evaluation_results.xlsx"

    evaluation_results_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
    evaluation_results_df.index.name = "mhc"
    evaluation_results_df = evaluation_results_df.reset_index()
    evaluation_results_df.to_excel(evaluation_results_xlsx_file, index=False)

    print(f"Saved evaluation results to {evaluation_results_xlsx_file}")
    
    # Done testing
    print("✅ Testing completed successfully.")
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run testing on benchmark datasets the DinoMHC model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Main arguments
    parser.add_argument(
        "--checkpoints",
        required=True,
        nargs="+",
        help="Paths to the model checkpoint files for each fold",
        type=str
    )
    
    parser.add_argument(
        "--benchmark_dataset",
        type=str,
        required=True,
        help="Path to the benchmark dataset file (CSV format)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Directory to save the evaluation results"
    )
    
    # Model configuration arguments
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model configuration file (YAML format)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification when evaluating predictions"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the evaluation on (e.g., 'cuda' or 'cpu')"
    )
    
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU device ID to use if running on CUDA"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--calculate_by_allele",
        action="store_true",
        help="Whether to calculate evaluation metrics separately for each MHC allele"
    )
    
    # Handling long sequences exceeding the model's maximum input length
    parser.add_argument(
        "--centroid_target_length",
        type=int,
        default=9,
        help="Length to trim peptides to if they exceed the max supported length."
    )

    parser.add_argument(
        "--n_anchors",
        type=int,
        default=4,
        help="Number of N-terminal anchors to preserve during centroid trimming."
    )
    
    args = parser.parse_args()
    
    # For debugging purposes, print the parsed arguments
    print(json.dumps(vars(args), indent=4))
    
    main(args)
    