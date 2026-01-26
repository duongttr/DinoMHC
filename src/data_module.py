"""
Data Module for MHC-Peptide Binding Prediction.

This module provides:
1. MHCPeptideDataset: PyTorch Dataset for loading and tokenizing peptide-MHC pairs
2. MHCPeptideDataModule: PyTorch Lightning DataModule for handling train/val/test splits

Supports both simple embedding tokenization and ESM-2 tokenization.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Callable

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

# Import MHC sequences from constants
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import MHC_PROT_SEQ


# Amino acid vocabulary for simple embedding encoder
AA_VOCAB = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,  # Unknown amino acid
    '<PAD>': 21  # Padding token
}


def simple_tokenize(
    sequence: str, 
    max_length: int,
    pad_token: int = 21
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple tokenization for embedding encoder.
    
    Args:
        sequence: Amino acid sequence string
        max_length: Maximum sequence length (will pad/truncate)
        pad_token: Padding token ID
    
    Returns:
        tokens: [max_length] token indices
        mask: [max_length] boolean mask (True = valid, False = padding)
    """
    # Convert sequence to token indices
    tokens = [AA_VOCAB.get(aa.upper(), AA_VOCAB['X']) for aa in sequence]
    
    # Truncate if necessary
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # Create mask - 'X' (unknown) tokens are masked as False (not trained)
    seq_len = len(tokens)
    mask = [(t != AA_VOCAB['X']) for t in tokens] + [False] * (max_length - seq_len)
    
    # Pad tokens
    tokens = tokens + [pad_token] * (max_length - seq_len)
    
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class ESMTokenizer:
    """
    Wrapper for ESM-2 tokenization using Hugging Face Transformers.
    Handles batch tokenization with proper padding.
    """
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D"):
        try:
            from transformers import EsmTokenizer
            self.tokenizer = EsmTokenizer.from_pretrained(model_name)
            self.padding_idx = self.tokenizer.pad_token_id
            self.cls_idx = self.tokenizer.cls_token_id  # BOS token
            self.eos_idx = self.tokenizer.eos_token_id
        except ImportError:
            raise ImportError("Transformers package not found. Install with: pip install transformers")
    
    def tokenize(
        self, 
        sequence: str, 
        max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a single sequence.
        
        ESM adds BOS (CLS) and EOS tokens:
        [CLS, seq..., EOS, PAD...]
        
        Args:
            sequence: Amino acid sequence string
            max_length: Maximum length AFTER adding BOS/EOS (so actual seq length = max_length - 2)
        
        Returns:
            tokens: [max_length] token indices including BOS/EOS
            mask: [max_length] boolean mask (True = non-padding)
        """
        # Tokenize using transformers
        encoded = self.tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        tokens = encoded['input_ids'].squeeze(0)  # [max_length]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [max_length]
        
        # Convert attention mask to boolean (True = non-padding)
        mask = attention_mask.bool()
        
        # Mask 'X' (unknown) tokens as False (not trained)
        # ESM tokenizer maps 'X' to token id 24
        x_token_id = self.tokenizer.convert_tokens_to_ids('X')
        mask = mask & (tokens != x_token_id)
        
        return tokens, mask
    
    def tokenize_batch(
        self, 
        sequences: List[str], 
        max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of sequences.
        
        Args:
            sequences: List of amino acid sequences
            max_length: Maximum length including BOS/EOS
        
        Returns:
            tokens: [batch, max_length] token indices
            mask: [batch, max_length] boolean mask
        """
        # Batch tokenization using transformers
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        tokens = encoded['input_ids']  # [batch, max_length]
        mask = encoded['attention_mask'].bool()  # [batch, max_length]
        
        # Mask 'X' (unknown) tokens as False (not trained)
        x_token_id = self.tokenizer.convert_tokens_to_ids('X')
        mask = mask & (tokens != x_token_id)
        
        return tokens, mask


class MHCPeptideDataset(Dataset):
    """
    Dataset for MHC-Peptide binding prediction.
    
    Supports:
    - CSV files with columns: peptide, mhc, label (and optionally nflank, cflank, prot)
    - Simple embedding tokenization or ESM-2 tokenization
    - Optional flanking sequences
    - Label binarization or continuous labels
    - Random flank masking for robustness (flank_mask_prob)
    """
    
    def __init__(
        self,
        data_path: str,
        mhc_seq_path: Optional[str] = None,
        tokenizer_type: str = 'embedding',  # 'embedding' or 'esm2'
        esm_model_name: str = 'facebook/esm2_t6_8M_UR50D',
        max_peptide_length: int = 15,
        max_mhc_length: int = 385,  # MHC-I pseudo sequence length + BOS/EOS for ESM
        use_flanks: bool = False,
        flank_length: int = 5,
        flank_mask_prob: float = 0.0,  # Probability to randomly mask flanks (0.0-1.0)
        binarize_labels: bool = False,
        label_threshold: float = 0.5,
        mhc_column: str = 'mhc',
        peptide_column: str = 'peptide',
        label_column: str = 'label',
    ):
        """
        Args:
            data_path: Path to CSV file with peptide-MHC-label data
            mhc_seq_path: Path to CSV with MHC allele sequences (columns: allele, sequence)
            tokenizer_type: 'embedding' for simple vocab, 'esm2' for ESM tokenization
            esm_model_name: ESM model name if using ESM tokenization
            max_peptide_length: Maximum peptide length (including BOS/EOS for ESM)
            max_mhc_length: Maximum MHC sequence length
            use_flanks: Whether to include flanking sequences
            flank_length: Length of N-terminal and C-terminal flanks
            flank_mask_prob: Probability to randomly mask flanks with 'X' (0.0-1.0).
                             When triggered, nflank is masked from left (e.g., XXXAA)
                             and cflank is masked from right (e.g., AAAXX).
                             This helps prevent model from biasing towards flank presence.
            binarize_labels: Whether to binarize labels
            label_threshold: Threshold for binarization
            mhc_column: Column name for MHC allele
            peptide_column: Column name for peptide sequence
            label_column: Column name for label
        """
        self.data_path = data_path
        self.tokenizer_type = tokenizer_type
        self.max_peptide_length = max_peptide_length
        self.max_mhc_length = max_mhc_length
        self.use_flanks = use_flanks
        self.flank_length = flank_length
        self.flank_mask_prob = flank_mask_prob
        self.binarize_labels = binarize_labels
        self.label_threshold = label_threshold
        
        # Load data
        self.df = pd.read_csv(data_path)
        self.mhc_column = mhc_column
        self.peptide_column = peptide_column
        self.label_column = label_column
        
        # Load MHC sequences - prioritize constants, then external file
        self.mhc_sequences = MHC_PROT_SEQ.copy()  # Use built-in sequences
        if mhc_seq_path is not None and os.path.exists(mhc_seq_path):
            # Override/extend with external file if provided
            mhc_df = pd.read_csv(mhc_seq_path)
            external_seqs = dict(zip(mhc_df['allele'], mhc_df['sequence']))
            self.mhc_sequences.update(external_seqs)
        
        # Initialize tokenizer
        if tokenizer_type == 'esm2':
            self.tokenizer = ESMTokenizer(esm_model_name)
        else:
            self.tokenizer = None  # Use simple_tokenize
        
        # Adjust max lengths for ESM (BOS + seq + EOS) and flanks
        # When use_flanks=True, peptide length = nflank + core_peptide + cflank
        # Max possible extension = 2 * flank_length
        if use_flanks:
            max_peptide_with_flanks = max_peptide_length + 2 * flank_length
        else:
            max_peptide_with_flanks = max_peptide_length

        if tokenizer_type == 'esm2':
            self.effective_peptide_length = max_peptide_with_flanks + 2  # +2 for BOS/EOS
            self.effective_mhc_length = max_mhc_length + 2
        else:
            self.effective_peptide_length = max_peptide_with_flanks
            self.effective_mhc_length = max_mhc_length
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _get_mhc_sequence(self, allele: str) -> str:
        """Get MHC sequence for an allele."""
        if allele in self.mhc_sequences:
            return self.mhc_sequences[allele]
        else:
            # Log warning for missing allele (only once per allele)
            if not hasattr(self, '_warned_alleles'):
                self._warned_alleles = set()
            if allele not in self._warned_alleles:
                print(f"Warning: MHC allele '{allele}' not found in sequence database, using placeholder")
                self._warned_alleles.add(allele)
            return "X" * 365  # Placeholder - average MHC length
    
    def _prepare_peptide(self, row: pd.Series) -> tuple:
        """Prepare peptide sequence, optionally with flanks.

        Returns:
            tuple: (full_peptide, nflank_len, cflank_len)

        Note:
            - Ensures extended peptide doesn't exceed max_peptide_length + 2*flank_length
            - If the core peptide is too long, flanks may be truncated or removed
            - Priority: core peptide > flanks (since core is essential for binding)
            - When flank_mask_prob > 0, randomly masks flanks with 'X' characters
              to prevent model from biasing towards flank presence
        """
        peptide = row[self.peptide_column]
        nflank_len = 0
        cflank_len = 0
        original_peptide_len = len(peptide)

        if self.use_flanks:
            nflank = row.get('nflank', '')
            cflank = row.get('cflank', '')

            # Take last flank_length of N-flank and first flank_length of C-flank
            nflank = str(nflank)[-self.flank_length:] if pd.notna(nflank) else ''
            cflank = str(cflank)[:self.flank_length] if pd.notna(cflank) else ''

            # Random flank masking for robustness training
            # This helps the model not bias towards flank presence
            if self.flank_mask_prob > 0 and np.random.rand() < self.flank_mask_prob:
                # Randomly choose how many positions to mask (0 to full length)
                if len(nflank) > 0:
                    # Mask nflank from left: XXXAA, XXAAA, etc.
                    n_mask_positions = np.random.randint(0, len(nflank) + 1)
                    nflank = 'X' * n_mask_positions + nflank[n_mask_positions:]
                
                if len(cflank) > 0:
                    # Mask cflank from right: AAAXX, AAXXX, etc.
                    c_mask_positions = np.random.randint(0, len(cflank) + 1)
                    cflank = cflank[:len(cflank) - c_mask_positions] + 'X' * c_mask_positions

            # Check if extended sequence would exceed max length
            # max_peptide_length is for core peptide only
            total_len = len(nflank) + len(peptide) + len(cflank)
            max_total = self.max_peptide_length + 2 * self.flank_length

            if total_len > max_total:
                # Core peptide is too long, need to truncate flanks
                # This is rare but possible with very long peptides
                available_for_flanks = max_total - len(peptide)
                if available_for_flanks <= 0:
                    # Core peptide itself exceeds max, remove flanks entirely
                    nflank = ''
                    cflank = ''
                else:
                    # Distribute available space between N-flank and C-flank
                    # Prefer to keep both flanks partially if possible
                    max_flank = available_for_flanks // 2
                    nflank = nflank[-max_flank:] if len(nflank) > max_flank else nflank
                    remaining = available_for_flanks - len(nflank)
                    cflank = cflank[:remaining] if len(cflank) > remaining else cflank

            nflank_len = len(nflank)
            cflank_len = len(cflank)

            peptide = nflank + peptide + cflank

        return peptide, nflank_len, cflank_len, original_peptide_len
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Get sequences
        peptide, nflank_len, cflank_len, original_peptide_len = self._prepare_peptide(row)
        mhc_allele = row[self.mhc_column]
        mhc_seq = self._get_mhc_sequence(mhc_allele)

        # Get label
        label = row[self.label_column]
        if self.binarize_labels:
            label = 1.0 if label >= self.label_threshold else 0.0

        # Tokenize
        if self.tokenizer_type == 'esm2':
            peptide_tokens, peptide_mask = self.tokenizer.tokenize(
                peptide, self.effective_peptide_length
            )
            mhc_tokens, mhc_mask = self.tokenizer.tokenize(
                mhc_seq, self.effective_mhc_length
            )
        else:
            peptide_tokens, peptide_mask = simple_tokenize(
                peptide, self.effective_peptide_length
            )
            mhc_tokens, mhc_mask = simple_tokenize(
                mhc_seq, self.effective_mhc_length
            )

        return {
            'peptide_tokens': peptide_tokens,
            'peptide_mask': peptide_mask,
            'mhc_tokens': mhc_tokens,
            'mhc_mask': mhc_mask,
            'mhc_name': mhc_allele,  # Include MHC allele name for per-MHC metrics
            'label': torch.tensor(label, dtype=torch.float32),
            'idx': torch.tensor(idx, dtype=torch.long),
            'nflank_len': torch.tensor(nflank_len, dtype=torch.long),
            'cflank_len': torch.tensor(cflank_len, dtype=torch.long),
            'original_peptide_len': torch.tensor(original_peptide_len, dtype=torch.long)
        }


def mhc_peptide_collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function to handle mixed tensor and string data.

    Args:
        batch: List of sample dictionaries from dataset

    Returns:
        Collated batch with tensors stacked and strings as lists
    """
    result = {}

    for key in batch[0].keys():
        if key in ['mhc_name', 'peptide_seq']:
            # Keep as list of strings
            result[key] = [sample[key] for sample in batch]
        else:
            # Stack tensors
            result[key] = torch.stack([sample[key] for sample in batch])

    return result


class MHCPeptideDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MHC-Peptide binding prediction.
    
    Handles:
    - K-fold cross validation setup
    - Train/Val/Test data loading
    - Batch collation
    - Distributed training compatibility
    """
    
    def __init__(
        self,
        data_dir: str,
        mhc_seq_path: Optional[str] = None,
        fold: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
        tokenizer_type: str = 'embedding',
        esm_model_name: str = 'facebook/esm2_t6_8M_UR50D',
        max_peptide_length: int = 15,
        max_mhc_length: int = 385,
        use_flanks: bool = False,
        flank_length: int = 5,
        flank_mask_prob: float = 0.0,  # Probability to randomly mask flanks during training
        binarize_labels: bool = False,
        label_threshold: float = 0.5,
        pin_memory: bool = True,
        test_files: Optional[List[str]] = None
    ):
        """
        Args:
            data_dir: Directory containing fold files (fold_X_train.csv, fold_X_val.csv)
            mhc_seq_path: Path to CSV with MHC allele sequences
            fold: Which fold to use (0-4 for 5-fold CV)
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            tokenizer_type: 'embedding' or 'esm2'
            esm_model_name: ESM model name if using ESM
            max_peptide_length: Max peptide length
            max_mhc_length: Max MHC sequence length
            use_flanks: Whether to use flanking sequences
            flank_length: Flank length
            flank_mask_prob: Probability to randomly mask flanks during training (0.0-1.0).
                             Only applied to training data, not validation/test.
                             Helps prevent model from biasing towards flank presence.
            binarize_labels: Whether to binarize labels
            label_threshold: Threshold for binarization
            pin_memory: Whether to pin memory for faster GPU transfer
            test_files: List of test file paths (optional)
            train_subset_fraction: Fraction of training data to use (0.0-1.0).
                                   Use 0.1 for 10%, 0.5 for 50%, etc. Default 1.0 (full data)
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.mhc_seq_path = mhc_seq_path
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_type = tokenizer_type
        self.esm_model_name = esm_model_name
        self.max_peptide_length = max_peptide_length
        self.max_mhc_length = max_mhc_length
        self.use_flanks = use_flanks
        self.flank_length = flank_length
        self.flank_mask_prob = flank_mask_prob
        self.binarize_labels = binarize_labels
        self.label_threshold = label_threshold
        self.pin_memory = pin_memory
        self.test_files = test_files or []
        
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = {}
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage (fit, validate, test, predict)."""
        
        common_kwargs = {
            'mhc_seq_path': self.mhc_seq_path,
            'tokenizer_type': self.tokenizer_type,
            'esm_model_name': self.esm_model_name,
            'max_peptide_length': self.max_peptide_length,
            'max_mhc_length': self.max_mhc_length,
            'use_flanks': self.use_flanks,
            'flank_length': self.flank_length,
            'binarize_labels': self.binarize_labels,
            'label_threshold': self.label_threshold,
        }
        
        if stage == 'fit' or stage is None:
            # Training and validation datasets
            train_path = self.data_dir / f"fold_{self.fold}_train.csv"
            val_path = self.data_dir / f"fold_{self.fold}_val.csv"
            
            if train_path.exists():
                self.train_dataset = MHCPeptideDataset(
                    data_path=str(train_path),
                    flank_mask_prob=self.flank_mask_prob,  # Only apply to training
                    **common_kwargs
                )
            
            if val_path.exists():
                self.val_dataset = MHCPeptideDataset(
                    data_path=str(val_path),
                    flank_mask_prob=0.0,  # No masking for validation
                    **common_kwargs
                )
        
        if stage == 'test' or stage is None:
            # Test datasets
            for test_file in self.test_files:
                test_path = Path(test_file)
                if test_path.exists():
                    self.test_datasets[test_path.stem] = MHCPeptideDataset(
                        data_path=str(test_path),
                        flank_mask_prob=0.0,  # No masking for test
                        **common_kwargs
                    )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # For consistent batch sizes
            persistent_workers=self.num_workers > 0,
            collate_fn=mhc_peptide_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=mhc_peptide_collate_fn
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.test_datasets) == 0:
            return None
        
        if len(self.test_datasets) == 1:
            dataset = list(self.test_datasets.values())[0]
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=mhc_peptide_collate_fn
            )
        
        # Multiple test datasets
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=mhc_peptide_collate_fn
            )
            for dataset in self.test_datasets.values()
        ]
    
    def predict_dataloader(self) -> DataLoader:
        """Use validation data for prediction by default."""
        return self.val_dataloader()
    
    @property
    def train_dataset_size(self) -> int:
        return len(self.train_dataset) if self.train_dataset else 0
    
    @property
    def val_dataset_size(self) -> int:
        return len(self.val_dataset) if self.val_dataset else 0


# =============================================================================
# Utility functions for data inspection
# =============================================================================

def inspect_dataset(dataset: MHCPeptideDataset, num_samples: int = 1_000_000):
    """Print sample data from dataset for inspection."""
    print(f"Dataset size: {len(dataset)}")
    print(f"Tokenizer type: {dataset.tokenizer_type}")
    print(f"Max peptide length: {dataset.effective_peptide_length}")
    print(f"Max MHC length: {dataset.effective_mhc_length}")
    print()
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if sample['peptide_mask'].sum().item() > 14:
            print(f"Sample {i}:")
            print(sample['peptide_mask'])
            print(sample['peptide_tokens'])
            print(f"  Peptide tokens shape: {sample['peptide_tokens'].shape}")
            print(f"  Peptide mask sum (actual length): {sample['peptide_mask'].sum().item()}")
            print(f"  MHC tokens shape: {sample['mhc_tokens'].shape}")
            print(f"  MHC mask sum (actual length): {sample['mhc_mask'].sum().item()}")
            print(f"  Label: {sample['label'].item():.4f}")
            print()


if __name__ == "__main__":
    # Test ESMTokenizer
    esm_tokenizer = ESMTokenizer()
    tokens, mask = esm_tokenizer.tokenize("MKTIIALSYIMKTIIALSYIMKTIIALSYI", 15)
    print(tokens)
    print(mask)
    print('='*50)

    # Test simple tokenize
    tokens, mask = simple_tokenize("MKTIIALSYIMKTIIALSYIMKTIIALSYI", 15)
    print(tokens)
    print(mask)
    print('='*50)