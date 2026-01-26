import os
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

import grain.python as grain
from transformers import EsmTokenizer
# Import MHC sequences from constants
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from constants import MHC_PROT_SEQ
except ImportError:
    MHC_PROT_SEQ = {}


AA_VOCAB = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
    '<PAD>': 21
}


def simple_tokenize(
    sequence: str,
    max_length: int,
    pad_token: int = 21
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple tokenization returning NumPy arrays."""
    tokens = [AA_VOCAB.get(aa.upper(), AA_VOCAB['X']) for aa in sequence]
    
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    seq_len = len(tokens)
    mask = [(t != AA_VOCAB['X']) for t in tokens] + [False] * (max_length - seq_len)
    tokens = tokens + [pad_token] * (max_length - seq_len)
    
    return np.array(tokens, dtype=np.int32), np.array(mask, dtype=np.bool_)


class ESMTokenizer:
    """ESM-2 tokenizer wrapper."""
    
    _instance = None
    _model_name = None
    
    @classmethod
    def get_instance(cls, model_name: str = "facebook/esm2_t30_150M_UR50D"):
        """Singleton pattern for tokenizer (avoid re-loading in workers)."""
        if cls._instance is None or cls._model_name != model_name:
            cls._instance = EsmTokenizer.from_pretrained(model_name)
            cls._model_name = model_name
        return cls._instance
    
    def __init__(self, model_name: str = "facebook/esm2_t30_150M_UR50D"):
        self.model_name = model_name
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.get_instance(self.model_name)
        return self._tokenizer
    
    def tokenize(
        self,
        sequence: str,
        max_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='np'
        )
        
        tokens = encoded['input_ids'].squeeze(0).astype(np.int32)
        mask = encoded['attention_mask'].squeeze(0).astype(np.bool_)
        
        x_token_id = self.tokenizer.convert_tokens_to_ids('X')
        mask = mask & (tokens != x_token_id)
        
        return tokens, mask


@dataclass
class DataConfig:
    """Shared data configuration."""
    tokenizer_type: str = 'esm2'
    esm_model_name: str = 'facebook/esm2_t30_150M_UR50D'
    max_peptide_length: int = 15
    max_mhc_length: int = 385
    use_flanks: bool = False
    flank_length: int = 5
    flank_mask_prob: float = 0.0
    binarize_labels: bool = False
    label_threshold: float = 0.5
    mhc_column: str = 'mhc'
    peptide_column: str = 'peptide'
    label_column: str = 'label'
    seed: int = 42


class MHCPeptideDataSource:
    """
    Grain-compatible data source for MHC-peptide data.
    """

    def __init__(
        self,
        data_path: str,
        config: DataConfig,
        mhc_seq_path: Optional[str] = None,
    ):
        self.config = config
        self.data_path = data_path

        # Load data lazily
        self._df = None
        self._mhc_sequences = None
        self._tokenizer = None
        self._mhc_seq_path = mhc_seq_path

        # Pre-calculate lengths
        if config.use_flanks:
            max_peptide_with_flanks = config.max_peptide_length + 2 * config.flank_length
        else:
            max_peptide_with_flanks = config.max_peptide_length

        if config.tokenizer_type == 'esm2':
            self.effective_peptide_length = max_peptide_with_flanks + 2
            self.effective_mhc_length = config.max_mhc_length + 2
        else:
            self.effective_peptide_length = max_peptide_with_flanks
            self.effective_mhc_length = config.max_mhc_length
        
        # Get length without loading full dataframe
        self._length = len(pd.read_csv(data_path))

        # Set seed for reproducibility
        np.random.seed(config.seed)
    
    @property
    def df(self):
        """Lazy load dataframe."""
        if self._df is None:
            self._df = pd.read_csv(self.data_path)
        return self._df
    
    @property
    def mhc_sequences(self):
        """Lazy load MHC sequences."""
        if self._mhc_sequences is None:
            self._mhc_sequences = MHC_PROT_SEQ.copy()
            if self._mhc_seq_path and os.path.exists(self._mhc_seq_path):
                mhc_df = pd.read_csv(self._mhc_seq_path)
                self._mhc_sequences.update(dict(zip(mhc_df['allele'], mhc_df['sequence'])))
        return self._mhc_sequences
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None and self.config.tokenizer_type == 'esm2':
            self._tokenizer = ESMTokenizer(self.config.esm_model_name)
        return self._tokenizer
    
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get item by index."""
        row = self.df.iloc[idx]

        # Prepare peptide
        peptide = row[self.config.peptide_column]
        nflank_len = 0
        cflank_len = 0
        original_len = len(peptide)

        if self.config.use_flanks:
            nflank = row.get('nflank', '')
            cflank = row.get('cflank', '')
            
            nflank = str(nflank)[-self.config.flank_length:] if pd.notna(nflank) else ''
            cflank = str(cflank)[:self.config.flank_length] if pd.notna(cflank) else ''

            if self.config.flank_mask_prob > 0.0 and np.random.uniform() < self.config.flank_mask_prob:
                if len(nflank) > 0:
                    n_mask_positions = np.random.randint(0, len(nflank) + 1)
                    nflank = 'X' * n_mask_positions + nflank[n_mask_positions:]

                if len(cflank) > 0:
                    c_mask_positions = np.random.randint(0, len(cflank) + 1)
                    cflank = cflank[:len(cflank) - c_mask_positions] + 'X' * c_mask_positions
            
            # Check if extended sequence would exceed max length
            # max_peptide_length is for core peptide only
            total_len = len(nflank) + len(peptide) + len(cflank)
            max_total = self.config.max_peptide_length + 2 * self.config.flank_length

            if total_len > max_total:
                available_for_flanks = max_total - len(peptide)
                if available_for_flanks <= 0:
                    nflank = ''
                    cflank = ''
                else:
                    max_flank = available_for_flanks // 2
                    nflank = nflank[-max_flank:] if len(nflank) > max_flank else nflank
                    remaining = available_for_flanks - len(nflank)
                    cflank = cflank[:remaining] if len(cflank) > remaining else cflank

            nflank_len = len(nflank)
            cflank_len = len(cflank)

            peptide = nflank + peptide + cflank

        
        # Get MHC sequence
        mhc_allele = row[self.config.mhc_column]
        mhc_seq = self.mhc_sequences.get(mhc_allele, "X" * self.config.max_mhc_length)

        # Get label
        label = row[self.config.label_column]
        if self.config.binarize_labels:
            label = 1.0 if label >= self.config.label_threshold else 0.0
        
        # Tokenize
        if self.config.tokenizer_type == 'esm2':
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
            'label': np.float32(label),
            'idx': np.int32(idx),
            'nflank_len': np.int32(nflank_len),
            'cflank_len': np.int32(cflank_len),
            'original_peptide_len': np.int32(original_len),
        }


@dataclass
class GrainDataModule:
    """
    Data module using Grain for optimal JAX/TPU data loading.
    """
    data_dir: str
    fold: int = 0
    batch_size: int = 32
    mhc_seq_path: Optional[str] = None
    tokenizer_type: str = 'esm2'
    esm_model_name: str = 'facebook/esm2_t30_150M_UR50D'
    max_peptide_length: int = 15
    max_mhc_length: int = 385
    use_flanks: bool = False
    flank_length: int = 5
    flank_mask_prob: float = 0.0
    binarize_labels: bool = False
    label_threshold: float = 0.5
    seed: int = 42
    num_workers: int = 4
    prefetch_size: int = 2
    test_files: List[str] = field(default_factory=list)

    # Data sources (populated by setup())
    train_source: Optional[MHCPeptideDataSource] = field(default=None, init=False)
    val_source: Optional[MHCPeptideDataSource] = field(default=None, init=False)
    test_sources: Dict[str, MHCPeptideDataSource] = field(default_factory=dict, init=False)

    def _create_config(self, flank_mask_prob) -> DataConfig:
        """Create data configuration."""
        return DataConfig(
            tokenizer_type=self.tokenizer_type,
            esm_model_name=self.esm_model_name,
            max_peptide_length=self.max_peptide_length,
            max_mhc_length=self.max_mhc_length,
            use_flanks=self.use_flanks,
            flank_length=self.flank_length,
            flank_mask_prob=flank_mask_prob,
            binarize_labels=self.binarize_labels,
            label_threshold=self.label_threshold,
            seed=self.seed,
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup data sources."""
        data_dir = Path(self.data_dir)

        if stage == "fit" or stage is None:
            train_path = data_dir / f"fold_{self.fold}_train.csv"
            val_path = data_dir / f"fold_{self.fold}_val.csv"

            if train_path.exists():
                self.train_source = MHCPeptideDataSource(
                    data_path=str(train_path),
                    config=self._create_config(flank_mask_prob=self.flank_mask_prob),
                    mhc_seq_path=self.mhc_seq_path
                )

            if val_path.exists():
                self.val_source = MHCPeptideDataSource(
                    data_path=str(val_path),
                    config=self._create_config(flank_mask_prob=0.0),
                    mhc_seq_path=self.mhc_seq_path
                )

            if self.test_files:
                for test_file in self.test_files:
                    test_path = Path(test_file)
                    if test_path.exists():
                        self.test_sources[test_path.stem] = MHCPeptideDataSource(
                            data_path=str(test_path),
                            config=self._create_config(flank_mask_prob=0.0),
                            mhc_seq_path=self.mhc_seq_path
                        )
        
        if stage == "test" or stage is None:
            for test_file in self.test_files:
                test_path = Path(test_file)
                if test_path.exists():
                    self.test_sources[test_path.stem] = MHCPeptideDataSource(
                        data_path=str(test_path),
                        config=self._create_config(flank_mask_prob=0.0),
                        mhc_seq_path=self.mhc_seq_path
                    )

    def _create_loader(
        self,
        source: MHCPeptideDataSource,
        shuffle: bool = True,
        shard_options: Optional[grain.ShardOptions] = None
    ) -> grain.DataLoader:
        """Create a Grain dataloader."""

        # Index sampler
        if shuffle:
            sampler = grain.IndexSampler(
                num_records=len(source),
                num_epochs=None,
                shard_options=shard_options or grain.NoSharding(),
                shuffle=True,
                seed=self.seed
            )
        else:
            sampler = grain.IndexSampler(
                num_records=len(source),
                num_epochs=1,
                shard_options=shard_options or grain.NoSharding(),
                shuffle=False
            )

        # Build transformations
        transforms = [
            grain.Batch(
                batch_size=self.batch_size,
                drop_remainder=shuffle
            )
        ]

        # Create dataloader
        loader = grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=transforms,
            worker_count=self.num_workers,
            worker_buffer_size=self.prefetch_size
        )

        return loader

    def train_dataloader(
        self,
        shard_options: Optional[grain.ShardOptions] = None
    ) -> grain.DataLoader:
        """Get training dataloader."""
        return self._create_loader(
            self.train_source,
            shuffle=True,
            shard_options=shard_options
        )

    def val_dataloader(
        self,
        shard_options: Optional[grain.ShardOptions] = None
    ) -> grain.DataLoader:
        """Get validation dataloader."""
        return self._create_loader(
            self.val_source,
            shuffle=False,
            shard_options=shard_options
        )

    def test_dataloader(
        self,
        name: Optional[str] = None,
        shard_options: Optional[grain.ShardOptions] = None
    ) -> Union[grain.DataLoader, Dict[str, grain.DataLoader]]:
        """Get test dataloader(s)."""
        if name is not None:
            return self._create_loader(
                self.test_sources[name],
                shuffle=False,
                shard_options=shard_options
            )
        
        return {
            name: self._create_loader(source, shuffle=False)
            for name, source in self.test_sources.items()
        }

    @property
    def train_dataset_size(self) -> int:
        return len(self.train_source) if self.train_source else 0
    
    @property
    def val_dataset_size(self) -> int:
        return len(self.val_source) if self.val_source else 0

    @property
    def test_dataset_size(self) -> int:
        return {name: len(source) for name, source in self.test_sources.items()}

    @property
    def steps_per_epoch(self) -> int:
        if self.train_source is None:
            return 0
        return len(self.train_source) // self.batch_size

def create_shard_options(
    process_index: Optional[int] = None,
    process_count: Optional[int] = None
) -> grain.ShardOptions:
    """Create shard options for distributed training."""

    if process_index is None:
        process_index = jax.process_index()
    if process_count is None:
        process_count = jax.process_count()

    return grain.ShardOptions(
        shard_index=process_index,
        shard_count=process_count
    )


if __name__ == "__main__":
    import os

    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
    os.environ['TPU_HOST_BOUNDS'] = '1,1,1'

    print(f"JAX devices: {jax.local_devices()}")
    print(f"Process index: {jax.process_index()}")
    print(f"Process count: {jax.process_count()}")
    print()

    dm = GrainDataModule(
        data_dir="/home/cbbl2/digermhc/datasets/MAYBE_FINAL_DATA_23DEC",
        fold=0,
        batch_size=32,
        tokenizer_type="esm2",
        num_workers=0,
        test_files=[
            "/home/cbbl2/digermhc/datasets/MAYBE_FINAL_DATA_23DEC/Experimental_Data_Final.csv",
            "/home/cbbl2/digermhc/datasets/MAYBE_FINAL_DATA_23DEC/COMBINED_ALL_DATASETS_23Dec_TEST.csv"
        ],
        use_flanks=True,
        flank_mask_prob=0.9,
    )

    dm.setup(stage='fit')

    print(f"Train size: {dm.train_dataset_size}")
    print(f"Val size: {dm.val_dataset_size}")
    print(f"Test size: {dm.test_dataset_size}")
    print(f"Steps per epoch: {dm.steps_per_epoch}")
    print()

    # Test iteration
    print("Testing train loader...")
    train_loader = dm.train_dataloader()
    test_loaders = dm.test_dataloader()
    
    # for name, test_loader in test_loaders.items():
    #     print(f"Testing {name} loader...")
    #     for i, batch in enumerate(test_loader):
    #         print(f"Batch {i}:")
    #         for k, v in batch.items():
    #             if hasattr(v, 'shape'):
    #                 print(f"  {k}: {v.shape}")
    #                 print(v)
    #                 print("="*50)
            
    #         break
    #     break
    
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
                print(v)
                print()
        
        break
    
    print("\nGrain DataModule test passed!")