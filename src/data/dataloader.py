"""
PyTorch DataLoader for brain MRI data with PCA dimensionality reduction
"""
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm


# Load PCA model once at module level
PCA_MODEL = None

def _load_pca_model():
    """Load PCA model from file"""
    global PCA_MODEL
    if PCA_MODEL is not None:
        return PCA_MODEL
    
    pca_path = Path('data/processed/pca_model.pkl')
    if pca_path.exists():
        try:
            with open(pca_path, 'rb') as f:
                pca_data = pickle.load(f)
                PCA_MODEL = pca_data['pca']
                print(f"  ✓ Loaded PCA model with {PCA_MODEL.n_components_} components")
                return PCA_MODEL
        except Exception as e:
            print(f"  ⚠ Warning: Could not load PCA model: {str(e)}")
            return None
    else:
        print(f"  ⚠ Warning: PCA model not found at {pca_path}")
        return None


def brain_aging_collate_fn(batch):
    """
    Custom collate function that:
    1. Pads images to same size
    2. Flattens them
    3. Applies PCA reduction
    """
    images, ages = zip(*batch)
    
    # Find max dimensions across batch
    max_shape = tuple(max(img.shape[i] for img in images) for i in range(3))
    
    # Process each image
    processed_images = []
    for img in images:
        # Pad to max shape
        pad_width = [(0, max_shape[i] - img.shape[i]) for i in range(3)]
        padded = np.pad(img, pad_width, mode='constant', constant_values=0)
        
        # Flatten
        flattened = padded.flatten().astype(np.float32)
        
        # Apply PCA if available
        pca = _load_pca_model()
        if pca is not None:
            try:
                flattened = pca.transform(flattened.reshape(1, -1))[0].astype(np.float32)
            except Exception as e:
                print(f"  ⚠ PCA transform failed: {str(e)}, using raw flattened data")
        
        processed_images.append(torch.from_numpy(flattened).float())
    
    # Stack into batch tensors
    images_batch = torch.stack(processed_images)
    ages_batch = torch.stack([torch.tensor(age, dtype=torch.float32) for age in ages])
    
    return images_batch, ages_batch


class BrainAgingDataset(Dataset):
    """Dataset for brain aging analysis"""
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        split: str = 'train',
        transform=None,
    ):
        """
        Args:
            data_dir: Directory containing processed brain scans
            metadata_file: CSV file with metadata
            split: 'train', 'val', or 'test'
            transform: Optional data augmentation (not used for now)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_file)
        
        # Get file paths
        self.file_paths = self._get_file_paths()
        
        print(f"    ✓ Loaded {len(self.file_paths)} {split} samples")
    
    def _load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load metadata file"""
        metadata_file = str(metadata_file)  # Convert Path to string
        
        if metadata_file.endswith('.csv'):
            df = pd.read_csv(metadata_file)
        elif metadata_file.endswith('.xls') or metadata_file.endswith('.xlsx'):
            try:
                df = pd.read_excel(metadata_file)
            except ImportError:
                print("  ⚠ Warning: openpyxl/xlrd not available. Install with: pip install openpyxl xlrd")
                df = pd.DataFrame()
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_file}")
        
        # Filter by split if column exists
        if 'split' in df.columns:
            df = df[df['split'] == self.split]
        
        return df
    
    def _get_file_paths(self) -> List[Tuple[Path, float]]:
        """Get list of (file_path, age) tuples"""
        file_paths = []
        
        for _, row in self.metadata.iterrows():
            try:
                # Get file path from metadata
                file_path_str = row.get('file_path', None)
                age = row.get('age', None)
                
                if file_path_str is None or pd.isna(age):
                    continue
                
                # Convert to Path object
                file_path = Path(file_path_str)
                
                # Check if file exists
                if not file_path.exists():
                    continue
                
                file_paths.append((file_path, float(age)))
            
            except Exception as e:
                continue
        
        return file_paths
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        """Get a single sample"""
        file_path, age = self.file_paths[idx]
        
        try:
            # Load preprocessed numpy array
            data = np.load(file_path).astype(np.float32)
            
            # Apply augmentation if available
            if self.transform is not None:
                data = self.transform(data)
            
            return data, age
        
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {str(e)}")
            # Return dummy data on error
            return np.zeros((96, 112, 96), dtype=np.float32), 50.0


class BrainAgingDataModule:
    """Data module for easy data loading"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.train_config = config['training']
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup train/val/test datasets"""
        data_dir = self.data_config['processed_data_dir']
        metadata_file = Path(self.data_config['metadata_dir']) / 'processed_metadata.csv'
        
        # Create datasets
        self.train_dataset = BrainAgingDataset(
            data_dir=data_dir,
            metadata_file=str(metadata_file),
            split='train',
            transform=None
        )
        
        self.val_dataset = BrainAgingDataset(
            data_dir=data_dir,
            metadata_file=str(metadata_file),
            split='val',
            transform=None
        )
        
        self.test_dataset = BrainAgingDataset(
            data_dir=data_dir,
            metadata_file=str(metadata_file),
            split='test',
            transform=None
        )
        
        print(f"\n  ✓ Data module setup complete")
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=brain_aging_collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=brain_aging_collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=brain_aging_collate_fn
        )
