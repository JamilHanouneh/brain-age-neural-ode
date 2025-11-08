"""
Data preprocessing pipeline for brain MRI
"""
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle


class BrainPreprocessor:
    """Preprocessor for brain MRI data"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocess_config = config['data']['preprocessing']
        self.target_shape = tuple(self.preprocess_config['target_shape'])
        
        # PCA for dimensionality reduction
        self.pca = None
        self.pca_mean = None
    
    def preprocess_dataset(
        self,
        raw_dir: str,
        output_dir: str,
        metadata_file: str
    ):
        """
        Preprocess entire dataset
        
        Args:
            raw_dir: Directory with raw NIFTI files
            output_dir: Directory to save preprocessed data
            metadata_file: Path to metadata CSV/Excel file
        """
        print("\n" + "="*60)
        print("Brain MRI Preprocessing Pipeline")
        print("="*60)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
        
        # Load metadata
        print(f"\n✓ Loading metadata from {metadata_file}")
        if metadata_file.endswith('.csv'):
            metadata = pd.read_csv(metadata_file)
        elif metadata_file.endswith('.xls') or metadata_file.endswith('.xlsx'):
            metadata = pd.read_excel(metadata_file)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_file}")
        
        print(f"  Found {len(metadata)} subjects")
        
        # Split data if not already split
        if 'split' not in metadata.columns:
            metadata = self._split_data(metadata)
        
        # Process each subject
        processed_data = []
        print(f"\n✓ Processing {len(metadata)} subjects...")
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            try:
                # Get subject info
                subject_id = row.get('subject_id', row.get('ID', f'subject_{idx}'))
                age = row.get('age', row.get('AGE', None))
                split = row.get('split', 'train')
                
                if age is None:
                    print(f"  ⚠ Skipping {subject_id}: no age information")
                    continue
                
                # Find raw file
                raw_file = self._find_raw_file(raw_dir, subject_id)
                if raw_file is None:
                    print(f"  ⚠ Skipping {subject_id}: file not found")
                    continue
                
                # Process brain scan
                processed = self._process_single_brain(raw_file)
                
                if processed is None:
                    print(f"  ⚠ Skipping {subject_id}: processing failed")
                    continue
                
                # Save preprocessed data
                output_path = os.path.join(output_dir, split, f"{subject_id}.npy")
                np.save(output_path, processed)
                
                # Store metadata
                processed_data.append({
                    'subject_id': subject_id,
                    'age': age,
                    'split': split,
                    'file_path': output_path
                })
                
            except Exception as e:
                print(f"  ✗ Error processing {subject_id}: {str(e)}")
                continue
        
        # Save processed metadata
        processed_metadata = pd.DataFrame(processed_data)
        metadata_output = os.path.join(
            self.config['data']['metadata_dir'],
            'processed_metadata.csv'
        )
        os.makedirs(os.path.dirname(metadata_output), exist_ok=True)
        processed_metadata.to_csv(metadata_output, index=False)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"  Processed: {len(processed_data)} subjects")
        print(f"  Train: {len(processed_metadata[processed_metadata['split']=='train'])}")
        print(f"  Val: {len(processed_metadata[processed_metadata['split']=='val'])}")
        print(f"  Test: {len(processed_metadata[processed_metadata['split']=='test'])}")
        print(f"  Metadata saved to: {metadata_output}")
        
        # Compute PCA on training data
        if self.preprocess_config.get('pca_components', 0) > 0:
            print(f"\n✓ Computing PCA...")
            self._fit_pca(output_dir, processed_metadata)
    
    def _process_single_brain(
        self,
        file_path: str
    ) -> Optional[np.ndarray]:
        """
        Process a single brain MRI scan
        
        Args:
            file_path: Path to NIFTI file
        
        Returns:
            Preprocessed brain array or None if failed
        """
        try:
            # Load NIFTI file
            nii_img = nib.load(file_path)
            brain_data = nii_img.get_fdata()
            
            # Basic skull stripping (simple threshold-based)
            # In production, use HD-BET or similar
            brain_data = self._simple_skull_strip(brain_data)
            
            # Resize to target shape
            brain_data = self._resize_image(brain_data, self.target_shape)
            
            # Intensity normalization
            if self.preprocess_config['normalize']:
                brain_data = self._normalize_intensity(brain_data)
            
            return brain_data.astype(np.float32)
            
        except Exception as e:
            print(f"Error in _process_single_brain: {str(e)}")
            return None
    
    def _simple_skull_strip(self, image: np.ndarray) -> np.ndarray:
        """
        Simple skull stripping using Otsu's thresholding
        Note: For production, use proper tools like HD-BET
        """
        # Compute threshold
        threshold = np.percentile(image[image > 0], 50)
        
        # Create brain mask
        mask = image > threshold
        
        # Apply mask
        return image * mask
    
    def _resize_image(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Resize image to target shape using SimpleITK"""
        # Convert to SimpleITK image
        sitk_image = sitk.GetImageFromArray(image)
        
        # Calculate new spacing
        original_size = sitk_image.GetSize()
        original_spacing = sitk_image.GetSpacing()
        
        new_spacing = [
            old_spacing * (old_size / new_size)
            for old_spacing, old_size, new_size in zip(
                original_spacing, original_size, target_shape
            )
        ]
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_shape)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled_image = resampler.Execute(sitk_image)
        
        # Convert back to numpy
        return sitk.GetArrayFromImage(resampled_image)
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity"""
        method = self.preprocess_config['normalization_method']
        
        # Get brain voxels (non-zero)
        brain_voxels = image[image > 0]
        
        if len(brain_voxels) == 0:
            return image
        
        if method == 'z-score':
            # Z-score normalization
            mean = brain_voxels.mean()
            std = brain_voxels.std()
            if std > 0:
                image = (image - mean) / std
        
        elif method == 'min-max':
            # Min-max normalization to [0, 1]
            min_val = brain_voxels.min()
            max_val = brain_voxels.max()
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
        
        return image
    
    def _split_data(
        self,
        metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """Split data into train/val/test"""
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        n_samples = len(metadata)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Shuffle
        metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Assign splits
        metadata['split'] = 'test'
        metadata.loc[:n_train, 'split'] = 'train'
        metadata.loc[n_train:n_train+n_val, 'split'] = 'val'
        
        return metadata
    
    def _find_raw_file(
        self,
        raw_dir: str,
        subject_id: str
    ) -> Optional[str]:
        """Find raw NIFTI file for subject"""
        raw_path = Path(raw_dir)
        
        # Common patterns
        patterns = [
            f"{subject_id}*.nii.gz",
            f"{subject_id}*.nii",
            f"*{subject_id}*.nii.gz",
            f"*{subject_id}*.nii"
        ]
        
        for pattern in patterns:
            matches = list(raw_path.rglob(pattern))
            if matches:
                return str(matches[0])
        
        return None
    
    def _fit_pca(self, data_dir: str, metadata: pd.DataFrame):
        """Fit PCA on training data"""
        n_components = self.preprocess_config['pca_components']
        
        # Load training data
        train_data = []
        train_metadata = metadata[metadata['split'] == 'train']
        
        print(f"  Loading {len(train_metadata)} training samples...")
        for _, row in tqdm(train_metadata.iterrows(), total=len(train_metadata)):
            file_path = row['file_path']
            data = np.load(file_path)
            train_data.append(data.flatten())
        
        train_data = np.array(train_data)
        
        # Fit PCA
        print(f"  Fitting PCA with {n_components} components...")
        self.pca = PCA(n_components=n_components)
        self.pca.fit(train_data)
        
        # Save PCA model
        pca_path = os.path.join(self.config['data']['processed_data_dir'], 'pca_model.pkl')
        with open(pca_path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'shape': self.target_shape
            }, f)
        
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"  ✓ PCA fitted: {explained_var:.2%} variance explained")
        print(f"  Saved to: {pca_path}")
    def _get_file_paths(self):
        """Get list of (file_path, age) tuples"""
        file_paths = []

        # Debug: Print column names
        print(f"  Metadata columns: {list(self.metadata.columns)}")

        # Handle different column names
        id_col = None
        age_col = None

        # Try different column name variations
        for col in self.metadata.columns:
            if col.lower() in ['ixi_id', 'subject_id', 'id']:
                id_col = col
            if col.lower() in ['age', 'ages']:
                age_col = col

        if id_col is None or age_col is None:
            print(f"  ⚠ Available columns: {list(self.metadata.columns)}")
            raise ValueError(f"Could not find ID or AGE columns in metadata")

        print(f"  Using columns: ID='{id_col}', AGE='{age_col}'")

        for idx, row in self.metadata.iterrows():
            subject_id = row.get(id_col, None)
            age = row.get(age_col, None)

            if subject_id is None or age is None:
                continue
            
            # Convert subject_id to string for file matching
            subject_id = str(subject_id).strip()

            # Look for processed or raw file
            if self.use_processed:
                file_path = self.data_dir / 'processed' / self.split / f"{subject_id}.npy"
            else:
                # Try to find raw NIFTI file
                raw_dir = self.data_dir / 'raw'

                # Try exact match and pattern matching
                patterns = [
                    f"{subject_id}*.nii.gz",
                    f"{subject_id}*.nii",
                    f"*{subject_id}*.nii.gz",
                    f"*{subject_id}*.nii"
                ]

                file_path = None
                for pattern in patterns:
                    matches = list(raw_dir.rglob(pattern))
                    if matches:
                        file_path = matches[0]
                        break
                    
            if file_path and file_path.exists():
                file_paths.append((file_path, float(age)))

        print(f"  Found {len(file_paths)} matching files")
        return file_paths
    def _load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load metadata file"""
        print(f"  Loading metadata from {metadata_file}")

        if metadata_file.endswith('.csv'):
            df = pd.read_csv(metadata_file)
        elif metadata_file.endswith('.xls') or metadata_file.endswith('.xlsx'):
            try:
                df = pd.read_excel(metadata_file)
            except Exception as e:
                print(f"  ⚠ Error reading Excel: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_file}")

        print(f"  Loaded {len(df)} subjects")
        print(f"  Columns: {list(df.columns)}")

        # Filter by split if column exists
        if 'split' in df.columns:
            df = df[df['split'] == self.split]
            print(f"  Filtered to {len(df)} subjects for '{self.split}' split")

        return df

