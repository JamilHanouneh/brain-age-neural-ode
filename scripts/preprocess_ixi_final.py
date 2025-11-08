"""
IXI Preprocessing - Production Ready Version
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
import pickle

# Configuration
TARGET_SHAPE = (96, 112, 96)
PCA_COMPONENTS = 500
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Paths
raw_dir = Path('data/raw/IXI')
processed_dir = Path('data/processed')
metadata_dir = Path('data/metadata')

for d in [processed_dir, metadata_dir]:
    d.mkdir(parents=True, exist_ok=True)

for split in ['train', 'val', 'test']:
    (processed_dir / split).mkdir(exist_ok=True)

print("=" * 70)
print("IXI Preprocessing - Production Version")
print("=" * 70)

# ============================================================
# Step 1: Load and validate metadata
# ============================================================
print("\n[Step 1/5] Loading metadata...")
metadata_file = raw_dir / 'IXI.xls'
metadata = pd.read_excel(metadata_file)

# Ensure IXI_ID is integer
metadata['IXI_ID'] = metadata['IXI_ID'].astype(int)
metadata['AGE'] = pd.to_numeric(metadata['AGE'], errors='coerce')

# Filter out subjects with missing age
metadata_valid = metadata.dropna(subset=['AGE'])
print(f"  ✓ Loaded {len(metadata)} subjects")
print(f"  ✓ Valid subjects (non-NaN age): {len(metadata_valid)}")
print(f"  ✓ Age range: {metadata_valid['AGE'].min():.1f} - {metadata_valid['AGE'].max():.1f} years")

# ============================================================
# Step 2: Find and match files
# ============================================================
print("\n[Step 2/5] Finding and matching files...")
nifti_files = sorted(raw_dir.glob('IXI*-T1.nii.gz'))
print(f"  ✓ Found {len(nifti_files)} NIFTI files")

# Create mapping
file_map = {}
for nifti_file in nifti_files:
    ixi_id = int(nifti_file.name[3:6])
    file_map[ixi_id] = nifti_file

# Keep only files with valid metadata
valid_ids = set(metadata_valid['IXI_ID'].values)
file_map_valid = {k: v for k, v in file_map.items() if k in valid_ids}

print(f"  ✓ Matched {len(file_map_valid)} files to valid metadata")

if len(file_map_valid) == 0:
    print("  ✗ ERROR: No valid files found!")
    sys.exit(1)

# ============================================================
# Step 3: Process files
# ============================================================
print("\n[Step 3/5] Processing files...")

ixi_ids_sorted = sorted(file_map_valid.keys())
n_total = len(ixi_ids_sorted)
n_train = int(TRAIN_RATIO * n_total)
n_val = int(VAL_RATIO * n_total)

processed_data = []
errors = []

for split_idx, ixi_id in enumerate(tqdm(ixi_ids_sorted, desc="Processing")):
    try:
        nifti_file = file_map_valid[ixi_id]
        
        # Get metadata
        meta_row = metadata_valid[metadata_valid['IXI_ID'] == ixi_id].iloc[0]
        age = float(meta_row['AGE'])
        
        # Load image
        nii_img = nib.load(nifti_file)
        img_data = nii_img.get_fdata().astype(np.float32)
        
        # ---- Preprocessing ----
        
        # 1. Skull stripping (simple thresholding)
        threshold = np.percentile(img_data[img_data > 0], 15)
        mask = img_data > threshold
        img_data_masked = np.where(mask, img_data, 0).astype(np.float32)
        
        # 2. Resample to target shape
        zoom_factors = np.array(TARGET_SHAPE) / np.array(img_data_masked.shape)
        img_resampled = zoom(img_data_masked, zoom_factors, order=1)
        
        # Ensure exact target shape (handle floating point errors)
        if img_resampled.shape != TARGET_SHAPE:
            resized = np.zeros(TARGET_SHAPE, dtype=np.float32)
            slices = tuple(slice(0, min(img_resampled.shape[i], TARGET_SHAPE[i])) for i in range(3))
            resized[slices] = img_resampled[slices]
        else:
            resized = img_resampled
        
        # 3. Normalize intensity (Z-score on valid voxels)
        valid_voxels = resized[resized > 0]
        if len(valid_voxels) > 100:
            mean_val = valid_voxels.mean()
            std_val = valid_voxels.std()
            resized = (resized - mean_val) / (std_val + 1e-8)
        
        # ---- Split assignment ----
        if split_idx < n_train:
            split = 'train'
        elif split_idx < n_train + n_val:
            split = 'val'
        else:
            split = 'test'
        
        # ---- Save ----
        subject_name = f"IXI{ixi_id:03d}"
        output_path = processed_dir / split / f"{subject_name}.npy"
        np.save(output_path, resized)
        
        processed_data.append({
            'subject_id': subject_name,
            'ixi_id': ixi_id,
            'age': age,
            'split': split,
            'file_path': str(output_path)
        })
        
    except Exception as e:
        errors.append(f"IXI{ixi_id:03d}: {str(e)}")

print(f"\n  ✓ Successfully processed: {len(processed_data)}")
if errors:
    print(f"  ⚠ Errors encountered: {len(errors)}")
    for err in errors[:3]:
        print(f"      - {err}")

# ============================================================
# Step 4: Save metadata and split info
# ============================================================
print("\n[Step 4/5] Saving metadata...")

if processed_data:
    processed_metadata = pd.DataFrame(processed_data)
    
    # Save CSV
    metadata_csv = metadata_dir / 'processed_metadata.csv'
    processed_metadata.to_csv(metadata_csv, index=False)
    
    # Print statistics
    splits = processed_metadata['split'].value_counts().sort_index()
    print(f"  ✓ Metadata saved to: {metadata_csv}")
    for split_name in ['train', 'val', 'test']:
        count = splits.get(split_name, 0)
        if count > 0:
            ages = processed_metadata[processed_metadata['split'] == split_name]['age']
            print(f"    {split_name:5s}: {count:3d} subjects (age: {ages.min():.1f}-{ages.max():.1f})")

# ============================================================
# Step 5: Fit and save PCA
# ============================================================
print("\n[Step 5/5] Fitting PCA...")

train_meta = processed_metadata[processed_metadata['split'] == 'train']
print(f"  Loading {len(train_meta)} training samples...")

train_arrays = []
for _, row in tqdm(train_meta.iterrows(), total=len(train_meta), desc="Loading", leave=False):
    try:
        arr = np.load(row['file_path'])
        train_arrays.append(arr.flatten())
    except Exception as e:
        print(f"    ✗ Error loading {row['subject_id']}: {str(e)}")

if train_arrays:
    train_arrays = np.array(train_arrays)
    print(f"  ✓ Loaded array shape: {train_arrays.shape}")
    
    # ---- FIXED: Adjust PCA components ----
    n_samples = train_arrays.shape[0]
    n_features = train_arrays.shape[1]
    
    # PCA components cannot exceed min(n_samples, n_features)
    # Usually limited by n_samples for high-dimensional data
    max_components = min(PCA_COMPONENTS, n_samples - 1)
    
    print(f"  Fitting PCA with {max_components} components")
    print(f"    (max possible with {n_samples} samples)")
    
    pca = PCA(n_components=max_components)
    pca.fit(train_arrays)
    
    # Save PCA model
    pca_path = processed_dir / 'pca_model.pkl'
    with open(pca_path, 'wb') as f:
        pickle.dump({
            'pca': pca,
            'shape': TARGET_SHAPE,
            'mean': pca.mean_,
            'components': pca.components_,
            'variance': pca.explained_variance_ratio_,
            'n_components': max_components
        }, f)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  ✓ PCA: {explained_var:.2%} variance explained")
    print(f"  ✓ Saved to: {pca_path}")

print("\n" + "=" * 70)
print("✓ Preprocessing Complete!")
print("=" * 70)
print("\nYou can now run training:")
print("  python scripts/train.py --config config/config.yaml")
print()
