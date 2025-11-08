"""
Verify downloaded data integrity
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import pandas as pd


def verify_ixi_data(data_dir='data/raw/IXI'):
    """Verify IXI dataset"""
    print("\n" + "="*60)
    print("Verifying IXI Dataset")
    print("="*60)
    
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        print(f"✗ Error: Directory not found: {data_dir}")
        return False
    
    # Check for NIFTI files
    nifti_files = list(data_path.glob("*.nii.gz")) + list(data_path.glob("*.nii"))
    print(f"\n✓ Found {len(nifti_files)} NIFTI files")
    
    if len(nifti_files) == 0:
        print("✗ Error: No NIFTI files found. Please download IXI-T1.tar")
        return False
    
    # Check for metadata file
    metadata_files = list(data_path.glob("*.xls")) + list(data_path.glob("*.xlsx"))
    if len(metadata_files) > 0:
        print(f"✓ Found metadata file: {metadata_files[0].name}")
        
        # Try to read metadata
        try:
            df = pd.read_excel(metadata_files[0])
            print(f"✓ Metadata loaded: {len(df)} subjects")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"⚠ Warning: Could not read metadata: {str(e)}")
    else:
        print("⚠ Warning: No metadata file found (IXI.xls)")
    
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    print(f"✓ Data directory: {data_path}")
    print(f"✓ NIFTI files: {len(nifti_files)}")
    print(f"✓ Ready for preprocessing!")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Verify downloaded data')
    parser.add_argument(
        '--dataset',
        type=str,
        default='IXI',
        help='Dataset to verify (IXI or OASIS)'
    )
    args = parser.parse_args()
    
    if args.dataset == 'IXI':
        verify_ixi_data()
    else:
        print(f"Verification for {args.dataset} not yet implemented")


if __name__ == '__main__':
    main()
