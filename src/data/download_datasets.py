"""
Dataset download utilities
"""
import os
import urllib.request
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
from typing import Optional


class DownloadProgressBar(tqdm):
    """Custom progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """
    Download file from URL with progress bar
    
    Args:
        url: URL to download from
        output_path: Path to save file
        chunk_size: Download chunk size
    
    Returns:
        Success flag
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {url}...")
        
        with DownloadProgressBar(
            unit='B',
            unit_scale=True,
            miniters=1,
            desc=output_path.name
        ) as t:
            urllib.request.urlretrieve(
                url,
                str(output_path),
                reporthook=t.update_to,
                data=None
            )
        
        print(f"✓ Downloaded to {output_path}")
        return True
    
    except Exception as e:
        print(f"✗ Download failed: {str(e)}")
        return False


def extract_tar(file_path: str, extract_to: str) -> bool:
    """Extract tar.gz file"""
    try:
        print(f"Extracting {file_path}...")
        
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        
        print(f"✓ Extracted to {extract_to}")
        return True
    
    except Exception as e:
        print(f"✗ Extraction failed: {str(e)}")
        return False


def extract_zip(file_path: str, extract_to: str) -> bool:
    """Extract zip file"""
    try:
        print(f"Extracting {file_path}...")
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✓ Extracted to {extract_to}")
        return True
    
    except Exception as e:
        print(f"✗ Extraction failed: {str(e)}")
        return False


class DatasetDownloader:
    """Download various datasets"""
    
    IXI_BASE_URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"
    OASIS_BASE_URL = "https://www.oasis-brains.org"
    
    @staticmethod
    def download_ixi(output_dir: str = "data/raw/IXI") -> bool:
        """Download IXI dataset"""
        print("\n" + "="*60)
        print("Downloading IXI Dataset")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download T1 images
        t1_url = f"{DatasetDownloader.IXI_BASE_URL}/IXI-T1.tar"
        t1_file = output_path / "IXI-T1.tar"
        
        if not t1_file.exists():
            success = download_file(t1_url, str(t1_file))
            if not success:
                print("Failed to download T1 images")
                return False
        else:
            print(f"✓ T1 file already exists: {t1_file}")
        
        # Extract
        if not (output_path / "IXI-T1").exists():
            extract_tar(str(t1_file), str(output_path))
        
        # Download metadata
        metadata_url = f"{DatasetDownloader.IXI_BASE_URL}/IXI.xls"
        metadata_file = output_path / "IXI.xls"
        
        if not metadata_file.exists():
            download_file(metadata_url, str(metadata_file))
        else:
            print(f"✓ Metadata already exists: {metadata_file}")
        
        print("\n✓ IXI dataset download complete!")
        return True
    
    @staticmethod
    def download_mni152_template(output_dir: str = "data/templates") -> bool:
        """Download MNI152 template"""
        print("\n" + "="*60)
        print("Downloading MNI152 Template")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        url = "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09a_nifti.zip"
        output_file = output_path / "mni_icbm152.zip"
        
        if not output_file.exists():
            success = download_file(url, str(output_file))
            if not success:
                return False
        
        # Extract
        extract_zip(str(output_file), str(output_path))
        
        print("✓ MNI152 template download complete!")
        return True
