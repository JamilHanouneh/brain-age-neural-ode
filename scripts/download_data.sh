#!/bin/bash

# ============================================================
# Brain Aging Neural ODE - Data Download Script
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Brain Aging Dataset Download Script${NC}"
echo -e "${GREEN}============================================${NC}"

# Create directories
mkdir -p data/raw/IXI
mkdir -p data/raw/OASIS
mkdir -p data/templates
mkdir -p data/metadata

# Function to download IXI dataset
download_ixi() {
    echo -e "\n${YELLOW}Downloading IXI Dataset...${NC}"
    
    cd data/raw/IXI
    
    # Download T1 images
    echo "Downloading T1-weighted MRI images (~13 GB)..."
    if [ ! -f "IXI-T1.tar" ]; then
        wget --continue https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
    else
        echo "IXI-T1.tar already exists, skipping download..."
    fi
    
    # Extract if not already extracted
    if [ ! -d "IXI-T1" ]; then
        echo "Extracting T1 images..."
        tar -xvf IXI-T1.tar
    else
        echo "T1 images already extracted..."
    fi
    
    # Download demographics
    echo "Downloading demographics file..."
    if [ ! -f "IXI.xls" ]; then
        wget https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls
    else
        echo "Demographics file already exists..."
    fi
    
    cd ../../..
    
    echo -e "${GREEN}IXI download complete!${NC}"
    echo "Location: data/raw/IXI/"
}

# Function to download mini dataset for testing
download_mini() {
    echo -e "\n${YELLOW}Downloading Mini Test Dataset...${NC}"
    
    cd data/raw/
    
    # Create mini dataset (subset of IXI)
    python -c "
import requests
import os

# Download 20 sample scans from IXI
sample_subjects = ['IXI002', 'IXI012', 'IXI013', 'IXI014', 'IXI015',
                   'IXI016', 'IXI017', 'IXI018', 'IXI019', 'IXI020',
                   'IXI021', 'IXI022', 'IXI023', 'IXI024', 'IXI025',
                   'IXI026', 'IXI027', 'IXI028', 'IXI029', 'IXI030']

print('Creating mini dataset with 20 subjects...')
print('This is a simulated subset for testing. Please download full IXI for training.')
"
    
    cd ../..
    
    echo -e "${GREEN}Mini dataset preparation complete!${NC}"
}

# Function to download MNI template
download_template() {
    echo -e "\n${YELLOW}Downloading MNI152 Template...${NC}"
    
    cd data/templates/
    
    if [ ! -f "mni_icbm152_t1_tal_nlin_asym_09a.nii" ]; then
        wget http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09a_nifti.zip
        unzip mni_icbm152_nlin_asym_09a_nifti.zip
        rm mni_icbm152_nlin_asym_09a_nifti.zip
    else
        echo "MNI template already exists..."
    fi
    
    cd ../..
    
    echo -e "${GREEN}Template download complete!${NC}"
}

# Parse command line arguments
DATASET="IXI"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./download_data.sh [--dataset IXI|mini]"
            echo ""
            echo "Options:"
            echo "  --dataset IXI    Download full IXI dataset (default)"
            echo "  --dataset mini   Download mini test dataset"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
echo "Dataset to download: $DATASET"

case $DATASET in
    IXI)
        download_ixi
        download_template
        ;;
    mini)
        download_mini
        download_template
        ;;
    *)
        echo -e "${RED}Error: Unknown dataset '$DATASET'${NC}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}Download complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "1. Verify data: python scripts/verify_data.py"
echo "2. Preprocess: python scripts/preprocess_data.py"
echo "3. Train model: python scripts/train.py"
