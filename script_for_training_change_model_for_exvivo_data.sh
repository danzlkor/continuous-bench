#!/bin/bash

# Local script for training change models with 20,000 samples

# Directories
rootpath="$(pwd)"  # Set root path to current directory
acqnpz="$rootpath/data/acquisition_params_for_exvivo_data.npz" # Acquisition parameters file
outputdir="$rootpath/change_model/"  # Output directory

# Create output directory if it doesn't exist
mkdir -p "$outputdir"

# Parameters
num_samples=20000  # Number of samples

# Display information
echo "Running training script locally"
echo "Samples: $num_samples"
echo "Output directory: $outputdir"

# Run the training script
python training_change_model_for_exvivo_data.py \
    -ad "$acqnpz" \
    -num "$num_samples" \
    -od "$outputdir"