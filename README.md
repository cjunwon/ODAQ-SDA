# Discoveries of interacting relational patterns between listeners and recordings within Open Dataset of Audio Quality

This repository contains the code and data for the paper "Discoveries of interacting relational patterns between listeners and recordings within Open Dataset of Audio Quality".

## Dataset Information

The dataset used in this study is the Open Dataset of Audio Quality (ODAQ), which contains audio recordings and listener ratings. The dataset is publicly available and can be accessed [here](https://zenodo.org/records/10405774). The additional data from Ball State University (BSU) can also be found [here](https://zenodo.org/records/10405774). Details about the dataset can be found in the original paper's repository [here](https://github.com/Fraunhofer-IIS/ODAQ?tab=readme-ov-file).

The datasets are already included in this repository under the `Data\ODAQ` and `ODAQ_v1_BSU` directories.

## Project Setup Instructions

After cloning the repository, you should create a virtual environment and install the required packages. You can do this by running the following commands:

### 1. Move to the project folder
```bash
cd ODAQ_CEDA  # or the name of your cloned folder
```

### 2. Create a virtual environment and activate it

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For MacOS and Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

## Script Usage

1. `ODAQ_competition_ranking_experts_students.py` and `ODAQ_kmeans_ranking_experts_students.py`

    These scripts are used to analyze the ODAQ dataset and perform ranking based on expert and student ratings. They can be run directly after setting up the environment and installing the required packages. The scripts will generate output files containing the results of the analysis as well as visualizations such as heatmaps, clustermaps, and spaghetti plots. The intermediate results (ranking and clustering) are saved in the `Results` directory as .pkl files.

2. `contingency_table.py`

    This script is used to create a contingency table from the ODAQ dataset. It processes the data and generates a table that summarizes the interactions between listeners and recordings. While the table in the paper displays frequencies, this script allows you to see which recordings and listeners belong to each grouping. The contingency tables are output as CSV files in the `Results` directory.