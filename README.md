# EMIT-ECOSTRESS
This repository hosts the code for a project from Caltech's CS101 in Fall 2023. The project goal is to use machine learning to predict urban land surface temperature as measured by the NASA thermal imager ECOSTRESS by using the surface's reflectance spectra as recorded by the NASA hyperspectral imager EMIT. The repository contains code for dataset creation and data processing as well as all modeling and analysis.

## Quick Start
1. Clone this repository.

2. Download the data from the team's Google Drive here: https://drive.google.com/drive/folders/1F0khkxABuI1tzEYNzjSlvQq6dORTq9Zq?usp=drive_link. (Alternatively, you can follow the `data_prep/data_download` notebook `01_Finding_Concurrent_Data_UrbanHeat.ipynb`, followed by the `data_prep/dataset_creation` notebooks `Data_Matching.ipynb`, `Collapsing_Dataset.ipynb`, and `Dataset_Splitting.ipynb` to build your own copy of the dataset.)

3. Place the data in a directory alongside the cloned repository

4. Start a new virtual environment and install the dependencies found in the `requirements.txt` file above using `pip install -r requirements.txt`.

5. Follow the `modeling/Patch_to_Pixel.ipynb` notebook (or the `modeling/CNN.ipynb` notebook) to train your own temperature prediction models.