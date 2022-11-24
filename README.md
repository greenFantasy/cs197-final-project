# cs197-final-project

This is the repo for our CS 197 final project. 

# Setup (osX only currently)

After cloning the directory, please run:

```
conda create --name <env> --file requirements.txt
```

If adding packages to the directory, please update the `requirements.txt`:
```
conda list --explicit > requirements.txt
```
from the repo root directory.

### Data and Model Training Setup with AWS - Aakash Mishra (aakamishra)

- You can access the aws ami which is shared under the name complete-data-cs197

After launching and instance with at least 500 GiB the ami will come pre-loaded with the data and model training directory.

You can `cd CheXzero` and then access the data in the `cxr_data` folder

All of the data has been preprocessed and the new h5 and csv path files are stored in the `data` folder. 

In order to run pre-processing again, one will have to run the following command: (ETA 4 hrs)

```
python run_preprocessing.py --chest_x_ray_path cxr_data/files --radiology_reports_path cxr_data/reports/files
```

In order to run the model training process, please run the following -ETA 25 mins:

```
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv
```

In order to run the model training process with the original CheXzero model, please run the following.

```
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_chexzero_text
```

In order to deal with memory issues with data downloading please use:

```
sync; echo 1 > /proc/sys/vm/drop_caches
```

This will write RAM data to disk using the sync operation and free up CPU space that can help speed up training and avoid issues regarding memory during model checkpointing. 

# Zero-shot Dataset Downloads

All information to setup and download cheXpert and Padchest datasets located in `data/README.md`.

# Pre-processing

python run_preprocess.py --csv_out_path=./data/CheXpert/test_labels.csv --dataset_type='chexpert-test' --chest_x_ray_path='./data/CheXpert/'