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

In order to run the model training process, please run the following -ETA 25 mins.

```
python run_train.py locking.cxr_filepath data/cxr.h5 locking.txt_filepath data/mimic_impressions.csv
```

We support the use of hydra for our experiments, so you can also run:

```
python run_train.py locking={desired_configuration, e.g. Uu}
```

and pick the desired configuration to run the training of any of the models!

In order to deal with memory issues with data downloading please use:

```
sync; echo 1 > /proc/sys/vm/drop_caches
```

This will write RAM data to disk using the sync operation and free up CPU space that can help speed up training and avoid issues regarding memory during model checkpointing. 

# Experiments

For our experiments we are testing out different combinations of the CheXzero vision and text stacks with our experimental vision and text stacks. By default, when you run the training command from above you will be training the CheXzero model. Our experiments also involve either training or locking the two stacks of our model. By default, the training command leaves both parts of the model unlocked. However, you can include different parameter flags in your training command to make the adjustments that you would prefer. As a reminder, the basic training command is the following.

```
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv
```

In order to run the model training process with the CXR-BERT model for the text stack, include the following at the end of the command.

```
locking.use_huggingface_bert=True
```

In order to run the model training process with the ViTMAE image stack you need to include two flags. The first is a boolean flag to indicate you want to use it and the second is a path for the model checkpoint. An example of these flags is as follows.

```
locking.use_vitmae=True locking.vitmae_path="./masked-autoencoders/output_dir/checkpoint-0.pth"
```

In order to train the model with a locked text stack, include the following at the end of the training command.

```
locking.lock_text=True
```

In order to train the model with a locked vision stack, include the following at the end of the training command.

```
locking.lock_vision=True
```

A combination of the above parameter flags may be used to achieve the appropriate experimental training settings. We now list out the full commands for the various experiment options (as denoted by our two character notation). We have since made this process easier (see above), you can run any experiments by just specifying the locking parameter and let hydra do the work!

```
# uu
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv
# uU
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_huggingface_bert
# uL
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_huggingface_bert --lock_text
# Uu
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_vitmae --vitmae_path {ViTMAE_CHECKPOINT_PATH}
# Lu
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_vitmae --vitmae_path {ViTMAE_CHECKPOINT_PATH} --lock_vision
# UU
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_vitmae --vitmae_path {ViTMAE_CHECKPOINT_PATH} --use_huggingface_bert
# LU
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_vitmae --vitmae_path {ViTMAE_CHECKPOINT_PATH} --lock_vision --use_huggingface_bert
# UL
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_vitmae --vitmae_path {ViTMAE_CHECKPOINT_PATH} --use_huggingface_bert --lock_text
# LL
python run_train.py --cxr_filepath data/cxr.h5 --txt_filepath data/mimic_impressions.csv --use_vitmae --vitmae_path {ViTMAE_CHECKPOINT_PATH} --lock_vision --use_huggingface_bert --lock_text
```

# Zero-shot Dataset Downloads

All information to setup and download CheXpert, VinDr-CXR, and Padchest datasets located in `data/README.md`.

# Pre-processing

```
python run_preprocess.py --csv_out_path=./data/CheXpert/test_labels.csv --dataset_type='chexpert-test' --chest_x_ray_path='./data/CheXpert/'
```

# Running Automated Zero-Shot Evaluation

The zeroshot_analysis.py python file provides bootstrapped results for a specified set of model pointers. The files takes in several arguments, including:

`--cxr_filepath`, Path to test data for zero-shot analysis.

`--cxr_labels`, Path to true labels for zero shot eval.

`--model_dir`, Directory path for model pointers that need to be evaluated.

`--use_huggingface_bert`, Boolean flag for indicating model type.

`--results_dir`, Location for storing pandas dataframes as CSVs for bootstrapped results.
