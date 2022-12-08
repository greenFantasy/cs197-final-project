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

- You can access the aws ami which is shared under the name full-data-197-final

After launching and instance with at least 500 GiB the ami will come pre-loaded with the data and model training directory.

You can `cd CheXzero` and then access the data in the `cxr_data` folder

All of the data has been preprocessed and the new h5 and csv path files are stored in the `data` folder. 

In order to run pre-processing again, one will have to run the following command: (ETA 4 hrs)

```
python run_preprocessing.py --chest_x_ray_path cxr_data/files --radiology_reports_path cxr_data/reports/files
```

In order to run the model training process, please run the following.

```
python run_train.py --image_tower_type=clip --model_name=baseline
```

In order to pick and run training your desired model configuration, see the experiments section below!

In order to deal with memory issues with data downloading please use:

```
sync; echo 1 > /proc/sys/vm/drop_caches
```

This will write RAM data to disk using the sync operation and free up CPU space that can help speed up training and avoid issues regarding memory during model checkpointing. 

# Experiments

For our experiments we are testing out different combinations of the generalized CheXzero vision and text stacks with our specialized experimental vision and text stacks. By default, when you run the training command from above you will be training the baseline CheXzero model. Our experiments also involve either training or locking the two stacks of our model. By default, the training command leaves both parts of the model unlocked. However, you can include different parameter flags in your training command to make the adjustments (in terms of architecture and locking) that you would prefer. As a reminder, the basic training command is the following.

```
python run_train.py --image_tower_type=clip --model_name=baseline
```

In order to train the model with a locked text stack, include the following at the end of the training command.

```
--lock_text
```

In order to train the model with a locked vision stack, include the following at the end of the training command.

```
--lock_vision
```

The three specialized text stacks we support are CXRBERT, BlueBERT, and ClinicalBERT. To use these in your model you must include both of the following flags in your train command. The keys for the three models are cxr, blue, and clinical respectively.

```
---use_huggingface_bert --huggingface_bert_key={key}
```

The three specialized image stacks we support are BioVision, MedAug, and MoCoCXR. To use these models you must include the following flag in your train command. The types for the three models are biovision, medaug, and mococxr respectively.

```
--image_tower_type={type}
```

A combination of the above parameter flags may be used to achieve the appropriate experimental training settings. We now list out the full commands for the various experiment options.

```
# Baseline
python run_train.py --image_tower_type=clip --model_name=baseline
# CXRBERT locked
python run_train.py --image_tower_type=clip --use_huggingface_bert --huggingface_bert_key=cxr --lock_text --model_name=cxrbert_locked
# CXRBERT unlocked
python run_train.py --image_tower_type=clip --use_huggingface_bert --huggingface_bert_key=cxr --model_name=cxrbert_unlocked
# BlueBERT locked
python run_train.py --image_tower_type=clip --use_huggingface_bert  --huggingface_bert_key=blue --lock_text --model_name=bluebert_locked
# BlueBERT unlocked
python run_train.py --image_tower_type=clip --use_huggingface_bert  --huggingface_bert_key=blue --model_name=bluebert_unlocked
# ClinicalBERT locked
python run_train.py --image_tower_type=clip --use_huggingface_bert  --huggingface_bert_key=clinical --lock_text --model_name=clinicalbert_locked
# ClinicalBERT unlocked
python run_train.py --image_tower_type=clip --use_huggingface_bert  --huggingface_bert_key=clinical --model_name=clinicalbert_unlocked
# BioVision locked
python run_train.py --image_tower_type=biovision --lock_vision --model_name=biovision_locked
# BioVision unlocked
python run_train.py --image_tower_type=biovision --model_name=biovision_unlocked
# MedAug locked
python run_train.py --image_tower_type=medaug --lock_vision --model_name=medaug_locked
# MedAug unlocked
python run_train.py --image_tower_type=medaug --model_name=medaug_unlocked
# MoCoCXR locked
python run_train.py --image_tower_type=mococxr --lock_vision --model_name=mococxr_locked
# MoCoCXR unlocked
python run_train.py --image_tower_type=mococxr --model_name=mococxr_unlocked
```

# Zero-shot Dataset Downloads

All information to setup and download CheXpert, VinDr-CXR, and Padchest datasets located in `data/README.md`.

# Pre-processing

In order to preprocess the CheXpert test dataset, use the following command.

```
python run_preprocess.py --csv_out_path=./data/CheXpert/test_labels.csv --dataset_type='chexpert-test' --chest_x_ray_path='./data/CheXpert/'
```

In order to preprocess the VinDr-CXR test dataset, use the following command.

```
python preprocess_vindr_dataset.py
```

You will potentially have to preprocess the VinDr dataset twice since the image transformations are different for BioVision compared to all the other models. To do this, go into the preprocess_vindr_dataset.py file and change the use_biovision flag to True or False depending on your use case.

# Running Automated Zero-Shot Evaluation

The `generate_chexpert_zeroshot_predictions.py` python file provides model predictions results for a specified set of model pointers. The file takes in several arguments, including:

`--cxr_filepath`, Path to test data for zero-shot analysis.

`--cxr_labels`, Path to true labels csv for zero shot eval.

`--model_path`, Math to model checkpoint that will be used to generate predictions.

`--image_tower_type`, Type of image tower of model (clip, biovision, medaug, mococxr).

`--use_huggingface_bert`, Boolean flag for indicating if a specialized text stack will be used.

`--huggingface_bert_key`, Key for specialized text stack option (cxr, blue, clinical).

`--image_csv_path`, Path to labels csv (which includes image file names or file paths).

`--results_dir`, Location for storing pandas dataframes as CSVs for bootstrapped results.

`--dataset`, Dataset being evaluated on (chexpert or vindr).

`--group`, Flag to generate predictions for all final models at once.

# Bootstrap AUC Results

The `paired_bootstrap.py` file generates AUC results after the predictions have been generated. The file takes in the following arguments.

`--model_names`, List of model names that you will be bootstrapping (eg: baseline,cxrbert_locked).

`--pred_csv_paths`, List of paths to the model prediction csv files.

`--num_samples`, Number of samples to take for the bootstrap.

`--true_label_path`, Path to the csv with the true labels.

`--table`, Flag to print out a latex table of the results.

`--eval_vindr`, Flag to indicate the results are for the VinDr dataset (leaving this out indicates CheXpert).

If exactly two models are included in the arguments, then graphs for paired bootstrap AUC results will be created.
