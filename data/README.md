# Downloading and Processing Data Instructions

## CheXpert Data

The cheXpert data has a train, val, and test set. Below are the instructions for downloading the **test** set.

Download azcopy (https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy)

Go to https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c to get a Azure download link to the dataset. I had to use Internet explorer because Chrome was not working for some reason when trying to access the data. On the top left of the page, click on the download option and fill out the form, you will see a link that we can use to download the data using azcopy.

Navigate to the folder in which azcopy was downloaded. Now run

```
./azcopy copy <chexpert_link_here> <local_data_directory_here> --recursive=true
```

The local data directory link should point to the folder `data/` inside this repository. This will download two folders `CheXpert` and `chexlocalize`. The `chexlocalize` directory is extra and can now be deleted. We will be using the `CheXpert` dataset for zero-shot evaluation in this project.

## VinDr-CXR Data

You can find information about the VinDr-CXR dataset here: https://www.nature.com/articles/s41597-022-01498-w. Also, you can download the dataset from here: https://physionet.org/content/vindr-cxr/1.0.0/. We will not be using the train data so you can refrain from downloading it. 

We will work under the assumption that once the data has been downloaded, it is found within the `vindr-cxr/{version}` folder. In this case the dataset version is 1.0.0 so the data is found in the `vindr-cxr/1.0.0` folder. From there the structure will be as shown on PhysioNet.

We will then need to process the dataset with the following command.

```
python run_preprocess.py --csv_out_path data/vindr_cxr_paths.csv --cxr_out_path data/vindr_cxr.h5 --dataset_type vindr --chest_x_ray_path vindr-cxr/1.0.0/test
```

This will create a `vindr_cxr.h5` file in the `data` folder.
