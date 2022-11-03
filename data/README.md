# Downloading Data Instructions

## CheXpert Data

The cheXpert data has a train, val, and test set. Below are the instructions for downloading the **test** set.

Download azcopy (https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy)

Go to https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c to get a Azure download link to the dataset. I had to use Internet explorer because Chrome was not working for some reason when trying to access the data. On the top left of the page, click on the download option and fill out the form, you will see a link that we can use to download the data using azcopy.

Navigate to the folder in which azcopy was downloaded. Now run

```
./azcopy copy <chexpert_link_here> <local_data_directory_here> --recursive=true
```

The local data directory link should point to the folder `data/` inside this repository. This will download two folders `CheXpert` and `chexlocalize`. The `chexlocalize` directory is extra and can now be deleted. We will be using the `CheXpert` dataset for zero-shot evaluation in this project.