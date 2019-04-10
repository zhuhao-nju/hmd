## Despription
We assemble a quite large dataset (named as WILD dataset in the paper) for training and testing by extracting from 5 human datasets, including Leeds Sport Pose dataset (LSP) and its extension dataset (LSPet), MPII human pose database (MPII), Common Objects in Context dataset (COCO), and Human3.6M dataset (H36M).

We filter out images with incomplete human or low-quality silhouette, and re-arrange the data in the order of "LSP - LSPET - MPII - COCO - H36M".  A data log will be produced during the generation. In each line, the log is written as: 
```
[src_dataset]  [src_num]  [data_type]  [new_num]
```
+ [src_dataset]: enum {LSP, LSPET, MPII, COCO, H36M}.

+ [src_num]: ID number in the source dataset.

+ [data_type]: enum {TRAIN (select as training data), TEST (select as testing data), BAN (not selected)}

+ [new_num]: ID number in the WILD dataset.

This will help trace the source of each tuple of the data.

## HMR configure
We use HMR to predict the initial shape for our method.  In order to speed up the training, we pre-compute the HMR results and save them in the dataset.  The HMR results include predicted keypoint position, SMPL mesh, rendered silhouette and other related attributes.  HMR repository is required during the data processing. Please follow the instructions [here](https://github.com/akanazawa/hmr/blob/master/README.md) to setup the enviroment for HMR, and make sure the basic demo in HMR works.  Then set the HMR location to *HMR_PATH* in "/conf.ini".

## Data acquire
Download and extract the source data, then set the path in the [DATA] part of "/conf.ini" as following:

+ **LSP**. Acquire the data from http://sam.johnson.io/research/lsp.html by clicking "Download (ZIP file)". Extract the data and set the extracting path to *lsp_path*.

+ **LSPet**. Acquire the data from http://sam.johnson.io/research/lspet.html by clicking "Dataset (ZIP file)". Extract the data and set the extracting path to *lspet_path*.

+ **UP** (contains MPII Human Shape data). Acquire the data from http://files.is.tuebingen.mpg.de/classner/up/ by clicking "UPi-S1h (26,294 images, 44.3Gb)". Extract the data and set the extracting path to *upi_path*.

+ **H36M**. Acquire the data from http://vision.imar.ro/human3.6m/description.php. Register and apply for the access to the data. The selectedd data are listed in "h36m_list.txt". Extract the data and set the extracting path to *h36m_path*.  The joints in H36M dataset are saved in cdf files.  As the cdf accessor in python is quite difficult to install, we use matlab to tansform all the *.cdf* files to *.mat* files, which are easier to access in python.  The transforming script is saved in *h36m_cdf2mat.m*.

+ **COCO**. Visit http://cocodataset.org/#download to download the COCO API, then set the API path to *coco_api_path*.   COCO data is stored online, so the data will be downloaded through the API while processing.  Download the data list (json file) for densepose as following and set the path to *coco_list_path*:
```
cd $certain path$
wget https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_train.json
```
**Note:** Please follow the license agreements of the above-mentioned datasets.

## Run the processing
Finally, set the path to save the dataset in *tgt_path* and run the processing:
```
cd datasets
python proc_all_data.py
```
The dataset requires 26 GB space, and the processing will last for roughly 11 hours.  We provide the download links of testing set in *Demo* part.  
