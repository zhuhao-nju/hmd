## Evaluation

This part generates quantitative evaluation results on WILD, RECON, and SYN testing datasets.  For WILD test, 2D metrics including 2D joint error and silhouette IoU are computed; For RECON and SYN test, 3D surface error and silhouette IoU are computed.

### WILD test

Download Wild dataset from google [Google Drive](https://drive.google.com/open?id=1ifcvLFJb1t9uS9bz0CxqhaYUfXvQNHC4) or [Baidu Netdisk](https://pan.baidu.com/s/1OVfM4ETgkFiUgmGpp0Cb4A)(extracting code:0ch3) (the same links as demo part), save the downloaded zip file to the folder "/eval/eval_data/".  Extract the data:
```
mkdir ./eval_data/wild_set/
unzip ./eval_data/wild_set/test.zip -d ./eval_data/wild_set/
```
The enviroment settings are the same as the demo part in [README.md](/README.md).  Please make sure the demo runs properly before next step.  Run the evaluation:
```
conda activate py27-hmd
python eval_wild.py
```
This will take several hours.  The reports are stored in "/eval/eval_reports/wild_reports.txt".  The meaning of each column are shown in the header of the generated txt files.  The mean values are save in the last line.  Note that the evaluation on wild test doesn't predict shading part, as the shading part shows the same performance in 2D metrics as the anchor-deformed mesh.

### RECON and SYN test

Download RECON and SYN sets from [Google Drive](https://drive.google.com/file/d/1hWsMwcDw5FX1lRyR8qGkNLGJ4pNhpNFL/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1ZAj3E2m0WmjhCTmLpu8uDQ)(extracting code:8tji), save the zip file to the folder "/eval/eval_data/".  Extract the data:
```
unzip ./eval_data/eval_recon_and_syn_sets.zip -d ./eval_data/
```

Install open3D for 3D registration.
```
conda install -c open3d-admin open3d==0.3

```
The enviroment settings are the same as the demo wild part in [demo_wild/demo_wild.md](/demo_wild/demo_wild.md).  Please make sure the demo wild runs properly before next step.  Run the evaluation on the eval set:
```
conda activate py27-hmd
python pred_hmr.py --set recon --num 150
python pred_hmd_ja.py --set recon --num 150
python pred_hmd_s.py --set recon --num 150
python eval_recon.py --tgt hmr
python eval_recon.py --tgt j
python eval_recon.py --tgt a
python eval_recon.py --tgt s
```

Run the evaluation on the recon set:
```
conda activate py27-hmd
python pred_hmr.py --set syn --num 300
python pred_hmd_ja.py --set syn --num 300
python pred_hmd_s.py --set syn --num 300
python eval_syn.py --tgt hmr
python eval_syn.py --tgt j
python eval_syn.py --tgt a
python eval_syn.py --tgt s
```
The mean error will be shown after evaluation, and also be saved in "./eval_reports/*.txt".  The meaning of each column are shown in the header of the generated txt files.  The mean values are save in the last line.
