# Detailed Human Shape Estimation from a Single Image by Hierarchical Mesh Deformation

[Hao Zhu](https://zhuhao-nju.github.io/), [Xinxin Zuo](https://sites.google.com/site/xinxinzuohome/), [Sen Wang](https://sites.google.com/site/senwang1312home/), [Xun Cao](https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html), [Ruigang Yang](https://orcid.org/0000-0001-5296-6307) &nbsp; &nbsp; CVPR 2019 Oral

**[[Project Page]](http://cite.nju.edu.cn/Researches/3DCaptureandReconstruction/20190621/i5141.html)** &nbsp; &nbsp; **[[Arxiv]](https://arxiv.org/abs/1904.10506)**

<img src="https://github.com/zhuhao-nju/hmd/blob/master/demo/results/2726.gif" width="200"> <img src="https://github.com/zhuhao-nju/hmd/blob/master/demo/results/0002.gif" width="200"> <img src="https://github.com/zhuhao-nju/hmd/blob/master/demo/results/0477.gif" width="200"> <img src="https://github.com/zhuhao-nju/hmd/blob/master/demo/results/2040.gif" width="200">

From green bounded frame:
**source image** --> **initial guess** --> **joint deform** -->  **anchor deform** --> **vertex deform**

## Requirements
The project is tested on ubuntu 16.04 with python 2.7, PyTorch 1.0.  We recomend using [Anaconda](https://www.anaconda.com/download/#linux) to create a new enviroment:
```
conda create -n py27-hmd python=2.7
conda activate py27-hmd
```

Install dependecies:
```
sudo apt-get install libsuitesparse-dev
pip install -r requirements.txt
```

The installation of OpenDR is unstable now, we recommend using a old stable version:
```
sudo apt-get install python-dev python-pip cython python-numpy \
    python-scipy python-matplotlib libopencv-dev python-opencv \
    libosmesa6-dev freeglut3-dev
pip install pip==8.1.1
pip install opendr==0.78
pip install --upgrade pip
```

Refer to the [guide](https://pytorch.org/get-started/locally/) to install the PyTorch 1.0.

## Demo
Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1LaQBoJPvrqAkF2AGzeV97lmMu9OhsGAS/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/11NpU9NAiO6KOHveWo6tRAg)(extracting code:q23f), place the file in "/hmd/demo/", then extract the pretrained model:
```
cd demo
chmod +x download_pretrained_model.sh
./download_pretrained_model.sh
```
Run the demo:
```
python demo.py --ind 2 # or 477, 2040, 2726
```
The results will be saved in the folder "demo/results/" by default.  Run "python demo.py -h" for more usages.

This repository merely contains 4 samples for demo. To run the full test data, download the test set from [Google Drive](https://drive.google.com/open?id=1ifcvLFJb1t9uS9bz0CxqhaYUfXvQNHC4) or [Baidu Netdisk](https://pan.baidu.com/s/1OVfM4ETgkFiUgmGpp0Cb4A)(extracting code:0ch3).  Extract the test set and change the "dataset_path" in "conf.ini" to the extracted location.  The range of test data number is [0-4624].  You can also follow the instructions in the "Data preparation" part to generate testing data together with training data.

In the generation of the dataset, we predicted the initial mesh using [HMR](https://github.com/akanazawa/hmr) and saved it as "/para/\*.json" files.  To test on images beyond the dataset, you have to run HMR to get the initial mesh firstly.

## Demo wild
This demo runs for images out of the dataset.  Please see [demo_wild/demo_wild.md](/demo_wild/demo_wild.md).

## Data preparation
Please see [datasets/data.md](/datasets/data.md) for detail.

## Training
After data preparation, run the traning of anchor_net and joint_net:
```
conda activate py27-hmd
cd src
python ./train_joint.py
python ./train_anchor.py
python ./train_shading.py
```
If the training data location changed, the "tgt_path" in "/conf.ini" should be changed accordingly.

## Evaluation
Please see [eval/eval.md](/eval/eval.md) for detail.

## Citation
If you find this project useful for your research, please consider citing:
```
@article{zhu2022detailed,
  title={Detailed Avatar Recovery from Single Image},
  author={Zhu, Hao and Zuo, Xinxin and Yang,Haotian and Wang, Sen and Cao, Xun and Yang, Ruigang},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={7363--7379},
  year={2022},
}
```

```
@inproceedings{zhu2019detailed,
  title={Detailed human shape estimation from a single image by hierarchical mesh deformation},
  author={Zhu, Hao and Zuo, Xinxin and Wang, Sen and Cao, Xun and Yang, Ruigang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4491--4500},
  year={2019}
}
```

