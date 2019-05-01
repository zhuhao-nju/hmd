## Demo wild
This demo runs for images out of the dataset.  HMR repository is required to predict the initial shape. Please follow the instructions [here](https://github.com/akanazawa/hmr/blob/master/README.md) to clone the HMR repository and setup the enviroment, and make sure the basic demo in HMR works.  Then set the HMR location to *HMR_PATH* in "/conf.ini".

The input image is put in the folder "/demo_wild/input".  The input image will be padded to the square, and scaled to 224x224 automatically.  Note that the human should locate roughly in the middle of the image, and is roughly 60% - 70% in height (150 pixel) comparing to the image size (224 pixel).  We use "--crop_x", "--crop_y" and "--pad" to simply crop the source image to satisfy these rules.  [HMR Demo](https://github.com/akanazawa/hmr/blob/master/README.md) also provides an automatic way to scale and crop the image using [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) output.

Refer to the Demo part in README.md to download the pretrained model.

Run the demo for "Duncan.jpg":
```
cd demo_wild
python ./predict_hmr.py --img ./input/Duncan.jpg --crop_x 100 100
python ./predict_hmd.py --img ./input/Duncan.jpg
```

Run the demo for "James.jpg":
```
python ./predict_hmr.py --img ./input/James.jpg
python ./predict_hmd.py --img ./input/James.jpg
```

The results will be saved in the folder "/demo_wild/result/[imgName]/" by default. 

Available options for predict_hmr.py:
+ --img [imgName] &nbsp; &nbsp; # input image name
+ --pad [vaule] &nbsp; &nbsp; # padding length
+ --crop_x [value_left] [value_right] &nbsp; &nbsp; # cropping length in left and right
+ --crop_y [vaule_up] [value_down] &nbsp; &nbsp; # cropping length in upper and bottom
+ --outf [outFolder] &nbsp; &nbsp; # output folder

Available options for predict_hmd.py:
+ --img [imgName] &nbsp; &nbsp; # input image name
+ --GIF True &nbsp; &nbsp; # make gif result
+ --gpu False &nbsp; &nbsp; # use cpu only
+ --mesh True &nbsp; &nbsp; # save obj model
+ --outf [outFolder] &nbsp; # &nbsp; output folder
+ --step False &nbsp; &nbsp; # avoid saving results for each step

[imgName] and [outFolder] for predict_hmr and predict_hmd should be the same.
