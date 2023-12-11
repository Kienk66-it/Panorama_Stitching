# Panorama Stitching Image using SIFT, KNN, and RANSAC
## How a Panorama Image is created?
![The steps of Panorama process](./img/process.png)
From two image paths (left & right) create a panorama from them.
* Assign two string as images path
* Read image from image path
* Detect Keypoint and Descriptors by SIFT
* Finding potential matches by KNN
* Computing Homography matrix, and remove outliers by RANSAC
* Warping and Merging and Plotting a complete panorama image
* Plot out panorama image 

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.
```bash
pip install numpy matplotlib opencv-python random tqdm scipy
```

## Usage
```bash
python pano-stitch.py
```
