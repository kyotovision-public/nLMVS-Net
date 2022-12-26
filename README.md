# nLMVS-Net

This repository provides an inplementation of our paper nLMVS-Net: Deep Non-Lambertian Multi-View Stereo in WACV 2023. If you use our code and data please cite our paper.

Please note that this is research software and may contain bugs or other issues – please use it at your own risk. If you experience major problems with it, you may contact us, but please note that we do not have the resources to deal with all issues.

```
@InProceedings{Yamashita_2023_WACV,
    author    = {Kohei Yamashita and Yuto Enyo and Shohei Nobuhara and Ko Nishino},
    title     = {nLMVS-Net: Deep Non-Lambertian Multi-View Stereo},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {Jan},
    year      = {2023}
}
```

## Prerequisites

We tested our code with Python 3.7.6 on Ubuntu 20.04 LTS. Our code depends on the following modules.

* numpy
* opencv-python
* pytorch
* numba
* tqdm
* meshlabserver
* moderngl
* matplotlib
* open3d
* trimesh

You can use `nlmvsnet.def` to build your singularity container by
```
$ singularity build --fakeroot nlmvsnet.sif nlmvsnet.def
```

Also, please prepare the following files.
* Download ```module.py``` of [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch/tree/e0f2ae3d7cb2dd13807b775f2075682eaa7f1521) and save it to ```./core```.
* Download ```alum-bronze.pt``` from [MERL BRDF Database](https://www.merl.com/brdf/) and save it to ```./data```.
* Download ```ibrdf.pt``` from [here](https://drive.google.com/drive/folders/1IWr1KXGxYMEUIHxOobygxApA6_UTZYmD?usp=share_link) and save it to ```./data```.
* Download ```merl_appearance_ratio.pt``` and ```merl_mask.pt``` from [here](https://drive.google.com/drive/folders/1IWr1KXGxYMEUIHxOobygxApA6_UTZYmD?usp=share_link) and save them to ```./core/ibrdf/render```.

We provide pretrained weights for our networks.
* Download pretrained weight files from [here](https://drive.google.com/drive/folders/1IWr1KXGxYMEUIHxOobygxApA6_UTZYmD?usp=share_link) and save them to ```./weights/sfsnet``` and ```./weights/nlmvsnet```.

## nLMVS-Synth and nLMVS-Real datasets

### License

The nLMVS-Synth and nLMVS-Real datasets are provided under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

### Download

We provide the raw and also preprocessed data.  As the raw data is so large, we recommend you to use the preprocessed data which contains HDR images.

* nLMVS-Synth
  * Training Set
    * [Training Data for Shape-from-Shading Network (6.6GB)](https://drive.google.com/uc?id=1E9Uk3RTYPnwsKNjkLsO_XoPzwpCsRWzN)
    * [Training Data for Cost Volume Filtering Network (22GB)](https://drive.google.com/uc?id=1hzTYRSgP5mqq6s3rXDpoxkBUgX1viAf3)
  * [Test set (39GB)](https://drive.google.com/uc?id=12Mkcbe2JuJTrJRfKbXhg_buuQvbHM6FW)
* nLMVS-Real
  * [Preprocessed data (48GB)](https://drive.google.com/uc?id=19uYwiQLaSKAA6XGnnmsHZ9if4JOAFmIk)
  * Raw data
    * [court_duralumin & court_blue-metallic (159GB)](https://drive.google.com/uc?id=119IKcnoxvppk6wngybhH8KjCSNX_xjQg)
    * [court_white-primer & entrance_bright-red (152GB)](https://drive.google.com/uc?id=1raqMa3Vtz_aogGjWzVAHW0MG-Av786Qm)
    * [buildings_bright-red & buildings_duralumin (154GB)](https://drive.google.com/uc?id=1oWrilKNdMuZB7zM5b4WhcbdpXD-xEzXj)
    * [laboratory_bright-red (81GB)](https://drive.google.com/uc?id=1Gb-qwUQUVBEsq-JK-FpAXLfO4_H9Jy4c)
    * [court_bright-red (70GB)](https://drive.google.com/uc?id=1pAVDtJCep86ZEzgv-ry6RDOt-XXHxFn9)
    * [buildings_white-primer & buildings_blue-metallic (158GB)](https://drive.google.com/uc?id=19b9BfZnWNCPl10gb1B4PHecY2d_xGy_p)
    * [laboratory_duralumin (85GB)](https://drive.google.com/uc?id=1oWgdmUuqRAsfMo4zOVbU6ZlSTo76YBfc)
    * [laboratory_blue-metallic (82GB)](https://drive.google.com/uc?id=1ARqUqaWYVol_NssWyMfRHfnNdQIpMoiE)
    * [entrance_duralumin & entrance_blue-metallic (152GB)](https://drive.google.com/uc?id=1Zsf0mrbFHqVSgrrtN6mPNY-4eyXH2Wsp)
    * [entrance_white-primer (78GB)](https://drive.google.com/uc?id=15AeANXlhQyGuKnBYFCp8IzRl6est5gJr)
    * [laboratory_white-primer (84GB)](https://drive.google.com/uc?id=1W25iq5JO_XKRj6HDyKn4w1GVUUfyObEj)
    * [manor_bright-red & manor_duralumin (175GB)](https://drive.google.com/uc?id=1BW5k8393XTa0o9rb9777Nc8OX0tHy6Ip)
    * [manor_blue-metallic & manor_white-primer (185GB)](https://drive.google.com/uc?id=1AaJioKdFZ9egEa5edu0h0ylXPsIUK471)
    * [chapel_bright-red & chapel_duralumin (158GB)](https://drive.google.com/uc?id=1EZAP9RnMKvtt9Qu852X6m-aHWn9tcBtS)
    * [chapel_white-primer (89GB)](https://drive.google.com/uc?id=16euRh8XzOlCJIhE0rZdFS_4qfarFnUu6)
    * [chapel_blue-metallic (85GB)](https://drive.google.com/uc?id=11WzKcxcblIaDj-KUSVHBxq3yb8LOgnAe)


Our dataset is organized as follows.

### nLMVS-Synth (Training Set)
The training set consists of .pt files (e.g., ```./00000000.pt```) which we can load using torch.load() of PyTorch library. Each file contains:
* Training data for shape-from-shading network
  * 'img': A HDR image of a object
  * 'rmap': A reflectance map (an image of a sphere whose material is the same as the object)
  * 'mask': An object segmentation mask
  * 'normal': A ground truth normal map
* Training data for cost volume filtering network
  * 'imgs': Three view images of a object
  * 'rmap': Three view reflectance maps
  * 'intrinsics': Intrinsic matrices of the views
  * 'proj_matrices': Projection matrices of the views
  * 'rot_matrices': Rotation matrices of the views
  * 'depth_values': Discretized depth values which are used to construct a cost volume
  * 'depths': Ground truth depth maps
  * 'normals': Ground truth normal maps

### nLMVS-Synth (Test Set)
Please see [nLMVS-Synth-Eval.md](./nLMVS-Synth-Eval.md).

### nLMVS-Real (Preprocessed Data)
Please see [nLMVS-Real.md](./nLMVS-Real.md).

### nLMVS-Real (Raw Data)
* Raw images can be found at ```./data/${illum_name}_${mat_name}/${shape_name}/raw```.
* Raw panorama images can be found at ```./data/${illum_name}_${mat_name}/${shape_name}/theta_raw```.

Although we do not provide detailed documentation, there are also python scripts and intermediate data (e.g., uncropped HDR images) for preprocessing the raw data. ```./README.md``` briefly describes the usage of the python scripts.

## Demo
### Depth, Normal, and Reflectance Estimation from 5 view images
You can recover depths, surface normals, and reflectance from 5 view images in the nLMVS-Synth dataset by runninng ```run_est_shape_mat_per_view_nlmvss.py```.
```
Usage: python run_est_shape_mat_per_view_nlmvss.py ${OBJECT_NAME} ${VIEW_INDEX} --dataset-path ${PATH_TO_DATASET}
Example: python run_est_shape_mat_per_view_nlmvss.py 00152 5 --dataset-path /data/nLMVS-Synth-Eval/nlmvs-synth-eval
```

You can recover depths, surface normals, and reflectance from 5 view images in the nLMVS-Real Dataset by runninng ```run_est_shape_mat_per_view_nlmvsr.py```.
```
Usage: python run_est_shape_mat_per_view_nlmvsr.py ${ILLUMINATION_NAME}_${PAINT_NAME} ${SHAPE_NAME} ${VIEW_INDEX} --dataset-path ${PATH_TO_DATASET}
Example: python run_est_shape_mat_per_view_nlmvsr.py laboratory_blue-metallic horse 0 --dataset-path /data/nLMVS-Real/nlmvs-real
```

Estimation results are saved to ```./run/est_shape_mat_per_view```.

### Whole 3D Shape Recovery 
You can recover whole object 3D shape and reflectance from 10 (or 20) view images in the nLMVS-Synth dataset by running ```run_est_shape_mat_nlmvss.py```.
```
Usage: python run_est_shape_mat_nlmvss.py ${OBJECT_NAME} --dataset-path ${PATH_TO_DATASET} --exp-name ${EXPERIMENT_NAME}
Example: python run_est_shape_mat_nlmvss.py 00152 --dataset-path /data/nLMVS-Synth-Eval/nlmvs-synth-eval-10 --exp-name nlmvss10
```

For reconstruction from the nLMVS-Real dataset, you can use ```run_est_shape_mat_nlmvsr.py```.
```
Usage: python run_est_shape_mat_nlmvsr.py ${ILLUMINATION_NAME}_${PAINT_NAME} ${SHAPE_NAME} --dataset-path ${PATH_TO_DATASET}
Example: python run_est_shape_mat_nlmvsr.py laboratory_bright-red bunny --dataset-path /data/nLMVS-Real/nlmvs-real
```

Estimation results are saved to ```./run/est_shape_mat```.

### Mesh Reconstruction
You can recover 3D mesh models from the estimation results by using the following scripts.

```
python run_recover_mesh_per_view_nlmvss.py ${OBJECT_NAME} ${VIEW_INDEX}
```

```
python run_recover_mesh_per_view_nlmvsr.py ${ILLUMINATION_NAME}_${PAINT_NAME} ${SHAPE_NAME} ${VIEW_INDEX} --dataset-path ${PATH_TO_DATASET}
```

```
python run_recover_mesh_nlmvss.py ${OBJECT_NAME}
```

```
python run_recover_mesh_nlmvsr.py ${ILLUMINATION_NAME}_${PAINT_NAME} ${SHAPE_NAME} --dataset-path ${PATH_TO_DATASET}
```

### Training from scratch
You can train our shape-from-shading network with the nLMVS-Synth dataset by
```
python train_sfs.py --dataset-dir ${PATH_TO_DATASET}
```

You can train our cost volume filtering network with the nLMVS-Synth dataset by
```
python train_nlmvs.py --dataset-dir ${PATH_TO_DATASET}
```


## Acknowledgement
This work was in part supported by JSPS 20H05951, 21H04893, JST JPMJCR20G7, JPMJSP2110, and RIKEN GRP. We also thank Shinsaku Hiura for his help in 3D printing.

## Use of existing assets
We used the following existing 3D mesh models, BRDF data, and environment maps to create the nLMVS-Synth and nLMVS-Real datasets.

#### nLMVS-Synth (Training Set)
- 3D mesh models of [Xu et al.](https://cseweb.ucsd.edu/~viscomp/projects/SIG18Relighting/)
- BRDF data from [MERL BRDF Database](https://www.merl.com/brdf/)
- HDR Environment maps from [Laval Indoor HDR Dataset](http://vision.gel.ulaval.ca/~jflalonde/publications/projects/deepIndoorLight/index.html) and [Poly Haven](https://polyhaven.com/)

#### nLMVS-Synth (Test Set)
- 3D mesh models
  - Armadillo model from [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/)
  - Bunny model from [McGuire Computer Graphics Archive](https://casual-effects.com/data/). The original model is created by Stanford Computer Graphics Laboratory
  - Max-Planck Bust model from [Suggestive Contour Gallery](https://gfx.cs.princeton.edu/proj/sugcon/models/). The original model is digitized by Christian Rössl (MPI Informatik)
  - Golfball model from [Suggestive Contour Gallery](https://gfx.cs.princeton.edu/proj/sugcon/models/)
  - Teapot model from [PLY Files - an ASCII Polygon Format ](https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html)
- BRDF data from [MERL BRDF Database](https://www.merl.com/brdf/)
- Light probes from [High-Resolution Light Probe Image Gallery](https://vgl.ict.usc.edu/Data/HighResProbes/) (Ennis, Grace, Pisa, and Uffizi) and [Light Probe Image Gallery](https://www.pauldebevec.com/Probes/) (Forest (Eucalyptus Grove) and St. Peter's)

#### nLMVS-Real
- Bunny model from [McGuire Computer Graphics Archive](https://casual-effects.com/data/). The original model is created by Stanford Computer Graphics Laboratory.
- Max-Planck Bust model from [Suggestive Contour Gallery](https://gfx.cs.princeton.edu/proj/sugcon/models/). The original model is digitized by Christian Rössl (MPI Informatik).
- Horse and Shell models from [Multiview Objects Under Natural Illumination Database](https://vision.ist.i.kyoto-u.ac.jp/codeanddata/multinatgeom/).
