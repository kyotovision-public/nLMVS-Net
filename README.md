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

We tested our code with Python 3.? on Ubuntu 20.04 LTS. Our code depends on the following modules.

* numpy
* opencv-python
* ...

You can use `nlmvsnet.def` to build your singularity container by
```
$ singularity build --fakeroot nlmvsnet.sif nlmvsnet.def
```

Please prepare the following files.
* Download ```module.py``` of [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch/tree/e0f2ae3d7cb2dd13807b775f2075682eaa7f1521) and save it to ```./core```.
* Download ```alum-bronze.pt``` from [MERL BRDF Database](https://www.merl.com/brdf/) and save it to ```./data```.
* Download ```ibrdf.pt``` from [here]() and save it to ```./data```.
* Download ```merl_appearance_ratio.pt``` and ```merl_mask.pt``` from [here]() and save them to ```./core/ibrdf/render```.
* Download pretrained weight files from [here]() and save them to ```./weights/sfsnet``` and ```./weights/nlmvsnet```.

## nLMVS-Synth and nLMVS-Real datasets

### License

The nLMVS-Synth and nLMVS-Real datasets are provided under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

### Download

We provide the raw and also preprocessed data.  As the raw data is so large, we recommend you to use the preprocessed data which contains HDR images.

* nLMVS-Synth
  * Training Set
    * [Training Data for Shape-from-Shading Network (7.1GB)](nLMVS-Synth-Train-SfS.zip)
    * [Training Data for Cost Volume Filtering Network (23.4GB)](nLMVS-Synth-Train-CV.zip)
  * [Test set (??.?GB)](nLMVS-Synth-Eval.zip)
* nLMVS-Real
  * [Preprocessed data (??.?GB)](nLMVS-Real.zip)
  * Raw data
    * [court_duralumin & court_blue-metallic (??.?GB)](20211104.zip)
    * [court_white-primer & entrance_bright-red (??.?GB)](20211106.zip)
    * [buildings_bright-red & buildings_duralumin (??.?GB)](20211107.zip)
    * [laboratory_bright-red (??.?GB)](20211108.zip)
    * [court_bright-red (??.?GB)](20211202.zip)
    * [buildings_white-primer & buildings_blue-metallic (??.?GB)](20211205.zip)
    * [laboratory_duralumin (??.?GB)](20211207.zip)
    * [laboratory_blue-metallic (??.?GB)](20211208.zip)
    * [entrance_duralumin & entrance_blue-metallic (??.?GB)](20211211.zip)
    * [entrance_white-primer (??.?GB)](20211212.zip)
    * [laboratory_white-primer (??.?GB)](20211213.zip)
    * [manor_bright-red & manor_duralumin (??.?GB)](20211220.zip)
    * [manor_blue-metallic & manor_white-primer (??.?GB)](20211222.zip)
    * [chapel_bright-red & chapel_duralumin (??.?GB)](20220111.zip)
    * [chapel_white-primer (??.?GB)](20220117.zip)
    * [chapel_blue-metallic (??.?GB)](20220131.zip)


Our dataset is organized as follows.

### nLMVS-Synth (Training Set)
The training set consists of .pt files (e.g., ```./00000000.pt```) which can be loaded using torch.load() of PyTorch library. Each file contains the following data:
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

Although we do not provide any detailed documentation, there are also python scripts to preprocess the raw images and intermediate data created by the scripts (e.g., uncropped HDR images). ```./README.md``` briefly describes the usage of the python scripts.

## Usage
### Demo

```
python run_est_shape_mat_per_view_nlmvss.py ${OBJECT_NAME} ${VIEW_INDEX} --dataset-path ${PATH_TO_DATASET}
```

### Training with the nLMVS-Synth dataset
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
