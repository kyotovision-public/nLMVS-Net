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

## nLMVS-Synth and nLMVS-Real datasets

### License

The nLMVS-Synth and nLMVS-Real datasets are provided under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

### Download

We provide the raw and also preprocessed data.  As the raw data is so large, we recommend you to use the preprocessed data which contains HDR images.

* nLMVS-Synth
* nLMVS-Real
  * raw
  * preprocessed


Our dataset is organized as follows.
