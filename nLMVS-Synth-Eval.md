# [nLMVS-Synth Dataset (Test Set)](https://github.com/kyotovision-public/nLMVS-Net)

This dataset is provided under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/). If you use this dataset please cite our paper.

```
@InProceedings{Yamashita_2023_WACV,
    author    = {Kohei Yamashita and Yuto Enyo and Shohei Nobuhara and Ko Nishino},
    title     = {nLMVS-Net: Deep Non-Lambertian Multi-View Stereo},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {Jan},
    year      = {2023}
}
```

This dataset is organized as follows.

```
.
├── nlmvs-synth-eval
│   └── ?????
│       ├── illumination.hdr     - HDR environment map (please see below to prepare this file)
│       ├── illumination.hdr     - Tone-mapped SDR environment map (not to be used in experiments, please see below to prepare this file)
│       ├── pair.txt             - list of neighboring views for each view
│       ├── views.txt            - Text file describing intrinsic and extrinsic camera parameters for all views
│       ├── cams
│       │   └── ????????_cam.txt - Text file describing intrinsic and extrinsic camera parameters
│       ├── Depths
│       │   └── ????????.pfm     - Depth Map
│       ├── iamges
│       │   ├── ????????.exr     - Rendered HDR image
│       │   └── ????????.jpg     - Tone-mapped SDR image  (not to be used in experiments)
│       ├── Normals
│       │   ├── ????????.exr     - Normal Map
│       │   └── ????????.jpg     - Normal Map (8bit, not to be used in experiments)
│       └── reflectance_maps
│           ├── ????????.exr     - HDR reflectance map
│           └── ????????.jpg     - Tone-mapped SDR reflectance map
├── nlmvs-synth-eval-10          - The organization is the same as nlmvs-synth-eval but the number of views is 10
├── light_probe_to_panorama.py   - Script to covert a light probe into a panorama image
├── preprocess_illum_maps.py     - Script to create down-sized environment maps for each object
└── README.md                    - Describing license, data organization, and acknowledgement
```

## Downloading and preprocessing ground truth environment maps
1. Download light probes (.hdr files) from [High-Resolution Light Probe Image Gallery](https://vgl.ict.usc.edu/Data/HighResProbes/) and [Light Probe Image Gallery](https://www.pauldebevec.com/Probes/).
2. Convert the light probes from [Light Probe Image Gallery](https://www.pauldebevec.com/Probes/) into those in a latitude-longitude panoramic format using ```light_probe_to_panorama.py'''.
3. Run ```preprocess_illum_maps.py''' to create down-sized environment maps for each object.

## Camera parameters
The File Formats of pair.txt and ????????_cam.txt are the same as those of [MVSNet](https://github.com/YoYo000/MVSNet#file-formats). The file format of views.txt is similar to the nLMVS-Real dataset (preprocessed), but there is no cropping information (i.e., top left coordinate of the cropped image) as cropping is not applied to this dataset.

## Acknowledgement
This work was in part supported by JSPS 20H05951, 21H04893, JST JPMJCR20G7, JPMJSP2110, and RIKEN GRP.

## Use of existing assets
We used the following 3D mesh models, BRDF data, and Light Probes to render images in this dataset. 
- 3D mesh models
  - Armadillo model from [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/)
  - Bunny model from [McGuire Computer Graphics Archive](https://casual-effects.com/data/). The original model is created by Stanford Computer Graphics Laboratory
  - Max-Planck Bust model from [Suggestive Contour Gallery](https://gfx.cs.princeton.edu/proj/sugcon/models/). The original model is digitized by Christian Rössl (MPI Informatik)
  - Golfball model from [Suggestive Contour Gallery](https://gfx.cs.princeton.edu/proj/sugcon/models/)
  - Teapot model from [PLY Files - an ASCII Polygon Format ](https://people.sc.fsu.edu/~jburkardt/data/ply/ply.html)
- BRDF data from [MERL BRDF Database](https://www.merl.com/brdf/)
- Light probes from [High-Resolution Light Probe Image Gallery](https://vgl.ict.usc.edu/Data/HighResProbes/) (Ennis, Grace, Pisa, and Uffizi) and [Light Probe Image Gallery](https://www.pauldebevec.com/Probes/) (Forest (Eucalyptus Grove) and St. Peter's)