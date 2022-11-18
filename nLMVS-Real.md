# [nLMVS-Real Dataset (Preprocessed)](https://github.com/kyotovision-public/nLMVS-Net)

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
├── nlmvs-real
│   └── ${illum name}_${mat name}
│       ├── illumination.exr          - Environment map (captured using mirrored ball, not to be used in our experiments)
│       └── ${shape name}
│         ├── illumination.exr        - Environment map (captured using RICOH THETA Z1)
│         ├── mesh_to_world.npy       - Transformation matrix from coordinate system of mesh files in ./mesh_files to world coordinate system
│         ├── mesh_aligned.ply        - Ground truth 3D mesh file aligned with captured images (please see below to prepare this file)
│         ├── view-??.exr             - Linear HDR image
│         ├── view-??.jpg             - Tone-mapped SDR image (not to be used in our experiments)
│         ├── view-??_m.png           - Object segmentation mask
│         ├── view-??_d.npy           - Ground truth depth map
│         ├── view-??_n.npy           - Ground truth normal map
│         └── views.txt               - Text file containing intrinsic and extrinsic camera parameters
├── mesh_files
│   └── ${shape name}_processed.ply - Ground truth 3D mesh file (not aligned with captured images)
├── create_aligned_meshes.py        - Script to create ground truth 3D mesh models aligned with captured images.
├── preprocess_bunny.py             - Script to pre-process the Stanford Bunny model
├── preprocess_planck.py            - Script to pre-process the Max-Planck Bust model
├── script_for_bunny.mlx            - Meshlab script for mesh preprocessing
├── script_for_planck_1.mlx         - Meshlab script for mesh preprocessing
├── script_for_planck_2.mlx         - Meshlab script for mesh preprocessing
└── README.md                       - Describing license, data organization, and acknowledgement
```


## Creating ground truth 3D mesh models
1. Download bunny.obj and maxplanck.ply from [McGuire Computer Graphics Archive](https://casual-effects.com/data/) and [Suggestive Contour Gallery](https://gfx.cs.princeton.edu/proj/sugcon/models/), respectively. Save them to ```./mesh_files```.
2. Run ```python preprocess_bunny.py``` and ```python preprocess_planck.ply``` to preprocess the mesh files.
3. Run ```python create_aligned_meshes.py```.

## Acknowledgement
This work was in part supported by JSPS 20H05951, 21H04893, JST JPMJCR20G7, JPMJSP2110, and RIKEN GRP. We also thank Shinsaku Hiura for his help in 3D printing.

We used the following 3D mesh models to create this dataset. 
- Stanford Bunny model from [McGuire Computer Graphics Archive](https://casual-effects.com/data/). The original model is created by Stanford Computer Graphics Laboratory.
- Max-Planck Bust model from [Suggestive Contour Gallery](https://gfx.cs.princeton.edu/proj/sugcon/models/). The original model is digitized by Christian Rössl (MPI Informatik).
- Horse and Shell models from [Multiview Objects Under Natural Illumination Database](https://vision.ist.i.kyoto-u.ac.jp/codeanddata/multinatgeom/).
