# Scene Coordinate Reconstruction Priors

This repository contains code associated to the paper:
> **Scene Coordinate Reconstruction Priors**
>
> [Wenjing Bian](https://scholar.google.com/citations?user=IVfbqkgAAAAJ&hl=en), 
> [Axel Barroso-Laguna](https://scholar.google.com/citations?user=m_SPRGUAAAAJ&hl=en/), 
> [Tommaso Cavallari](https://scholar.google.it/citations?user=r7osSm0AAAAJ&hl=en),
> [Victor Adrian Prisacariu](https://www.robots.ox.ac.uk/~victor/), and
> [Eric Brachmann](https://ebrach.github.io/)
> 
> ICCV 2025

For further information please visit:
* [Project page](https://nianticspatial.github.io/scr-priors/)
* [arXiv](https://arxiv.org/abs/2510.12387)

**Important**: The main contributions of the paper, namely the reconstruction priors as well as the RGB-D version of ACE0, have been integrated into the 
[ACE0 codebase](https://github.com/nianticlabs/acezero) as additional features (deactivated by default).

This repository contains additional resources associated with the ICCV 2025 paper: 
* Scripts to replicate the main results of the experiments
* Code to train a 3D point cloud diffusion prior
* Code to fit a Laplace depth prior
* Environment file to work with the diffusion prior

## Setup

Checkout the [ACE0 code](https://github.com/nianticlabs/acezero) and follow its 
[installation instructions](https://github.com/nianticlabs/acezero), i.e. create the `ace0` environment and install the 
DSAC* C++/Python bindings.

For running the PSNR benchmark, additionally install Nerfstudio like explained [here](https://github.com/nianticlabs/acezero/blob/main/benchmarks/README.md#requirements).

Our experiments scripts assume that the ACE0 code resides in a folder `acezero` next to the folder with this code, but this can be adapted in the shell scrips.
These scripts further assume that Nerfstudio lives in a Conda environment called `nerfstudio`.

The default `ace0` environment should work with all priors except the diffusion prior, which has additional dependencies.
If you want to use the diffusion prior, or train your own diffusion model, you can use the environment file provided in 
this repo:

```shell
conda env create -f environment.yml
conda activate ace0_priors
```

## Replicate Experiments

We provide shell scripts to replicate the main results of the paper, corresponding to Table 1, 2 and 3.
The scripts are named with the pattern `<task>_<dataset>_<algorithm>.sh` where task is either reconstruction (ACE0) or 
relocalization (ACE) and the algorithm names specifies whether a prior is used and which one. 
Each script requires some specific data preparation to be done first, as explained in the subsections below. 

For the diffusion prior, we provide a pre-trained model [here](https://storage.googleapis.com/niantic-lon-static/research/scr-priors/diffusion_prior.pt).
Download it to `models/diffusion_prior.pt` where all the following experiments scripts expect it to be.

These scripts set all necessary parameters (if they differ from ACE0's defaults) and print evaluation results to 
the console after execution. 
All scripts can be configured to render a visualisation video for each scene (off by default).

### ScanNet

[Download the ScanNet dataset](https://github.com/ScanNet/ScanNet?tab=readme-ov-file#scannet-data) and point our shell
scripts to the `scans_test` folder via the `datasets_folder` variable. 
Our shell scripts point to `datasets/scannet/scans_test` by default.

Before running the scripts, create the custom 60/60 split for the PSNR benchmark as used in Table 1 of the paper:
```shell
python scripts/create_chunked_splits.py /path/to/scannet/scans_test splits
```
This will generate and store the necessary split files in a new folder `splits`.
You should now be able to run all ScanNet scripts. 

Note: In a previous version of this code, computation of pose evaluation results (ATE/RPE and median errors in Table 1)
were slightly off due to some ScanNet scenes having invalid ground truth poses. We corrected the corresponding 
evaluation script and report updated numbers here:

| Method                     | ATE/RPE(cm)      | Med.Err.(cm/°)   |
|----------------------------|------------------|------------------|
| ACE0                       | 24.1/4.1         | 17.6/8.3         |
| ACE0 + Laplace NLL         | 23.9/3.8         | 16.6/7.5         |
| ACE0 + Laplace WD          | 25.9/3.7         | 18.2/7.9         |
| ACE0 + Diffusion           | 23.7/**3.5**     | 17.6/6.9         |
| (RGB-D) ACE0 + DSAC* Loss  | 26.7/**3.5**     | 19.9/6.2         |
| (RGB-D) ACE0 + Laplace NLL | **17.4**/**3.5** | **12.3**/**4.1** |

Note: ScanNet images have a small black border. This does not affect ACE0, but it can lower numbers in the PSNR benchmark.
Therefore, we call the benchmark with the `--crop 15` option which removes 15px from each image side.

### Indoor-6

The [Indoor-6 dataset](https://github.com/microsoft/SceneLandmarkLocalization) needs to be converted to an ACE-compatible format. You can do this by running `setup_indoor6.py` in the `datasets` folder. 
By default, the script will download Indoor-6 to a new folder `indoor6_raw`, and create a ACE-compatible version in `indoor6` 
where images have been sorted into training (=mapping), validation and test (=query) subsets.

Our shell scripts point to `datasets/indoor6` by default. Nothing more is required to run the Indoor-6 shell scripts.

### 7Scenes

The 7Scenes dataset needs to be converted to an ACE-compatible format. You can do this by running ACE0's `setup_7scenes.py` from this repo's `datasets` folder:

```shell
python path/to/acezero/datasets/setup_7scenes.py --setup_ace_structure --depth calibrated --poses pgt
```

This will download the 7Scenes dataset to a new folder `7scenes` and create a ACE-compatible version in `7scenes_ace` where images have been sorted into training (=mapping) and test (=query) subsets.
The script also aligns Kinect depth maps with the RGB sensor and downloads COLMAP pseudo ground truth poses from [visloc_pseudo_gt_limitations](https://github.com/tsattler/visloc_pseudo_gt_limitations) (ICCV21). 

Our shell scripts point to `datasets/7scenes_ace` by default. Nothing more is required to run the 7Scenes shell scripts.

## Fit the Depth Prior

Fitting the depth prior is a simple process that fits a Laplace distribution to a histogram of depth values.
The output are two parameters: The Laplace location (~mean) and the bandwidth (~variance).

We provide a simple script that samples a set of depth images to fit the prior. Call it on the ScanNet training scenes to get the values that we report in the paper (up to some randomness):

```shell
python train_depth_prior.py "./datasets/scannet/scans/*/sensor_data/*.pgm"
```

Via the `--plot` flag, the script will also store the depth histogram and prior plot to `depth_laplace_fit.png`.

## Train a Diffusion Prior

We provide a script, `scripts/train_diffusion_prior.sh` to re-train the 3D diffusion prior on ScanNet training scenes, as we did in the paper.
This needs the `ace0_priors` environment and access to the ScanNet `scans` folder.

Training is a two-stage process. Firstly, the script extracts sub-sampled point clouds for each scene to speed up data-streaming
during training. Secondly, the training itself takes place.

By default, training will run for 100k iterations on a single GPU (~half a day on a V100). 
Results are stored in `out_diffusion_model_rotate/pvcnn_allscene_T_20_lr`, including network checkpoints. 
These check point files can be directly passed to ACE/ACE0 via `--prior_diffusion_model_path`.

We provide a pre-trained prior [here](https://storage.googleapis.com/niantic-lon-static/research/scr-priors/diffusion_prior.pt).

## Acknowledgements

Our diffusion prior builds on code from: 
  * [diffusionerf](https://github.com/nianticlabs/diffusionerf) (MIT license)
  * [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main) (MIT license)
  * [diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud/tree/main) (MIT license)
  * [projection-conditioned-point-cloud-diffusion](https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion/tree/main) (MIT license)
  * [pvcnn](https://github.com/mit-han-lab/pvcnn) (MIT license)

Furthermore:

* Experiments on 7Scenes use [COLMAP](https://github.com/colmap/colmap) poses from [here](https://github.com/tsattler/visloc_pseudo_gt_limitations).
* ACE0 initialises reconstructions using [ZoeDepth](https://github.com/isl-org/ZoeDepth) estimates

Please consider citing these projects if appropriate.

## Citation

Please consider citing our work:

```
@inproceedings{bian2024scrpriors,
    title={Scene Coordinate Reconstruction Priors},
    author={Bian, Wenjing and Barroso-Laguna, Axel and Cavallari, Tommaso and Prisacariu, Victor Adrian and Brachmann, Eric},
    booktitle={ICCV},
    year={2025},
}
```

This code builds directly on ACE, ACE0 and DSAC*. Please consider citing:

```
@inproceedings{brachmann2024acezero,
    title={Scene Coordinate Reconstruction: Posing of Image Collections via Incremental Learning of a Relocalizer},
    author={Brachmann, Eric and Wynn, Jamie and Chen, Shuai and Cavallari, Tommaso and Monszpart, {\'{A}}ron and Turmukhambetov, Daniyar and Prisacariu, Victor Adrian},
    booktitle={ECCV},
    year={2024},
}

@inproceedings{brachmann2023ace,
    title={Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses},
    author={Brachmann, Eric and Cavallari, Tommaso and Prisacariu, Victor Adrian},
    booktitle={CVPR},
    year={2023},
}

@article{brachmann2021dsacstar,
  title={Visual Camera Re-Localization from {RGB} and {RGB-D} Images Using {DSAC}},
  author={Brachmann, Eric and Rother, Carsten},
  journal={TPAMI},
  year={2021}
}
```

## License

Copyright © Niantic Spatial, Inc. 2025. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.
