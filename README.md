# Unsupervised 3D Keypoint Learning via Latent Diffusion Models for Shape Reconstruction

Rhys Newbury, Juyan Zhang, Tin Tran, Hanna Kurniawati, Dana Kulić

We present an unsupervised framework for learning semantically meaningful 3D keypoints from point cloud data using a latent diffusion model. Our method encodes input shapes into a structured latent space consisting of a set of 3D keypoints. These keypoints serve as a compact and interpretable representation that conditions an Elucidated Diffusion Model (EDM) to reconstruct the full shape. To ensure the extracted keypoints are both spatially meaningful and consistent across object instances, we introduce several geometric supervision strategies: a Chamfer loss to anchor keypoints near the input shape, and a deformation consistency loss to encourage robustness under geometric transformations.


A lot of this code is built upon the following repos:

- [KeyPointDeformer](https://github.com/tomasjakab/keypoint_deformer)
- [DPM](https://github.com/luost26/diffusion-point-cloud)
- [Point Transformer v3](https://github.com/Pointcept/PointTransformerV3)
- [EMDLoss](https://github.com/ZirongLiu/EMDLoss-for-large-scale-point-clouds)
- [EDM](https://github.com/NVlabs/edm)

## Training
Download ShapeNet from HuggingFace

To train a model on the airplane category with 8 unsupervised keypoints run:
```
python scripts/train_ae.py -c configs/airplane-8kpt.yaml
```

## Testing
To test the trained model run:
```
python scripts/train_ae.py -c configs/airplane-8kpt.yaml -t configs/test.yaml
```

## Contributing

### Git hooks

The CI will run several checks on the new code pushed to the repository. These checks can also be run locally without waiting for the CI by following the steps below:

1. [install `pre-commit`](https://pre-commit.com/#install),
2. Install the Git hooks by running `pre-commit install`.

Once those two steps are done, the Git hooks will be run automatically at every new commit.
The Git hooks can also be run manually with `pre-commit run --all-files`, and if needed they can be skipped (not recommended) with `git commit --no-verify`.

**Note:** you may have to run `pre-commit run --all-files` manually a couple of times to make it pass when you commit, as each formatting tool will first format the code and fail the first time but should pass the second time.

## Eval

<!-- Need NFS for database. -->
python3 plot_all.py --db "/run/user/1000/gvfs/smb-share:server=130.194.128.238,share=bryce-rhys/results.db"

<!--  -->