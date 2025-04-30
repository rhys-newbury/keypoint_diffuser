from setuptools import find_packages, setup


setup(
    name="keypoint_diffuser",
    version="0.1",
    packages=find_packages(
        where="src"
    ),  # Look for packages in the keypoint_diffuser directory
    package_dir={
        "": "src"
    },  # Map the package directory to the keypoint_diffuser folder
    install_requires=[
        "PyYAML==6.0",
        "torch==2.0.0",
        "torchvision==0.15.0",
        "einops==0.8.1",
        "tqdm==4.64.1",
        "scipy==1.11.4",
        "wandb==0.19.9",
        "addict==2.4.0",
        "tensorboardX==2.6.2.2",
        "open3d==0.18.0",
        "timm==1.0.15",
        "ConfigArgParse==1.4",
        "pytorch3d==0.7.8",
        "torch-scatter==2.1.2",
        "spconv-cu116==2.3.6",
    ],
)
