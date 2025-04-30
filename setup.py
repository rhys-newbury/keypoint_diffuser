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
)
