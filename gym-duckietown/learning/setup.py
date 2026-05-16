from setuptools import find_packages, setup

# Install PyTorch separately (CUDA wheels are not on PyPI), e.g.:
#   pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
#     -f https://download.pytorch.org/whl/cu113/torch_stable.html
#
# Install the gym env from the repo root first:
#   pip install -e ..

setup(
    name="duckietown-gym-learning",
    version="0.1.0",
    description="RL and imitation learning examples for gym-duckietown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.10.0,<=1.20.0",
        "gym>=0.17.1",
    ],
    python_requires=">=3.6",
)
