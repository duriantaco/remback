from setuptools import setup, find_packages

setup(
    name="remback",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "opencv-python>=4.5.0",
        "mtcnn>=0.1.0",
        "huggingface_hub>=0.10.0",
        "tqdm>=4.62.0",
        "torchvision>=0.15.1",
        "tensorflow>=2.19.0",
        "numpy>=1.26.4"
    ],
    author="oha",
    author_email="aaronoh2015@gmail.com",
    description="A library to remove backgrounds from profile pictures using a fine-tuned SAM model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/duriantaco/remback",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10,<3.12",
    entry_points={
        "console_scripts": [
            "remback=remback.cli:main",
        ],
    },
)