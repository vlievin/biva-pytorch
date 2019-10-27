import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='biva-pytorch',
    version='0.1.3',
    author="Valentin Lievin",
    author_email="valentin.lievin@gmail.com",
    description="Official PyTorch BIVA implementation (BIVA: A Very Deep Hierarchy of Latent Variables for Generative Modeling)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlievin/biva-pytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
        'tqdm',
        'numpy',
        'matplotlib'
    ],
)
