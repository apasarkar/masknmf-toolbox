from setuptools import setup, find_packages

setup(
    name='masknmf-toolbox',
    version='0.1.0',
    description='masknmf pipeline for motion correction, denoising, compression, and demixing of functional neuroimaging data',
    author='Amol Pasarkar',
    url='https://github.com/apasarkar/masknmf-toolbox',
    packages=find_packages(),
    install_requires=[
        'torch',  # PyTorch
        'fastplotlib',
        'future',
        'numpy',
        'scipy',
        'plotly',
        'line-profiler',
        'h5py',
        'matplotlib',
        'networkx',
        'scikit-image',
        'oasis-deconv',
        'tqdm',
        'hydra-core'
    ],

    extras_require={
        "notebook": [
            "jupyterlab",
        ],
    },

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
