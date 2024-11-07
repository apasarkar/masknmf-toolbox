from setuptools import setup, find_packages

setup(
    name='masknmf-toolbox',
    version='0.1.0',
    description='A toolbox for neuroimaging signal extraction and analysis',
    author='Amol Pasarkar',
    url='https://github.com/apasarkar/masknmf-toolbox',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxlib',
        'torch',  # PyTorch
        'fastplotlib',
        'hydra-core',
        'future',
        'numpy',
        'scipy',
        'matplotlib',
    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
