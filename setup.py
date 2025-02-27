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
        'plotly',
        'line-profiler',
        'matplotlib',
        'jnormcorre @ git+https://github.com/apasarkar/jnormcorre.git@main',
        'localmd @ git+https://github.com/apasarkar/localmd.git@main',
        'rlocalnmf @ git+https://github.com/apasarkar/rlocalnmf.git@main'
    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
