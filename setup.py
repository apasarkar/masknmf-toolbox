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
    dependency_links=[
        'https://github.com/apasarkar/jnormcorre/tarball/main#egg=jnormcorre',
        'https://github.com/apasarkar/localmd/tarball/main#egg=localmd',
        'https://github.com/apasarkar/rlocalnmf/tarball/main#egg=rlocalnmf',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
