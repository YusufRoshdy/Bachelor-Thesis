# Bachelor Thesis

This repo holds the code written for my bachelor thesis at Innopolis University.
It revolves around integrating an implementation of the [Annoy](https://github.com/spotify/annoy) algorithm for finding approximate nearest neighbors.

## Requirements

- Python 3.7 or newer
- Any compiler for C++11 or newer that supports [OpenMP](https://www.openmp.org/)

## Getting started

Install the requirements:
```sh
pip install -r requirements.txt
```

Build the Cython code:
```sh
cd CppAnnoy
python setup.py build_ext --inplace
cd ..
```

Run the comparison script:
```sh
python compare.py
```
Or open the [interactive notebook](compare.ipynb) in Jupyter Lab
```sh
jupyter lab
```
