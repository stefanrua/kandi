# kandi

This repository contains the code relevant to my bachelor's thesis.

## Contents

 - `dataio.py`: functions to help with I/O
 - `figures.py`: figures seen in the Methods section
 - `preprocessing.py`: preprocessing
 - `processing.py`: clustering and regression
 - `visualization.py`: figures seen in the Results section

## Running

To reproduce the results shown in the thesis:

 1. Create directories for data and figures:
    ```
    mkdir data/ figures/
    ```
 2. Acquire the original data.
    Note: the starting data I used is stored as python pickles.
    Please don't actually download and open them,
    [pickle files can execute arbitrary
    code](https://docs.python.org/3/library/pickle.html).
    I should convert them to csv for sharing.
 3. Preprocess, process, and visualize:
    ```
    python3 preprocessing.py && python3 processing.py && python3 visualization.py
    ```

Be warned, the clusterings can take a long time (~2.5 hours on
an i5-4690k).
