# Repo 

This repository contains code corresponding to:

J. Lundell, F. Verdoja and V. Kyrki, "Hallucinating Robots: Inferring Obstacle Distances from Partial Laser Measurements," 2018 _IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_, Madrid, 2018, pp. 4781-4787.

**doi:** [10.1109/IROS.2018.8594399](https://doi.org/10.1109/IROS.2018.8594399)

**preprint:** [arxiv](https://arxiv.org/abs/1805.12338)


Please cite as:
```
@inproceedings{lundell2018hallucinating,
    author={J. {Lundell} and F. {Verdoja} and V. {Kyrki}},
    booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    title={Hallucinating Robots: Inferring Obstacle Distances from Partial Laser Measurements},
    year={2018},
    pages={4781-4787},
    doi={10.1109/IROS.2018.8594399},
    ISSN={2153-0866}
}
```

## Dependencies
### Should be manually installed before setup:
- [PyTorch](https://pytorch.org/)

### Optional
- [Jupyter](http://jupyter.org/)

## Setup

1. Install the dependencies mentioned above.
2. Clone this repository

```
git clone git@version.aalto.fi:lundelj2/IROS2018-Hallucinating-Robots.git 
cd IROS2018-Hallucinating-Robots/
```
3. Fetch the training and test set either by:
    1. Running the installation script install.sh or
    2. Create the folder datasets/ and manually place the
    [dataset](https://drive.google.com/drive/u/1/folders/1krNFnAcJRq7za9mtRX59lAegJy2VCr1u) there.
4. Run a training instance either by:
    1. Running the python script 
```
python hallucinating.py
```
    2. or, alternatively, if the user installed jupyter run the jupyter notebook hallucinating.ipynb


