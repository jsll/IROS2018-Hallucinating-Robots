# Repo 

This repository contains code corresponding to:

J. Lundell, F. Verdoja, V. Kyrki. **Hallucinating Robots: Inferring Obstacle Distances from Partial Laser Measurements.**
IEEE/RSJ International Conference on Intelligent Robots and Systemsi (IROS), 2018.

Please cite as:
@article{lundell2018hallucinating,
    title={Hallucinating robots: Inferring obstacle distances from partial laser measurements},
    author={Lundell, Jens and Verdoja, Francesco and Kyrki, Ville},
    journal={arXiv preprint arXiv:1805.12338},
    year={2018}
}


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
    2. Create the folder datasets/ and manually place the [dataset](http://irobotics.aalto.fi/) there.
4. Run a training instance either by:
    1. Running the python script 
```
python hallucinating.py
```
    2. or, alternatively, if the user installed jupyter run the jupyter notebook hallucinating.ipynb


