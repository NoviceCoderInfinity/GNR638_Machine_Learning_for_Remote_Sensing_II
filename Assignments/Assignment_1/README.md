# Problem Statement
---
The problem statement is mentioned explicitly in `Assignment_1.txt` \
As per the instructions, we were required to make changes to a `Scene-Recognition-with-Bag-of-Words` directory on github. We've the forked the repository and added it as a sub-module to this main repo. It is situated in `Assignments/Assignment_1`

## Cloning the Repository
```
git clone --recurse-submodules https://github.com/NoviceCoderInfinity/GNR638_Machine_Learning_for_Remote_Sensing_II
```

## Setting Up the Environment
```
conda create -n assign_1 python=3.7
conda activate assign_1
```

## Installing the Required Libraries
```
conda install -c menpo cyvlfeat
pip install -r requirements.txt
```

## Running the Code
In order to run the code, you've to navigate inside the `code` directory located at `Assignments/Assignment_1/Scene-Recognition-with-Bag-of-Words/code` and run the following command
```
python proj3.py --feature <FEATURE_NAME> --classifier <CLASSIFIER_NAME>
```