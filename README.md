# DTKD

This is the Pytorch and RecBole implementation for our paper: Dual-Teacher Knowledge Distillation for Strict Cold-Start Recommendation. 

## How to use:
1. First make sure to creat a 'saved' file in the following structure:
```
- saved
  - DirectAUT
  - LinearT
  - MLPS
  - ml-1m
```

2. Train two types of teachers
``python run_recbole.py -r cs -m DirectAUT``
``python run_recbole.py -r cs -m LinearT``

3. Find the saved model paths for DirectAUT and LinearT in the saved file and copy the paths into the config.yaml
Then run the DTKD algorithm:
``python run_recbole.py -r kd``
