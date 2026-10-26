# aist4010-hw2-kaggle

in this repo, we have the following files : 

## config.py 
this file contain the config of character -> integer mapping, 
hyperparameters,
data preprocess config 
training config, 
etc.

## dataPreprocessor.py 

this code contain a class called DataPreprocessor, which read the FASTA file in ./data and map each character (amino acid) into a integer ID using a predefined hashmap
aa_alphabet. (this hashmap is in config)

## train.py 

Here contain the training loop 

## model.py 


Here contain the absolute baseline model. we will try to improve from this baseline. 

