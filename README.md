# Crop-GPA 2.0

## Introduction

Crop-GPA 2.0 is a deep-learning-based cross-species prediction framework for trait-associated single nucleotide polymorphisms (TA-SNPs). By integrating genotype-phenotype association (GPA) data from genome-wide association studies (GWAS), it adopts a transfer learning strategy of "pre-training-fine-tuning-model fusion" to achieve high-precision prediction of agronomy-related SNPs. The framework is applied to traits with limited training data such as Trait A and Trait B, successfully covering genome-wide TA-SNP prediction for 10 major crops and 8 key agronomic traits.

## Feature 

* Feature/Onehot.py:DNA feature extraction based on One-hot encoding
* Feature/DNA2vec.py:Vector representation method for DNA sequences
* Feature/DNABERT.py:DNA language model feature extraction based on BERT
* Feature/DNAshape.py:DNA structural feature extraction 


## Model

* Model/Pre-training.py:Implementation of cross-species pre-training model
* Model/Fine-tune.py:Fine-tuning module for specific traits
* Model/Fusion.py:Multi-model branch fusion strategy

## Setup environment

### create and activate virtual python environment

#### For Feature code:

* conda create -n Feature python=3.8
* conda activate Feature
* pip install Featurerequirements.txt

#### For Model code:

* conda create -n Model python=3.8
* conda activate Model
* pip install Modelrequirements.txt

