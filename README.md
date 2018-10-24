# HandDigit
My first MLPack Machine Learning Project.

The main goal was to predict the number from a Dataset of sign language digit photographs with an over-simplistic approach using K-nn algorithm.

## Requirements
* OpenCV
* Armadillo
* MLPack
* OpenMP

## Dataset
Taken from [https://github.com/ardamavi/Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)
But I will include it here

## Usage
First create the DB using the images from the Dataset
```bash
HandDigit --update-db
```
Then you can test the results
```bash
HandDigit --test
```
Or try to predict from a Camera Image in interactive Mode
```bash
HandDigit --camera
```
