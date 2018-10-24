# HandDigit
My first MLPack Machine Learning Project which aimed to predict the sign language digits based on images from a Kaggle Dataset using
an over-simplistic approach with Knn.

## Requirements
* OpenCV
* Armadillo
* MLPack
* OpenMP

## Dataset
Included!

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
