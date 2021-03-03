![verre vin](https://user-images.githubusercontent.com/29388984/109872963-1c9e8e80-7c3b-11eb-92f1-5e29efcf705a.PNG)

# Wine-quality-prediction




## Task
The objective is to develop a model for predicting the quality of a white wine using certain characteristics of the wine. The response variable, Y, is the wine quality score
and is an ordinal variable ranging from 1 to 3 where:\


1 = lower\
2 = medium\
3 = superior

The problem of predicting an ordinal variable is interesting because there are several ways to
approach it. The following article provides an interesting overview of the issue.



## Data

The data is comprised of 2000 observations and the input variables are based on physicochemical tests

fixedacidity \
volatileacidity \
citricacid \
residualsugar \
chlorides \
freesulfurdioxide \
totalsulfurdioxide \
density \
pH \
sulphates \
alcohol



## Results

Below are given the different models trained as well as their accuracy.

* **BASE** corresponds to models developed on the original dataset
* **RFE** refers to Recursive feature elimination for feature selection
* **VAR IMP** refers to variable importance provided from a GBM. 
* **PCA** refers to principal componnent analysis for feature reduction

| Models | Accuracy |
|:-:|:-:|
|H2O AutoML BASE|0,6725|
|H2O Ensemble RF + GBM BASE|0,6575|
|H2O Ensemble RF + GBM PCA|0,6575|
|Random Forest BASE|0,6525|
|H2O Ensemble RF + GBM VAR IMP|0,65|
|H2O AutoML PCA|0,6425|
|H2O GBM BASE|0,64|
|H2O Random Forest BASE|0,6375|
|H2O Ensemble RF + GBM RFE|0,6375|
|KNN BASE|0,635|
|Random Forest RFE|0,6325|
|H2O AutoML RFE|0,63|
|RF PCA|0,63|
|Random Forest VAR IMP|0,625|
|H2O AutoML VAR IMP|0,62|
|Ordinal Random Forest BASE     |0,615|
|GBM BASE|0,5825|
|SVM radial BASE|0,565|
|SVM polynomial BASE|0,555|
|Régression linéaire BASE|0,5325|
|Régression logistique ord BASE|0,525|
|Naïf - classe majoritaire Y = 2|0,4525|

## Resources

The following article presents different techniques for ordinal classification problems.

Gutierrez, P. A., Perez-Ortiz, M., Sanchez-Monedero, J., Fernandez-Navarro, F., & Hervas-
Martinez, C. (2015). Ordinal regression methods: survey and experimental study. IEEE
Transactions on Knowledge and Data Engineering, 28 (1), 127-146.
