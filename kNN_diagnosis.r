#Applies a kNN prediction technique to biopsy diagnosis data
#ways to tune this model - experiment with different normalization methods (z-score example below)
#also, try different values for k - the number of nearest neighbors to compare against
#choosing a large k reduces noise in data, smooths patterns
#choosing smaller k allows for smaller more defined groups, but can lead to overfitting risk

library('class')
library('gmodels')

setwd("C:/code/machine_learning_r/2148OS_code/chapter 3")

wbcd = read.csv('wisc_bc_data.csv', stringsAsFactors = F)
#drop the ID variable
wbcd = wbcd[-1]
#assign labels to the diagnosis variable
wbcd$diagnosis = factor(wbcd$diagnosis, levels = c('B', 'M'), labels = c('Benign', 'Malignant'))

#normalize the numeric variable data - different variables have drastic scale differences
#the normalize function below assigns each observation a value between 0 and 1.  
normalize = function (x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

wbcd_n = as.data.frame(lapply(wbcd[2:31], normalize))

#split the data into train and test sets - about 20% reserved for training.
#can just take the last 100 for test set since the file has been sorted in random order

wbcd_train = wbcd_n[1:469,]
wbcd_train_labels = wbcd[1:469, 1]
wbcd_test = wbcd_n[470:569, ]
wbcd_test_labels = wbcd[470:569, 1]

#can also use the built in scale() function to normalize values based on z-score
#this allows for outliers to influence the data more heavily
wbcd_z = as.data.frame(scale(wbcd[-1]))


#build the predictive model using kNN technique.  Use k value of 21 - odd number close to square root of training set size.
wbcd_test_pred = knn(wbcd_train, wbcd_test, wbcd_train_labels, k = 21)

#Can use this crosstab to compare results of prediction
#CrossTable(wbcd_test_labels, wbcd_test_pred, prop.chisq = F)

