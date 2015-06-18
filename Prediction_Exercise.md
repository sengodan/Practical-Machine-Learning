Practical Machine Learning - Prediction Exercise
========================================================


## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here in the [.csv file](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here in the [.csv file](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.

## Reproducibility

Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis.

## Selecting predictors
### The packages used in the analysis

The following packages are required for this analysis.


```r
library(caret)
library(dplyr)
library(randomForest)
```

### Reading the data

Read the training data into R, identifying "NA", "" and "#DIV/0!" as NA strings


```r
##  fileUrl_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
##  download.file(fileUrl_train, destfile = "pml-training.csv" )
  pml_train <- read.csv("./pml-training.csv", na.strings=c("NA","","#DIV/0!"))
```

### Spliting the data into Training and Test sets

Split the plm_train into a training set (for model training) and a test set (for predicting the out of sample error), splitting on the classe variable (this is the variable of interest) with a 70-30 split


```r
  set.seed(300)

## Taking 70% for the training data and 30% for the test data
  inTrain <- createDataPartition(y = pml_train$classe, list = FALSE, p=0.7)
  trainData <- pml_train[inTrain,]
  testData <- pml_train[-inTrain,]
```

The analysis is now conducted purely on the trainData until the model is build and an out of sample error is needed.

### Identify variables that are mostly NAs

There are a number of na's in the dataset


```r
  table(is.na(trainData))
```

```
## 
##   FALSE    TRUE 
##  849688 1348232
```

Find which variables that are mostly na values 

```r
  naprops <- colSums(is.na(trainData))/nrow(trainData)
  mostlyNAs <- names(naprops[naprops > 0.75]) # Mostly being 75%
  mostlyNACols <- which(naprops > 0.75) # There are about 100 of them  
```

### Take a random small sample from the training data

Take a small sample of the training data to work with


```r
  smallTrain <- trainData %>% tbl_df %>% sample_n(size=1000)
```

Remove the variables that are made up of mostly NAs

```r
  smallTrain <- smallTrain[,-mostlyNACols]
```
### Remove row number and user name as candidate predictors

Remove the row number (X) and user_name column

```r
  smallTrain <- smallTrain[,-grep("X|user_name",names(smallTrain))]
```
### Remove the cvtd_timestamp variable as a candidate predictor

This factor variable makes prediction of the test set difficult and is reduandant when raw time data is available in the data set.


```r
  smallTrain <- smallTrain[,-grep("cvtd_timestamp",names(smallTrain))]
```
### Remove candidate predictors that have near zero variance

```r
  smallTrain <- smallTrain[,-nearZeroVar(smallTrain)]
```

### A look at the Data

The variable "classe" contains 5 levels: A, B, C, D and E. A plot of the outcome variable will allow us to see the frequency of each levels in the smallTrain data set and compare one another.


```r
plot(smallTrain$classe, col="blue", main="Bar Plot of levels of the variable classe within the small Training data set", xlab="classe levels", ylab="Frequency")
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png) 

### List of candidate predictors

```r
  modelVars <- names(smallTrain)
  modelVars1 <- modelVars[-grep("classe",modelVars)] # remove the classe var
```

The predictors for the machine learning are

```r
  modelVars1
```

```
##  [1] "raw_timestamp_part_1" "raw_timestamp_part_2" "num_window"          
##  [4] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [7] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [10] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [13] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [16] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [19] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [22] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [25] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [28] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [31] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [34] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [37] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [40] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [43] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [46] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [49] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [52] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [55] "magnet_forearm_z"
```

## Build a random forest model

Using a random forest with the predictors in modelVars1 to predict the classe variable.


```r
  cleanedTrainData <- trainData[,modelVars]
  modelFit <- randomForest(classe ~., data=cleanedTrainData, type="class")
```

## Get Error Estimates

Begin with an insample error estimate (from trainData - which is 70% of pml-training.csv)


```r
## Get the values predicted by the model
  predTrain <- predict(modelFit,newdata=trainData)

## Use a confusion matrix to get the insample error
  confusionMatrix(predTrain,trainData$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
```

The in sample error is unrealistically high.

Now getting an out of sample error estimate (from testData - which is 30% of pml-training.csv)

```r
  classe_col <- grep("classe",names(testData))
  predTest <- predict(modelFit, newdata = testData[,-classe_col], type="class")
  confusionMatrix(predTest,testData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    3    0    0    0
##          B    1 1136    4    0    0
##          C    0    0 1022    0    0
##          D    0    0    0  963    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9983          
##                  95% CI : (0.9969, 0.9992)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9979          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9974   0.9961   0.9990   0.9991
## Specificity            0.9993   0.9989   1.0000   0.9998   0.9998
## Pos Pred Value         0.9982   0.9956   1.0000   0.9990   0.9991
## Neg Pred Value         0.9998   0.9994   0.9992   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1930   0.1737   0.1636   0.1837
## Detection Prevalence   0.2848   0.1939   0.1737   0.1638   0.1839
## Balanced Accuracy      0.9993   0.9982   0.9981   0.9994   0.9994
```

The model has an out of sample accuracy of: 0.9983. The expected out-of-sample error is estimated at 0.0017.
The expected out-of-sample error is calculated as 1 - accuracy for predictions made against the cross-validation set. Our Test data set comprises 20 cases. With an accuracy above 99.83% on our cross-validation data, we can expect that very few, or none, of the test samples will be missclassified.

## Prediciting exercise activity using the model

### Load the pml-test data


```r
##   fileUrl_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
##   download.file(fileUrl_test, destfile = "pml-testing.csv" )
   pml_test <- read.csv("./pml-testing.csv", na.strings=c("NA","","#DIV/0!"))
```

Perform the prediction


```r
  # plmtest predicition
  predplm_test <- predict(modelFit, newdata = pml_test, type="class")
```

The final outcome is here


```r
print(predplm_test)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

