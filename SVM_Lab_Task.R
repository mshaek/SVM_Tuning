
## SVM lab task

# for this Lab task Kernlab library has been used. Kernlab library has more kernals than e1701.

library(ggplot2)
library(e1071)
library(caret)
library(kernlab)

# Dataset
fileURL <- "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
download.file(fileURL, destfile="breast-cancer-wisconsin.data", method="curl")
# read the data
df <- read.table("breast-cancer-wisconsin.data", na.strings = "?", sep=",")
str(df)
# Name the columns. 
# These names are displayed in the tree to facilitate semantic interpretation
df <- df [ ,-1]

#Removing columns with missing Values
sum (is.na(df))
colSums(sapply(df,is.na))
#rowSums(sapply(df,is.na))
df$V7 <- NULL

df$V11 <- factor(df$V11, levels=c(2,4), labels=c("1", "2"))
# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(df$V11, SplitRatio = 0.7)
training_set = subset(df, split == TRUE)
test_set = subset(df, split == FALSE)
#we'll first implement the trainControl() method. This will control all the computational overheads so that we can use the train() function provided by the caret package. 
#The training method will train our data on different algorithms.

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
#trainControl() method returns a list. We are going to pass this on our train() method.

svm_Linear <- train(V11 ~., data = training_set, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
# "center" & "scale" transforms the training data with mean value between "-1" to "1". 
#The "tuneLength" parameter holds an integer value. This is for tuning our algorithm.
# Check the results
svm_Linear
# passing SVM_LINEAR model to make prediction on test_set
test_pred <- predict(svm_Linear, newdata = test_set)
test_pred

#Evaluating the accuracy 
confusionMatrix(table(test_pred, test_set$V11))
# The output show the accurary 95.69%

# Linear Classifier takes C (cost) value as 1 by default. "GRID" dataframe has been tested with specific C values 
grid <- expand.grid(C = c(0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_Linear_Grid <- train(V11 ~., data = training_set, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_Linear_Grid
plot(svm_Linear_Grid)

# plot is showing that this Linear SVM with Grid is giving best accuracy on C = 0.5 to 1.75

# making prediction using this model on test_set
test_pred_grid <- predict(svm_Linear_Grid, newdata = test_set)
test_pred_grid
# Confusion Matrix
confusionMatrix(table(test_pred_grid, test_set$V11))
# Accuracy with this model is 95.69%


xtrain <- training_set[,-9]
ytrain <- training_set[,9]

kernfit <- ksvm(as.matrix(xtrain), as.factor(ytrain), type = "C-svc", kernel = 'rbfdot', C = 1, scaled = c())
kernfit 
# Plot training data
plot(kernfit, data = xtrain)

test_pred_rbfdot <- predict(kernfit, newdata = test_set[,-9])
test_pred_rbfydot
# Confusion Matrix
confusionMatrix(table(test_pred_polydot, test_set$V11))
#Accuracy for this model is 93.78% which is lower then previous two model svm_linear and svm_linear_grid

svm_sigmoid <-  train(V11 ~., data = training_set, method = "sigmoid",
                     trControl=trctrl,
                     preProcess = c("center", "scale"),
                     tuneGrid = grid,
                     tuneLength = 10)
svm_sigmoid

# making prediction using this model on test_set
test_pred_sigmoid <- predict(svm_sigmoid, newdata = test_set)
test_pred_sigmoid
# Confusion Matrix
confusionMatrix(table(test_pred_sigmoid, test_set$V11))
# Accuracy with this model is 94.26%
library(xgboost)
xgb_grid <- expand.grid(
  nrounds= 2400,
  eta=c(0.01,0.001,0.0001),
  lambda = 1,
  alpha =0
)


svm_xgbtree <-  train(V11 ~., data = training_set, method = "xgbTree",
                     trControl=trctrl,
                     preProcess = c("center", "scale"),
                     tuneGrid = xgb_grid
                  )
svm_xgbtree

# making prediction using this model on test_set
test_pred_xgbtree <- predict(svm_xgbtree, newdata = test_set)
test_pred_xgbtree
# Confusion Matrix
confusionMatrix(table(test_pred_xgbtree, test_set$V11))
# Accuracy with this model is 94.26%

# Comparing all four models it can be concluded that svm_linear and svm_linear_grid model performed better.

Models <- matrix(c( "95.69%", "95.69%", "94.25%", "94.26%", "94.26%"),ncol=4,byrow=TRUE)
colnames(Models) <- c("SVM Linear","Linear with grid search","SVM with RBFdot","SVM with Sigmoid", "SVM with XGBTree")
rownames(Models) <- c("Accuracy_Percentage")
Models <- as.table(Models)
Models


models <- data(Technique = c("SVM Linear","Linear with grid search","SVM with RBFdot","SVM with Sigmoid", "SVM with XGBTree"), 
               Accuracy_Percentage = c( "95.69%", "95.69%", "94.25%", "94.26%", "94.26%") )
models







# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)

# CART
set.seed(7)
fit.cart <- train(V11~., data=training_set, method="rpart", trControl=control)

# LDA
set.seed(7)
fit.lda <- train(V11~., data=training_set, method="lda", trControl=control)

# SVM
set.seed(7)
fit.svm <- train(V11~., data=training_set, method="svmRadial", trControl=control)

# kNN
set.seed(7)
fit.knn <- train(V11~., data=training_set, method="knn", trControl=control)

# Random Forest
set.seed(7)
fit.rf <- train(V11~., data=training_set, method="rf", trControl=control)
#install.packages("gmodels")
#library(class)
#CrossTable(svm_linear,svm_Linear_Grid, kernfit)

# collect resamples
results <- resamples(list(CART=fit.cart, LDA=fit.lda, SVM=fit.svm, KNN=fit.knn, RF=fit.rf))

# summarize differences between models
summary(results)

# dot plots of accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(results, scales=scales)
