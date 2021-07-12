#Loading libraries
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(gbm)
library(caTools)
library(e1071)
library(mice)
library(MASS)
library(class)
library(flexclust)
library(party)

#Loading data
setwd("~/Downloads")
set.seed(1)
train2016 <- read.csv("train2016.csv")
train2016 <- train2016[ , -1]
imputed <- complete(mice(train2016[ , -6]))
polling <- imputed
polling$Party <- train2016$Party
summary(polling)

#Splitting the data
set.seed(1)
spl <- sample.split(polling$Party, SplitRatio = 0.7)
pollingTrain <- subset(polling, spl == TRUE)
pollingTest <- subset(polling, spl == FALSE)
pollingTrain1 <- subset(pollingTrain, pollingTrain$generation == "0 - 20")
pollingTrain2 <- subset(pollingTrain, pollingTrain$generation == "20 - 40")
pollingTrain3 <- subset(pollingTrain, pollingTrain$generation == "40 - 60")
pollingTrain4 <- subset(pollingTrain, pollingTrain$generation == "60 - Up")

pollingTest1 <- subset(pollingTest, pollingTest$generation == "0 - 20")
pollingTest2 <- subset(pollingTest, pollingTest$generation == "20 - 40")
pollingTest3 <- subset(pollingTest, pollingTest$generation == "40 - 60")
pollingTest4 <- subset(pollingTest, pollingTest$generation == "60 - Up")

train_CART1 <- train(Party ~ ., data = pollingTrain1, method = "rpart", trControl = folds)
pred_CART1 <- predict(train_CART1, newdata = pollingTest1, type = "prob")
probs_CART1 <- ifelse(pred_CART1[ , 2] > 0.5, "Republican", "Democrat")
accuracy_CART1 <- round(mean(pollingTest1$Party == probs_CART1)*100, 2)

train_CART2 <- train(Party ~ ., data = pollingTrain2, method = "rpart", trControl = folds)
pred_CART2 <- predict(train_CART2, newdata = pollingTest2, type = "prob")
probs_CART2 <- ifelse(pred_CART2[ , 2] > 0.5, "Republican", "Democrat")
accuracy_CART2 <- round(mean(pollingTest2$Party == probs_CART2)*100, 2)

train_CART3 <- train(Party ~ ., data = pollingTrain3, method = "rpart", trControl = folds)
pred_CART3 <- predict(train_CART3, newdata = pollingTest3, type = "prob")
probs_CART3 <- ifelse(pred_CART3[ , 2] > 0.5, "Republican", "Democrat")
accuracy_CART3<- round(mean(pollingTest3$Party == probs_CART3)*100, 2)

train_CART4 <- train(Party ~ ., data = pollingTrain4, method = "rpart", trControl = folds)
pred_CART4 <- predict(train_CART4, newdata = pollingTest4, type = "prob")
probs_CART4 <- ifelse(pred_CART4[ , 2] > 0.5, "Republican", "Democrat")
accuracy_CART4 <- round(mean(pollingTest4$Party == probs_CART4)*100, 2)

allProb_CART <- c(prob)
#Exploratory Data Analysis
ggplot(data = pollingTrain, aes(x = YOB)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black")

#Baseline model
table(pollingTest$Party)
accuracy_Base <- round((sum(pollingTest$Party == "Democrat")/nrow(pollingTest))*100, 2) #since Democrat has the higher freq

#Create CART model using cp
set.seed(1)
folds <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
cpGrid <- expand.grid(.cp = seq(0.001, 0.5, 0.001))
train_CART <- train(Party ~ ., 
					data = pollingTrain,
					method = "rpart",
					trControl = folds)
train_CART
pred_CART <- predict(train_CART, newdata = pollingTest, type = "prob")
probs_CART <- ifelse(pred_CART[ , 2] > 0.5, "Republican", "Democrat")
accuracy_CART <- round(mean(pollingTest$Party == probs_CART)*100, 2)

#Create randomForest model
set.seed(1)
mtryGrid <- expand.grid(.mtry = seq(2, 107, 5))
train_RF <- train(Party ~ .,
				data = pollingTrain,
				method = "rf",
				ntree = 1000,
				tuneGrid = mtryGrid,
				trControl = folds)
train_RF
predRandom <- predict(train_RF, newdata = pollingTest)
accuracy_Random <- round(mean(pollingTest$Party == predRandom)*100, 2)

#Create SVM
train_SVM <- train(Party ~ ., data = pollingTrain, method = "svmRadial", trControl = folds)
train_SVM
predSVM <- predict(svmModel, newdata = pollingTest)
accuracy_SVM <- round(mean(pollingTest$Party == predSVM)*100, 2)

#Create regularized regression -- Lasso, Ridge, etc
glmnet_grid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1),
                           lambda = seq(.01, .2, length = 20))
glmnet_fit <- train(Party ~ ., data = pollingTrain,
                    method = "glmnet",
                    preProcess = c("center", "scale"),
                    tuneGrid = glmnet_grid,
                    trControl = folds)
glmnet_fit
plot(glmnet_fit)
predLog <- predict(glmnet_fit, newdata = pollingTest, type = "prob")
probs <- ifelse(predLog[ , 2] > 0.5, "Republican", "Democrat")
accuracy_Log <- round(mean(pollingTest$Party == probs)*100, 2)

#Create linear discriminant analysis
ldaModel <- train(Party ~ ., data = pollingTrain, method = "lda", trControl = folds)
predLDA <- predict(ldaModel, newdata = pollingTest, type = "prob")
probsLDA <- ifelse(predLDA[ , 2] > 0.5, "Republican", "Democrat")
accuracy_LDA <- round(mean(pollingTest$Party == probsLDA)*100, 2)

#Create quadratic discriminant analysis
qdaModel <- train(Party ~ ., data = pollingTrain, method = "qda", trControl = folds)
predQDA <- predict(qdaModel, newdata = pollingTest, type = "prob")
probsQDA <- ifelse(predQDA[ , 2] > 0.5, "Republican", "Democrat")
accuracy_QDA <- round(mean(pollingTest$Party == probsQDA)*100, 2)

#Create Boosting trees
gbm_Grid <- expand.grid(shrinkage = seq(0.001, 0.1, length = 3),
						interaction.depth = seq(1, 4, 1),
						n.trees = seq(50, 1000, 50),
						n.minobsinnode = 20)
gbm_fit <- train(Party ~ ., data = pollingTrain,
						method = "gbm",
						trControl = folds,
						verbose = FALSE
						tuneGrid = gbm_Grid)
gbm_fit
predBoosting <- predict(gbm_fit, newdata = pollingTest, type = "response", gbm.perf(gbm_fit)
predBoosting <- as.data.frame(predBoosting)
probsBoosting <- ifelse(predBoosting[ , 2] > 0.5, "Republican", "Democrat")
accuracy_Boosting <- round(mean(pollingTest$Party == probsBoosting)*100, 2)

#Create naive bayes classifier
train_NB <- train(Party ~ ., data = pollingTrain, method = "nb", trControl = folds)
predNBayes <- predict(nBayes, newdata = pollingTest)
accuracy_nBayes <- round(mean(pollingTest$Party == predNBayes)*100, 2)

##Create Clusters
set.seed(1)
A <- model.matrix(~ . +0, data = pollingTrain[ , -107])
B <- model.matrix(~ . +0, data = pollingTest[ , -107])
km <- kmeans(A, centers = 6)
km.kcca <- as.kcca(km, A)
clusterTrain <- predict(km.kcca)
clusterTest <- predict(km.kcca, newdata = B)

#Assign each cluster to training and testing set
pollingTrain1 <- subset(pollingTrain, clusterTrain == 1)
pollingTrain2 <- subset(pollingTrain, clusterTrain == 2)
pollingTrain3 <- subset(pollingTrain, clusterTrain == 3)
pollingTrain4 <- subset(pollingTrain, clusterTrain == 4)
pollingTrain5 <- subset(pollingTrain, clusterTrain == 5)
pollingTrain6 <- subset(pollingTrain, clusterTrain == 6)

pollingTest1 <- subset(pollingTest, clusterTest == 1)
pollingTest2 <- subset(pollingTest, clusterTest == 2)
pollingTest3 <- subset(pollingTest, clusterTest == 3)
pollingTest4 <- subset(pollingTest, clusterTest == 4)
pollingTest5 <- subset(pollingTest, clusterTest == 5)
pollingTest6 <- subset(pollingTest, clusterTest == 6)

##Using predictive models applied to each training cluster
#randomForest model

predRandom1 <- predict(randomTree1, newdata = pollingTest1)
predRandom2 <- predict(randomTree2, newdata = pollingTest2)
predRandom3 <- predict(randomTree3, newdata = pollingTest3)
predRandom4 <- predict(randomTree4, newdata = pollingTest4)
predRandom5 <- predict(randomTree5, newdata = pollingTest5)
predRandom6 <- predict(randomTree6, newdata = pollingTest6)

#Logistic model
logModel1 <- glm(Party ~ ., data = pollingTrain1, family = "binomial")
logModel2 <- glm(Party ~ ., data = pollingTrain2, family = "binomial")
logModel3 <- glm(Party ~ ., data = pollingTrain3, family = "binomial")
logModel4 <- glm(Party ~ ., data = pollingTrain4, family = "binomial")
logModel5 <- glm(Party ~ ., data = pollingTrain5, family = "binomial")
logModel6 <- glm(Party ~ ., data = pollingTrain6, family = "binomial")

predLog1 <- predict(logModel1, newdata = pollingTest1, type = "response")
predLog2 <- predict(logModel2, newdata = pollingTest2, type = "response")
predLog3 <- predict(logModel3, newdata = pollingTest3, type = "response")
predLog4 <- predict(logModel4, newdata = pollingTest4, type = "response")
predLog5 <- predict(logModel5, newdata = pollingTest5, type = "response")
predLog6 <- predict(logModel6, newdata = pollingTest6, type = "response")

#Generalized boosting tree
fitBoosting1 <- gbm(Party ~ ., data = pollingTrain1, distribution = "multinomial", n.trees = 1000, cv.folds = 10, shrinkage = 0.01, interaction.depth = 4)
fitBoosting2 <- gbm(Party ~ ., data = pollingTrain2, distribution = "multinomial", n.trees = 1000, cv.folds = 10, shrinkage = 0.01, interaction.depth = 4)
fitBoosting3 <- gbm(Party ~ ., data = pollingTrain3, distribution = "multinomial", n.trees = 1000, cv.folds = 10, shrinkage = 0.01, interaction.depth = 4)
fitBoosting4 <- gbm(Party ~ ., data = pollingTrain4, distribution = "multinomial", n.trees = 1000, cv.folds = 10, shrinkage = 0.01, interaction.depth = 4)
fitBoosting5 <- gbm(Party ~ ., data = pollingTrain5, distribution = "multinomial", n.trees = 1000, cv.folds = 10, shrinkage = 0.01, interaction.depth = 4)
fitBoosting6 <- gbm(Party ~ ., data = pollingTrain6, distribution = "multinomial", n.trees = 1000, cv.folds = 10, shrinkage = 0.01, interaction.depth = 4)

predBoosting1 <- predict(fitBoosting1, newdata = pollingTest1, type = "response", n.trees = gbm.perf(fitBoosting1))
predBoosting2 <- predict(fitBoosting2, newdata = pollingTest2, type = "response", n.trees = gbm.perf(fitBoosting2))
predBoosting3 <- predict(fitBoosting3, newdata = pollingTest3, type = "response", n.trees = gbm.perf(fitBoosting3))
predBoosting4 <- predict(fitBoosting4, newdata = pollingTest4, type = "response", n.trees = gbm.perf(fitBoosting4))
predBoosting5 <- predict(fitBoosting5, newdata = pollingTest5, type = "response", n.trees = gbm.perf(fitBoosting5))
predBoosting6 <- predict(fitBoosting6, newdata = pollingTest6, type = "response", n.trees = gbm.perf(fitBoosting6))

#SVM
svmModel1 <- svm(Party ~ ., data = pollingTrain1, kernel = "radial", cost = 1, gamma = 0.00444444)
svmModel2 <- svm(Party ~ ., data = pollingTrain2, kernel = "radial", cost = 1, gamma = 0.00444444)
svmModel3 <- svm(Party ~ ., data = pollingTrain3, kernel = "radial", cost = 1, gamma = 0.00444444)

predSVM1 <- predict(svmModel1, newdata = pollingTest1)
predSVM2 <- predict(svmModel2, newdata = pollingTest2)
predSVM3 <- predict(svmModel3, newdata = pollingTest3)

#Testing models based on clusters
allActuals <- c(pollingTest1$Party, pollingTest2$Party, pollingTest3$Party, pollingTest4$Party, pollingTest5$Party)

allPredRandom <- c(predRandom1, predRandom2, predRandom3, predRandom4, predRandom5)

allPredLog <- c(predLog1, predLog2, predLog3, predLog4, predLog5)

predBoosting1 <- as.data.frame(predBoosting1)
predBoosting2 <- as.data.frame(predBoosting2)
predBoosting3 <- as.data.frame(predBoosting3)
predBoosting4 <- as.data.frame(predBoosting4)
predBoosting5 <- as.data.frame(predBoosting5)
colnames(predBoosting1) <- c("Democrat", "Republican")
colnames(predBoosting2) <- c("Democrat", "Republican")
colnames(predBoosting3) <- c("Democrat", "Republican")
colnames(predBoosting4) <- c("Democrat", "Republican")
colnames(predBoosting5) <- c("Democrat", "Republican")

allPredBoosting <- c(predBoosting1[ , 2], predBoosting2[ , 2], predBoosting3[ , 2], predBoosting4[ , 2], predBoosting5[ , 2])

allProbsLog <- ifelse(allPredLog > 0.5, 2, 1)

allProbsBoosting <- ifelse(allPredBoosting > 0.5, 2, 1)

allPredSVM <- c(predSVM1, predSVM2, predSVM3)

mean(allActuals == allPredRandom)

mean(allActuals == allProbsLog)

mean(allActuals == allProbsBoosting)

mean(allActuals == allPredSVM)

#Create K-Nearest Neighbors
knn.pred <- knn(A, B, pollingTrain$Party, k = 20)
accuracy_KNN <- round(mean(pollingTest$Party == knn.pred)*100, 2)

#-------------------------Submission----------------------------------------#

#Loading Test2016 data
Test2016 <- read.csv("test2016.csv", na.strings = c("", " ", "NA"))
set.seed(1)
Test <- complete(mice(Test2016[ , - 1]))
Test$USER_ID <- Test2016$USER_ID
predTest <- predict(randomTree, newdata = Test)
predTest1 <- predict(logModel, newdata = Test, type = "response")
predTest2 <- predict(fit, newdata = Test)
predTest3 <- predict(CART, newdata = Test, type = "class")
probs1 <- ifelse(predTest1 > 0.5, "Republican", "Democrat")
predTestBoosting <- predict(fitBoosting, newdata = Test, type = "response")
predTestBoosting <- as.data.frame(predTestBoosting)
probsBoostingTest <- ifelse(predTestBoosting[ , 2] > 0.5, "Republican", "Democrat")
predTest_SVM <- predict(svmModel, newdata = Test)
predTestNBayes <- predict(nBayes, newdata = Test)

#Write CSV file
MySubmission <- data.frame(USER_ID = Test$USER_ID, predTest, predTest2, predTest3, probsBoostingTest, predTest_SVM, predTestNBayes)
write.csv(MySubmission, "SubmissionSimpleModel11.csv", row.names = FALSE)
