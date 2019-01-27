setwd("C:/Users/LCDR-Data/Dropbox (Personal)/2018Spring/STP 494 MachineLearning/Homework/2018S_MachingLearning/input")
library(dplyr)
library(ggplot2)
library(caret)
library(DMwR)
library(nnet)
library(e1071)
library(devtools)
library(pROC)
library(glmnet)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
source_url('http://www.rob-mcculloch.org/2018_ml/webpage/notes/lift-loss.R')

##--- Import data and Clean
CreditData <- read.csv2(file="creditcard.csv", sep=",", dec =".", stringsAsFactors=FALSE)
CreditData <- CreditData[-1]

CreditData$Class <-as.factor(CreditData$Class)

head(CreditData)
str(CreditData)

range(CreditData$Amount)

#scale the variable "Amount"
scAmount <- scale(CreditData$Amount, center = TRUE)
CreditData$Amount <-scAmount
boxplot(Amount ~ Class, CreditData, axes = TRUE, outline = FALSE)

##Data Exploration
prop.table(table(CreditData$Class))
#The data we have is currently unbalanced.
#As recommended, we shall use alternative sampling techniques for train/test

#Random Split
set.seed(1234)
splitIndex <- createDataPartition(CreditData$Class, p=.50, list=FALSE, times = 1)

CreditTrain <- CreditData[splitIndex,]
CreditTest <- CreditData[-splitIndex,]

prop.table(table(CreditTrain$Class))
prop.table(table(CreditTest$Class))

##--- USE SMOTE (Synthetic Minority Resampling Techniques)

CreditTrain$Class <- as.factor(CreditTrain$Class)

CreditTrainSMOTE <- SMOTE(Class~. ,CreditTrain, perc.over = 100, perc.under=200)
#Not sure of it is valid to make dataset for testing.
CreditTestSMOTE <- SMOTE(Class~., CreditTest, petc.over=100, perc.under=200) #Not sure if Relevant
#check outcome
prop.table(table(CreditTrainSMOTE$Class))



##---Get the baseline accuracy rating
#the probability you would be right just guessing activitiy was not fraud
baseACC <- nrow(CreditData[CreditData$Class == 0,])/nrow(CreditData)*100



##--- Run a logisitc regression for comparison
x <- model.matrix(Class ~ ., CreditTrainSMOTE)[,-1]
glmmod <- glmnet(x, y=CreditTrainSMOTE$Class, alpha = 1, family = "binomial")

plot(glmmod, xvar = "lambda")

#Use CV to get lambda
cv.glmmod <- cv.glmnet(x, y = as.numeric(CreditTrainSMOTE$Class), alpha = 1)
plot(cv.glmmod)

bestlamda <- cv.glmmod$lambda.min

sprintf("The best lamda for the glmnet was %s . ", bestlamda)


#log predictions
x_test = model.matrix(Class~., CreditTest)[,-1]
y_test = as.numeric(CreditTest$Class)

predict_lasso_test = predict(cv.glmmod, newx = x_test, s=bestlamda)
PREDlasso <-predict(cv.glmmod, newx = x_test, s=bestlamda)

logAUC <- roc(response = y_test, predictor = as.numeric(predict_lasso_test), plot = TRUE, print.auc = TRUE)

plot()

##----Do a Neural Net!!!

decayValues = c(0.5, 0.1, .05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0001, 0.00001) #Decay values from the homework
sizeValues = c(2:29) #Recommended by "doug" on stackexchange https://stats.stackexchange.com/q/1097
maxD = length(decayValues)
maxS = length(sizeValues)
maxITS = 200

ACCmatrix = matrix(NA, nrow = maxS, ncol = maxD)
SENSmatrix = matrix(NA, nrow = maxS, ncol = maxD)
SPECIFmatrix = matrix(NA, nrow = maxS, ncol = maxD)
AUCmatrix = matrix(NA, nrow = maxS, ncol = maxD)

for(s in 1:maxS){
  for( d in 1:maxD){
    tempNN <- nnet(Class~., CreditTrainSMOTE, size = sizeValues[s], decay = decayValues[d], maxit = maxITS)
    tempPRED <- predict(tempNN, CreditTest, type = "class")
    tempCM <- confusionMatrix(factor(tempPRED, levels = c("0","1")), na.omit(factor(CreditTest$Class, levels = c("0", "1"))), positive = "1")
    tempAUC <-roc(response = CreditTest$Class, predictor = as.numeric(tempPRED))
    ACCmatrix[s,d] = tempCM$overall[["Accuracy"]]
    SENSmatrix[s,d] = tempCM$byClass[["Sensitivity"]]
    SPECIFmatrix[s,d] = tempCM$byClass[["Specificity"]] 
    AUCmatrix[s,d] = tempAUC$auc[1]
  }
}

#saves the matrices as rData files
saveRDS(ACCmatrix, paste0("ACCmatrix", Sys.Date(),".rds"))
saveRDS(SENSmatrix, paste0("SENSmatrix", Sys.Date(),".rds"))
saveRDS(SPECIFmatrix, paste0("SPECIFmatrix", Sys.Date(),".rds"))
saveRDS(AUCmatrix, paste0("AUCmatrix", Sys.Date(),".rds"))

#searches the AUC mmatrix for the optimal valuue

opt_vals <- which(AUCmatrix==AUCmatrix[which.is.max(AUCmatrix)], arr.ind=T) #the first index is  for rows (size) and the second is for col (decay)
opt_size = sizeValues[opt_vals[1]]
opt_decay = decayValues[opt_vals[2]]

sprintf("The optimal neural net, based on AUC, is a model with %s units and a decay value of %s", opt_size, opt_decay)

#add more iterations
maxITS = 500

optNN <- nnet(Class~., CreditTrainSMOTE, size = opt_size, decay = opt_decay, maxit = maxITS)
optPRED <- predict(optNN, CreditTest, type = "class")
optCM <- confusionMatrix(factor(optPRED, levels = c("0","1")), na.omit(factor(CreditTest$Class, levels = c("0", "1"))), positive = "1")
optAUC <- roc(response = CreditTest$Class, predictor = as.numeric(optPRED), plot = TRUE, col = "red", print.auc = TRUE)
opt_ACC <- optCM$overall[["Accuracy"]]
opt_SENS <- optCM$byClass[["Sensitivity"]]
opt_SPECIF <- optCM$byClass[["Specificity"]] 
opt_AUC <- optAUC$auc[1]

wts.in <-optNN$wts
struct <-optNN$n

par(mfrow=c(1,1), mar=numeric(4))
plot.nnet(optNN, bias = F, nid = F, rel.rsc = 100, circle.cex = 3, x.lab = rep(" ", 29),
          line.stag = .001, cex.val = .5, circle.col = "red", max.sp = F, alpha.val = .1)


lift.many.plot(list(as.numeric(PREDlasso), as.numeric(optPRED)), CreditTest$Class)


#black line disappeared , black line was the opttimal neural net

length(PREDlasso)
length(optPRED)
               