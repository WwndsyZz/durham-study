# 0. Set up the environment and random seed
set.seed(123) 
library(caret)      
library(ggplot2)    
library(corrplot)    
library(pROC)       
library(xgboost)     
library(e1071)       

# 1. Data preprocessing
data <- read.csv("C:/Users/wwnds/Downloads/bank_personal_loan.csv")
str(data)
summary(data)
if(anyNA(data)){
  num_cols <- sapply(data, is.numeric)
  data[num_cols] <- lapply(data[num_cols], function(x) 
    ifelse(is.na(x), median(x, na.rm=TRUE), x))
  fac_cols <- sapply(data, is.factor)
  data[fac_cols] <- lapply(data[fac_cols], function(x) {
    m <- names(sort(table(x), decreasing=TRUE))[1]
    replace(x, is.na(x), m) })
}

data$ID <- NULL
data$ZIP.Code <- NULL 
data$Personal.Loan <- factor(data$Personal.Loan, levels=c(0,1), labels=c("No","Yes"))
data$Education <- factor(data$Education, levels=c(1,2,3),
                         labels=c("Undergrad","Graduate","Adv/Prof"))
data$Family <- factor(data$Family)
data$Securities.Account <- factor(data$Securities.Account, levels=c(0,1), labels=c("No","Yes"))
data$CD.Account         <- factor(data$CD.Account, levels=c(0,1), labels=c("No","Yes"))
data$Online             <- factor(data$Online, levels=c(0,1), labels=c("No","Yes"))
data$CreditCard         <- factor(data$CreditCard, levels=c(0,1), labels=c("No","Yes"))
if(any(data$Experience < 0)){
  data$Experience[data$Experience < 0] <- 0
}

set.seed(123) 
trainIndex <- createDataPartition(data$Personal.Loan, p=0.7, list=FALSE) 
trainData <- data[trainIndex, ]
testData  <- data[-trainIndex, ]
dim(trainData); dim(testData)
prop.table(table(trainData$Personal.Loan))  
prop.table(table(testData$Personal.Loan))   

# 2. Visual analysis
numericVars <- c("Age","Experience","Income","CCAvg","Mortgage")
par(mfrow=c(2,3)) 
for(var in numericVars){
  hist(trainData[[var]], main=paste("Histogram of", var), xlab=var, col="skyblue", border="white")
}
par(mfrow=c(1,1))  

par(mfrow=c(2,3))
for(var in numericVars){
  boxplot(trainData[[var]], main=paste("Boxplot of", var), ylab=var, col="orange")
}
par(mfrow=c(1,1))
cateVars <- c("Education","Family","Securities.Account","CD.Account","Online","CreditCard")
par(mfrow=c(2,3))
for(var in cateVars){
  barplot(table(trainData[[var]]), main=paste("Distribution of", var),
          col="lightgreen", ylab="Count")
}
par(mfrow=c(1,1))

corrMat <- cor(as.data.frame(lapply(trainData[, numericVars], as.numeric)))  
print(round(corrMat, 3))  
corrplot(corrMat, method="shade", addCoef.col="black", tl.col="black", tl.srt=45,
         title="Correlation Matrix of Numeric Features", mar=c(0,0,1,0))
ggplot(trainData, aes(x=Personal.Loan, y=Income, fill=Personal.Loan)) + 
  geom_boxplot() + 
  labs(title="Income by Personal Loan Acceptance", x="Personal Loan", y="Income") +
  theme_bw()
ggplot(trainData, aes(x=Personal.Loan, y=CCAvg, fill=Personal.Loan)) + 
  geom_boxplot() + 
  labs(title="Credit Card Avg Spending by Personal Loan Acceptance", x="Personal Loan", y="CCAvg") +
  theme_bw()
ggplot(trainData, aes(x=Education, fill=Personal.Loan)) + 
  geom_bar(position="fill") + 
  labs(title="Loan Acceptance Rate by Education", y="Proportion of Yes") + 
  scale_y_continuous(labels=scales::percent) + 
  theme_bw()
ggplot(trainData, aes(x=CD.Account, fill=Personal.Loan)) + 
  geom_bar(position="fill") + 
  labs(title="Loan Acceptance Rate by CD Account", x="CD Account", y="Proportion of Yes") + 
  scale_y_continuous(labels=scales::percent) + 
  theme_bw()

# 3. Model training
trainCtrl <- trainControl(method="cv", number=5, classProbs=TRUE,
                          summaryFunction=twoClassSummary, savePredictions="final")
set.seed(123)
logModel <- train(Personal.Loan ~ ., data=trainData, method="glm", family="binomial",
                  trControl=trainCtrl, metric="ROC")
cartGrid <- expand.grid(.cp = seq(0.001, 0.05, by=0.002))
set.seed(123)
treeModel <- train(Personal.Loan ~ ., data=trainData, method="rpart",
                   trControl=trainCtrl, tuneGrid=cartGrid, metric="ROC")
rfGrid <- expand.grid(.mtry = 2:6)
set.seed(123)
rfModel <- train(Personal.Loan ~ ., data=trainData, method="rf",
                 trControl=trainCtrl, tuneGrid=rfGrid, metric="ROC")
xgbGrid <- expand.grid(nrounds = c(100, 200),
                       max_depth = c(3, 6, 9),
                       eta = c(0.1, 0.3),
                       gamma = 0,
                       colsample_bytree = 1,
                       min_child_weight = 1,
                       subsample = 1)
set.seed(123)
xgbModel <- train(Personal.Loan ~ ., data=trainData, method="xgbTree",
                  trControl=trainCtrl, tuneGrid=xgbGrid, metric="ROC")

# 4. Hyperparameter tuning results
cat("Logistic Regression has no tunable hyperparameters.\n")
cat("Decision Tree best cp:", treeModel$bestTune$cp, "\n")
cat("Random Forest best mtry:", rfModel$bestTune$mtry, "\n")
cat("XGBoost best params:\n")
print(as.list(xgbModel$bestTune)) 


# 5. Model Evaluation
logProb  <- predict(logModel, testData, type="prob")[,"Yes"]
treeProb <- predict(treeModel, testData, type="prob")[,"Yes"]
rfProb   <- predict(rfModel, testData, type="prob")[,"Yes"]
xgbProb  <- predict(xgbModel, testData, type="prob")[,"Yes"]
logROC  <- roc(response=testData$Personal.Loan, predictor=logProb, levels=c("No","Yes"))
treeROC <- roc(testData$Personal.Loan, treeProb, levels=c("No","Yes"))
rfROC   <- roc(testData$Personal.Loan, rfProb, levels=c("No","Yes"))
xgbROC  <- roc(testData$Personal.Loan, xgbProb, levels=c("No","Yes"))

logAUC  <- as.numeric(logROC$auc)
treeAUC <- as.numeric(treeROC$auc)
rfAUC   <- as.numeric(rfROC$auc)
xgbAUC  <- as.numeric(xgbROC$auc)
plot(logROC, col="blue", lwd=2, main="ROC Curves on Test Data")
lines(treeROC, col="green", lwd=2)
lines(rfROC, col="orange", lwd=2)
lines(xgbROC, col="red", lwd=2)
legend("bottomright",
       legend=c(sprintf("Logistic (AUC=%.3f)", logAUC),
                sprintf("Decision Tree (AUC=%.3f)", treeAUC),
                sprintf("Random Forest (AUC=%.3f)", rfAUC),
                sprintf("XGBoost (AUC=%.3f)", xgbAUC)),
       col=c("blue","green","orange","red"),
       lwd=2, title="Model (AUC)")

get_metrics <- function(true, pred_prob, threshold=0.5, positive="Yes"){
  pred_class <- factor(ifelse(pred_prob >= threshold, "Yes","No"), levels=c("No","Yes"))
  cm <- confusionMatrix(pred_class, true, positive=positive)
  acc <- cm$overall["Accuracy"]
  sens <- cm$byClass["Sensitivity"]    
  spec <- cm$byClass["Specificity"]
  ppv <- cm$byClass["Pos Pred Value"]  
  npv <- cm$byClass["Neg Pred Value"]
  f1  <- if(ppv + sens == 0) 0 else (2 * ppv * sens / (ppv + sens))
  metrics <- c(Accuracy=acc, Sensitivity=sens, Specificity=spec,
               Precision=ppv, F1=f1)
  return(list(cm=cm$table, metrics=metrics))
}

models <- list(Logistic=logProb, DecisionTree=treeProb, RandomForest=rfProb, XGBoost=xgbProb)
for(m in names(models)){
  res <- get_metrics(testData$Personal.Loan, models[[m]], threshold=0.5, positive="Yes")
  cat("Model:", m, "\n")
  print(res$cm)                     
  print(round(res$metrics, 3))        
  cat("----\n")
}

# 6. Optimal threshold selection
auc_values <- c(Logistic=logAUC, DecisionTree=treeAUC, RandomForest=rfAUC, XGBoost=xgbAUC)
best_model_name <- names(auc_values)[which.max(auc_values)]
cat("Best model by AUC is:", best_model_name, "with AUC =", max(auc_values), "\n")
best_model <- switch(best_model_name,
                     "Logistic" = logModel,
                     "DecisionTree" = treeModel,
                     "RandomForest" = rfModel,
                     "XGBoost" = xgbModel)
cv_preds <- best_model$pred  
if("parameter" %in% colnames(cv_preds)){
  for(param in names(best_model$bestTune)){
    cv_preds <- cv_preds[ cv_preds[[param]] == best_model$bestTune[[param]], ]
  }
}
true_cv <- cv_preds$obs           
prob_cv <- cv_preds[, "Yes"]      
thresholds <- seq(0.01, 0.99, by=0.01)
best_F1 <- 0
best_threshold <- 0.5
for(thr in thresholds){
  pred_yes <- prob_cv >= thr
  TP <- sum(pred_yes & true_cv=="Yes")
  FP <- sum(pred_yes & true_cv=="No")
  FN <- sum(!pred_yes & true_cv=="Yes")
  if(TP + FP == 0) {
    next  
  }
  precision <- TP / (TP + FP)
  recall <- if((TP+FN) == 0) 0 else TP / (TP + FN)
  if(precision + recall == 0) {
    F1 <- 0 
  } else {
    F1 <- 2 * precision * recall / (precision + recall)
  }
  if(F1 > best_F1){
    best_F1 <- F1
    best_threshold <- thr
  }
}
cat(sprintf("Best threshold for %s (based on CV predictions) = %.2f with F1 = %.3f\n",
            best_model_name, best_threshold, best_F1))

# 7. Final test set evaluation
best_prob_test <- models[[best_model_name]]  
best_pred_class <- factor(ifelse(best_prob_test >= best_threshold, "Yes","No"), levels=c("No","Yes"))
best_cm <- confusionMatrix(best_pred_class, testData$Personal.Loan, positive="Yes")
best_acc  <- best_cm$overall["Accuracy"]
best_sens <- best_cm$byClass["Sensitivity"]
best_spec <- best_cm$byClass["Specificity"]
best_ppv  <- best_cm$byClass["Pos Pred Value"]
best_f1   <- 2 * best_ppv * best_sens / (best_ppv + best_sens)
cat("Final Model Evaluation on Test Set:\n")
cat("Best Model:", best_model_name, "Threshold:", best_threshold, "\n")
print(best_cm$table) 
cat(sprintf("Accuracy = %.3f, Recall = %.3f, Precision = %.3f, F1 = %.3f, AUC = %.3f\n",
            best_acc, best_sens, best_ppv, best_f1, max(auc_values)))
