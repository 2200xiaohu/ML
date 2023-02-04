library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())
searchGridSubCol <- expand.grid(subsample = seq(0.5, 1,by=0.2), 
                                colsample_bytree = seq(0.5, 0.9, by=0.2),
                                max_depth = seq(7, 50, by=10),
                                min_child = c(1:5),
                                eta = seq(0.3, 1, by= 0.2),
                                gamma = seq(0, 3, by=1))
searchGridSubCol <- expand.grid(
  max_depth = 32,
  eta = 0.4259741,
  gamma = 0,
  min_child = 4.413045,
  subsample = 1,
  colsample_bytree=seq(0.5, 0.9, by=0.01)
)
searchGridSubCol <- expand.grid(
  max_depth = 2,
  eta = 0.6122684,
  gamma = 0,
  min_child_weight = 1.456841,
  subsample = 1,
  colsample_bytree=0.5
)
#                                

gamma = seq(0, 5, length = 20
            gamma=currentgamma,    "eta" = currentEta,                            )
#

ntrees <- 100
x<-model.matrix(Survived~.,Train)#为什么变小了
y<-model.frame(Survived~.,Train)[,"Survived"]
rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentDepth <- parameterList[["max_depth"]]
  currentEta <- parameterList[["eta"]]
  currentMinChild <- parameterList[["min_child"]]
  currentgamma <- parameterList[["gamma"]]
  #currentalpha <- parameterList[["alpha"]]
  xgboostModelCV <- xgb.cv(data =  x, nrounds = ntrees, nfold = 5, showsd = TRUE, label=y,
                           metrics = "error", verbose = TRUE, "eval_metric" = "error",
                           "objective" = "binary:logistic", "max.depth" = currentDepth,                              
                           "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate
                           , print_every_n = 10, "min_child_weight" = currentMinChild, booster = "gbtree",
                           #alpha=currentalpha,
                           gamma=currentgamma,eta=currentEta,
                           early_stopping_rounds = 15)
  
  xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
  numrounds<-min(which(xgboostModelCV$evaluation_log$test_error_mean == 
                         min(xgboostModelCV$evaluation_log$test_error_mean)))
  rmse <- xvalidationScores$test_error_mean[numrounds]
  trmse <- xvalidationScores$train_error_mean[numrounds]
  output <- return(c(rmse, trmse, currentSubsampleRate, currentColsampleRate, currentDepth, 
                     currentMinChild,currentEta,currentgamma,numrounds))})
#currentEta, 

output <- as.data.frame(t(rmseErrorsHyperparameters))
bestparam<-output[which(output$V1==min(output$V1),arr.ind=T),]
