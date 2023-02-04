library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(doParallel)
library(pROC)
library(ROCR)
library(caret)
library(xgboost)
library(mlr)
cl <- makeCluster(detectCores()-8)
registerDoParallel(cl)
set.seed(NA)

#工作目录
setwd("C:\\Users\\nigel\\Desktop\\titanic\\data")

#import data
ig.data<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\data\\train.csv',header=T)



#---------data---------#
data<-na.omit(ig.data)%>%select(Survived,Pclass,Sex,Age,SibSp,Parch,Fare)
data<-na.omit(ig.data)%>%select(Survived,Sex,Pclass,Age,SibSp,Parch,Fare,Embarked)

data<-na.omit(ig.data)%>%select(Survived,Sex,FareAdj,FamilySize)
#性别转换
tmp<-hot.enc.sex(data)
data<-tmp
data<-data[,-4]
#港口转换
#s=0,c=1,q=2
tmp<-hot.enc.Embarked1(data)
data<-tmp
data<-data[,-12]
#data$Sex<-as.factor(data$Sex)
#data<-data[,-4]
data<-na.omit(data)
head(data)
#------------Feature select---------------#
#information gain
library(FSelectorRcpp)
ig_res =information_gain(Survived ~ .,data)


#rfe
control<-rfeControl(functions = rfFuncs,method="repeatedcv",repeats=5,number=10,allowParallel=T)

rfe.result<-rfe(data[,-1],factor(data[,1]),c(1:ncol(data[,-1])),rfeControl=control)#还是用这个
rfe.result#Sex, Pclass, Fare, Age, Parch
feature<-predictors(rfe.result)
#sbf
control.sbf<-sbfControl(functions=rfSBF,method="cv",repeats=5,number=10,allowParallel=T)
sbf.feature<-sbf(data[,-1],factor(data[,1]),sbfControl = control.sbf)
sbf.feature#Pclass, Sex, Age, Parch, Fare


#---------split data------------#
#data<-data%>%select(feature,Survived)
data<-data%>%select(Survived,Sex,Pclass,Fare,Age,Parch)

data<-na.omit(data)
data$Survived<-as.factor(data$Survived)
trainIndex <- createDataPartition(data$Survived, p = .7, 
                                  list = FALSE, 
                                  times = 1)

Train <- data[ trainIndex,]
Test <- data[ -trainIndex,]



#---------------------------Train model--------------------#
#基线模型
#存活的占比 0.406
nrow(filter(Train,Survived==1))/nrow(Train)

#preparing matrix 

x<-model.matrix(Survived~.,Train)#为什么变小了
y<-model.frame(Survived~.,Train)[,"Survived"]

xvals<-model.matrix(Survived~.,Test)
yvals<-model.frame(Survived~.,Test)[,"Survived"]

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "error",
               eta=0.4259741, gamma=0, max_depth=32, min_child_weight=4.413045, 
               subsample=1, colsample_bytree=0.81)

xgbcv <- xgb.cv( params = params, data = x,label=y,
                 nrounds = 2000, nfold = 10, showsd = T, 
                 stratified = T, print_every_n = 1, 
                 early_stop_round = 20, maximize = F,
                 prediction = T)

xgbCV <- xgb.cv(params = params,
                data = x,
                label = y,
                nrounds = 100,
                prediction = TRUE,
                showsd = TRUE,
                early_stopping_rounds = 10,
                maximize = TRUE,
                nfold = 10,
                stratified = TRUE)
str(xgbcv)
graph.data<-xgbcv$evaluation_log
#
ggplot()+geom_line(graph.data,mapping=aes(x=iter,y=train_error_mean),color='blue')+
  geom_line(graph.data,mapping=aes(x=iter,y=test_error_mean),color='red')
pdf(file = "learn.pdf",
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
my_plots
dev.off()

min(xgbcv$evaluation_log$test_error_mean)
numrounds<-min(which(xgbcv$evaluation_log$test_error_mean == 
                           min(xgbcv$evaluation_log$test_error_mean)))

fit <- xgboost(params = params,
               data = x,
               label = y,
               nrounds = numrounds)

#check performance
pre<-predict(fit,xvals,type="res")
predictions2<-as.factor(ifelse(predict(fit,xvals)>0.5,1,0))
performance.nottune<-confusionMatrix(predictions2,factor(yvals))
performance.nottune
write.csv(data.frame(performance.nottune$overall),"超参数调整前模型.csv")
#feature importance
mat<-xgb.importance(colnames(data),fit)
xgb.plot.importance(mat)




#use MLR for hyperparameters tuning
library(mlr)
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())
getParamSet("classif.xgboost")#查看所有参数

traintask <- makeClassifTask (data = Train,target = "Survived")
testtask <- makeClassifTask (data = Test,target = "Survived")
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error")
#lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

params <- makeParamSet( makeDiscreteParam("booster",
                                          values = c("gbtree")),
                        makeIntegerParam("max_depth",lower = 3L,upper = 100L),
                        makeIntegerParam("min_child_weight",lower = 0L,upper = 20L),
                        makeIntegerParam("subsample",lower = 0,upper = 1), 
                        makeNumericParam("colsample_bylevel",lower = 0,upper = 1),
                        makeNumericParam("lambda",lower = 0,upper = 5),
                        makeNumericParam("gamma",lower = 0,upper = 5),
                        makeNumericParam("eta",lower = 0,upper = 1),
                        makeIntegerParam("nrounds",lower = 1,upper = 100),
                        makeIntegerParam("alpha",lower = 0,upper = 10),
                        makeNumericParam("max_delta_step",lower = 0,upper = 10),
                        makeDiscreteParam("predictor",values=c("cpu_predictor")),
                        makeDiscreteParam("grow_policy",values=c("lossguide")))
tune<-expand.grid(
  nrounds = c(50, 100, 250, 500), # number of boosting iterations
  eta = c(0.01, 0.1, 1),  # learning rate, low value means model is more robust to overfitting
  lambda = c(0.1, 0.5, 1), # L2 Regularization (Ridge Regression)
  alpha =  c(0.1, 0.5, 1) # L1 Regularization (Lasso Regression)
) 

rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 100)




mytune <- tuneParams(learner = lrn, 
                     task = traintask, 
                     resampling = rdesc, 
                     measures = acc, par.set = params, control = ctrl, show.info = T)

#params <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "error",
                       #eta=0.00803, gamma=0.408, max_depth=19, min_child_weight=9.31, 
                       #subsample=0.249, colsample_bytree=0.521,lambda=1.04)
mytune$y
mytune$x


lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)
xgmodel <- train(learner = lrn_tune,task = traintask)
xgpred <- predict(xgmodel,testtask)
xgpred$data$response
performance.tune<-confusionMatrix(xgpred$data$response,xgpred$data$truth)
performance.tune
write.csv(performance.tune$overall,"模型调整后.csv")


#基于已选择的最好算法，find threshold
pre<-cbind(xgpred$data$prob.1,Test$Survived)
x<-prediction(pre[,1],pre[,2])
pref<-ROCR::performance(x,"tpr","fpr")
plot(pref)
Auc<-auc(pre[,2],pre[,1])
cutoffs <- data.frame(cut=pref@alpha.values[[1]], fpr=pref@x.values[[1]], 
                      tpr=pref@y.values[[1]])

#use function to calculate best threshold
#根据距离点(0,1)最近的点作为阈值
threshold.four<-cutoffs[which.min((((1-cutoffs[,3])*(1-cutoffs[,3])+(cutoffs[,2])*(cutoffs[,2]))^(1/2))),1]

predictions2<-as.factor(ifelse(xgpred$data$prob.1>threshold.four,1,0))
cufs<-confusionMatrix(predictions2,factor(Test$Survived))
cufs
plotdata<-xgpred$data[,c(2,3,4)]
ggplot(plotdata,aes(y=`prob.1`,x=`prob.0`,col=factor(truth)))+geom_point(alpha=0.5,size=2)+geom_vline(xintercept = threshold.four)#在图中显示





#use in caret







#答案是有62%的0