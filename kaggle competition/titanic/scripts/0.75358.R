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
cl <- makeCluster(detectCores()-4)
registerDoParallel(cl)
set.seed(1234)

#工作目录
setwd("C:\\Users\\nigel\\Desktop\\titanic\\data")

#import data
ig.data<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\data\\train.csv',header=T)
#性别转换，0为女性，1为男性
ig.data$Sex[which(ig.data$Sex=="male")]<-1
ig.data$Sex[which(ig.data$Sex=="female")]<-0
ig.data$Sex<-as.numeric(ig.data$Sex)
#EDA
#0是死了
str(ig.data)

#缺失值
ig.data[which(is.na(ig.data),arr.ind = T)]
table(is.na(ig.data))
my_plots <- lapply(names(ig.data), function(var_x){
  p <- 
    ggplot(ig.data) +
    aes_string(var_x)
  if(is.numeric(ig.data[[var_x]])) {
    p <- p + geom_density()
  } else {
    p <- p + geom_bar()
  } 
})

pdf(file = "distribution.pdf",
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
my_plots
dev.off()

#存活情况
ggplot(ig.data,aes(x=factor(Survived),fill=factor(Survived)))+geom_bar()
#差不是2比1，应该不会出现分布不均衡的情况

#存货情况和船票等级
#一等票中，存活人数多且比死亡人数多；二等票，五五开；三等票，死的人更多
#看来存活情况和船票等级相关
ggplot(ig.data,aes(x=factor(Pclass),fill=factor(Survived)))+geom_bar()


#存货情况和性别
#男性死亡远多于女性，女性存活的比死亡的多
#问什么？猜测和船票等级有关
ggplot(ig.data,aes(x=factor(Sex),fill=factor(Survived)))+geom_bar()


#船票等级和性别
#男性有很多人是三等票，可能是这个原因
ggplot(ig.data,aes(x=factor(Sex),fill=factor(Pclass)))+geom_bar()


#有多少个兄弟姐妹和配偶登船，可以考虑分箱，或许可以用聚类进行分箱？或者分成有还是没有


#相关性
library(corrplot)
cortable<-data.frame(na.omit(ig.data)%>%select(Survived,Pclass,Sex,Age,SibSp,Parch,Fare)%>%cor())
write.csv(cortable,"cor.csv")
na.omit(ig.data)%>%select(Survived,Pclass,Sex,Age,SibSp,Parch,Fare)%>%cor()%>%corrplot()



#---------data---------#
data<-na.omit(ig.data)%>%select(Survived,Pclass,Sex,Age,SibSp,Parch,Fare)
data<-ig.data%>%select(Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Male,Female)
#------------Feature select---------------#
#information gain
library(FSelectorRcpp)
ig_res =information_gain(Survived ~ .,data)


#rfe
control<-rfeControl(functions = rfFuncs,method="repeatedcv",repeats=5,number=10,allowParallel=T)

rfe.result<-rfe(data[,-1],factor(data[,1]),c(1:ncol(data[,-1])),rfeControl=control)#还是用这个
rfe.result#Sex, Pclass, Age, Fare
feature<-predictors(rfe.result)
#sbf
control.sbf<-sbfControl(functions=rfSBF,method="cv",repeats=5,number=10,allowParallel=T)
sbf.feature<-sbf(data[,-1],factor(data[,1]),sbfControl = control.sbf)
sbf.feature#Pclass, Sex, Age, Parch, Fare


#---------split data------------#
data<-data%>%select(feature,Survived)
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

x<-model.matrix(Survived~.,Train)
y<-model.frame(Survived~.,Train)[,"Survived"]

xvals<-model.matrix(Survived~.,Test)
yvals<-model.frame(Survived~.,Test)[,"Survived"]

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "logloss",
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = x,label=y,
                 nrounds = 200, nfold = 10, showsd = T, 
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
ggplot()+geom_line(graph.data,mapping=aes(x=iter,y=train_logloss_mean),color='blue')+
  geom_line(graph.data,mapping=aes(x=iter,y=test_logloss_mean),color='red')


numrounds<-min(which(xgbcv$evaluation_log$test_logloss_mean == 
                       min(xgbcv$evaluation_log$test_logloss_mean)))

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
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

params <- makeParamSet( makeDiscreteParam("booster",
                                          values = c("gbtree","gblinear")),
                        makeIntegerParam("max_depth",lower = 3L,upper = 15L),
                        makeNumericParam("min_child_weight",lower = 0L,upper = 10L),
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 10L)




mytune <- tuneParams(learner = lrn, 
                     task = traintask, 
                     resampling = rdesc, 
                     measures = acc, par.set = params, control = ctrl, show.info = T)
mytune$y


lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)
xgmodel <- train(learner = lrn_tune,task = traintask)
xgpred <- predict(xgmodel,testtask)
xgpred$data$response
performance.tune<-confusionMatrix(xgpred$data$response,xgpred$data$truth)
performance.tune
write.csv(performance.tune$overall,"模型调整后.csv")














#submit

#import data
ctest<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\data\\test.csv',header=T)
tmp<-hot.enc.sex(ctest)
ctest<-tmp
ctest<-data.frame(Survived=c(0))%>%cbind(ctest)
#性别转换，0为女性，1为男性
ctest$Sex[which(ctest$Sex=="male")]<-1
ctest$Sex[which(ctest$Sex=="female")]<-0
ctest$Sex<-as.numeric(ctest$Sex)
#ctask <- na.omit(ctest)%>%select(Pclass,Sex,Age,SibSp,Parch,Fare,Survived)
ctask<-ctest%>%select(Pclass,Sex,Age,SibSp,Parch,Fare,Survived,Male,Female)
ctask$Survived<-as.factor(ctask$Survived)

ctest1<-makeClassifTask(data=ctask,target = "Survived")
pred<-predict(xgmodel,ctest1)
pred$data$response

cbind(ctest,pred$data$response)

result<-cbind(ctest,pred$data$response)
result<-result[,c(4,15)]
write.csv(result,"result.csv")

#答案是有62%的0