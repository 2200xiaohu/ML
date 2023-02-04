library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(doParallel)
cl <- makeCluster(detectCores()-4)
registerDoParallel(cl)
set.seed(1234)
setwd("C:/Users/nigel/Desktop/kaggle competition/playground1/data")
data<-read.csv("C:/Users/nigel/Desktop/kaggle competition/playground1/data/train.csv")


#product和属性的关系
gather(data,iv,dv,attribute_0:measurement_17)%>%ggplot(aes(x=dv,fill=factor(product_code)))+geom_histogram(position ="fill")+
  scale_fill_brewer(palette="Set1")+facet_wrap( ~ iv, scales = "free", ncol = 3)
my_plots <- lapply(names(data), function(var_x){
  p <- 
    ggplot(data,aes(col=factor(product_code))) +
    aes_string(var_x)
  if(is.numeric(data[[var_x]])) {
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

#attribute_0的：B值集中在5，其他集中在7
#attribute_1:D,B集中在5，E在中间，其他在7
#attribute中每个pruduct不一样


#product为A
data_a<-filter(data,product_code=="A")
my_plots_a <- lapply(names(data_a), function(var_x){
  p <- 
    ggplot(data_a,aes(fill=factor(failure))) +
    aes_string(var_x)
  if(is.numeric(data_a[[var_x]])) {
    p <- p + geom_density()
  } else {
    p <- p + geom_bar()
  } 
})
pdf(file = "distribution_A.pdf",
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
my_plots_a
dev.off()

#product为B
data_b<-filter(data,product_code=="B")
my_plots_b <- lapply(names(data_b), function(var_x){
  p <- 
    ggplot(data_b,aes(fill=factor(failure))) +
    aes_string(var_x)
  if(is.numeric(data_b[[var_x]])) {
    p <- p + geom_density()
  } else {
    p <- p + geom_bar()
  } 
})
pdf(file = "distribution_B.pdf",
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
my_plots_b
dev.off()

#product为C
data_c<-filter(data,product_code=="C")
my_plots_c <- lapply(names(data_c), function(var_x){
  p <- 
    ggplot(data_c,aes(fill=factor(failure))) +
    aes_string(var_x)
  if(is.numeric(data_c[[var_x]])) {
    p <- p + geom_density()
  } else {
    p <- p + geom_bar()
  } 
})
pdf(file = "distribution_C.pdf",
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
my_plots_c
dev.off()

#product为D
data_d<-filter(data,product_code=="D")
my_plots_d <- lapply(names(data_d), function(var_x){
  p <- 
    ggplot(data_d,aes(fill=factor(failure))) +
    aes_string(var_x)
  if(is.numeric(data_d[[var_x]])) {
    p <- p + geom_density()
  } else {
    p <- p + geom_bar()
  } 
})
pdf(file = "distribution_D.pdf",
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
my_plots_d
dev.off()


#product为E
data_e<-filter(data,product_code=="E")
my_plots_e <- lapply(names(data_e), function(var_x){
  p <- 
    ggplot(data_e,aes(fill=factor(failure))) +
    aes_string(var_x)
  if(is.numeric(data_e[[var_x]])) {
    p <- p + geom_density()
  } else {
    p <- p + geom_bar()
  } 
})
pdf(file = "distribution_E.pdf",
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches
my_plots_e
dev.off()



#cor
library(corrplot)

pdf(file = "cor.pdf",
    width = 4, # The width of the plot in inches
    height = 4) # The height of the plot in inches

dev.off()
cortable<-data.frame(na.omit(data[,3:26])%>%cor())
write.csv(cortable,"cor.csv")

#查看缺失值情况
na.table<-data.frame(data[0,])#获取列名
for (i in 1:ncol(data)){
  na.table[1,i]=sum(is.na(data[,i]))  
}
na.table<-gather(na.table,key=variable,value=num)#转换为长数据
na.table$num<-as.integer(na.table$num)
na.table<-mutate(na.table,proportion=na.table$num/nrow(data))
write.csv(na.table,"缺失值占比.csv")




naomit<-na.omit(data)
naomit$product_code[which(naomit$product_code=='A')]<-0
naomit$product_code[which(naomit$product_code=='B')]<-1
naomit$product_code[which(naomit$product_code=='C')]<-2
naomit$product_code[which(naomit$product_code=='D')]<-3
naomit$product_code[which(naomit$product_code=='E')]<-4
naomit$product_code<-as.numeric(naomit$product_code)
write.csv(naomit[,2:26],"data_omit.csv",row.names=F)


control<-rfeControl(functions = nbFuncs,method="repeatedcv",repeats=5,number=10,allowParallel=T)

set.seed(1234)

rfe.result<-rfe(naomit[,c(-1,-26)],factor(naomit$failure),c(1:ncol(naomit[,c(-1,-26)])),rfeControl=control)#还是用这个
rfe.result

#select feature 
feature<-predictors(rfe.result)
feature
write.csv(naomit[,c(feature,'failure')],"data_omit.csv",row.names=F)



data1<-naomit[,c(feature,'failure')]
data1<-naomit[,c(2,3,4,26)]
trainIndex <- createDataPartition(data1$failure, p = .7, 
                                  list = FALSE, 
                                  times = 1)
Train <- data1[ trainIndex,]
Test <- data1[ -trainIndex,]
library(xgboost)
x<-model.matrix(failure~.,Train)
y<-model.frame(failure~.,Train)[,"failure"]

xvals<-model.matrix(failure~.,Test)
yvals<-model.frame(failure~.,Test)[,"failure"]
params <- list(booster = "gbtree", objective = "binary:logistic", eval_metric = "logloss",
               eta=0.05, gamma=0, max_depth=3, min_child_weight=1, 
               subsample=1, colsample_bytree=1)
fit <- xgboost(params = params,
               data = x,
               label = y,
               nrounds = 10)

pre<-predict(fit,xvals,type="res")
predictions2<-as.factor(ifelse(predict(fit,xvals)>0.5,1,0))
performance.nottune<-confusionMatrix(predictions2,factor(yvals))
performance.nottune
