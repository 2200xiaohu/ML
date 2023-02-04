library(caret)


#串行
make.names("Survived")
x<-data1.frame(error$Survived)
names(x)<-c("X0")
error<-cbind(x,error)
error<-error[,-7]




Train1<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\Trainbest.csv',header=T)
Test1<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\Testbest.csv',header=T)
#SVM

Train1<-Train
#Train1$Survived<-as.numeric(Train1$Survived)
Train1$Survived[which(Train1$Survived==0,arr.ind=T)]<-"Die"
Train1$Survived[which(Train1$Survived==1,arr.ind=T)]<-"Save"
Train1$Survived<-as.factor(Train1$Survived)

Test1<-Test
#Test1$Survived<-as.numeric(Test1$Survived)
Test1$Survived[which(Test1$Survived==0,arr.ind=T)]<-"Die"
Test1$Survived[which(Test1$Survived==1,arr.ind=T)]<-"Save"
Test1$Survived<-as.factor(Test1$Survived)


#svm
train.control<-trainControl(method="cv",number=10,summaryFunction = multiClassSummary,
                            classProbs=T,savePredictions = "final",
                            index=createResample(Train1$Survived),verboseIter = T)

svm <- caret::train(Survived ~.,data =Train1 , 
             method = "svmPoly", trControl = train.control,  
             preProcess = c("center","scale"), tuneGrid = expand.grid(C = seq(0, 10, length = 10),degree=seq(0,5,length=11),scale=seq(0,1,length=11)))
svm <- caret::train(Survived ~.,data =Train1 , 
                    method = "svmLinear", trControl = train.control,  
                    preProcess = c("center","scale"), tuneGrid = expand.grid(C = seq(0, 10, length = 1000)))
svm<-caret::train(Survived~.,data=Train1,method="svmLinear3",preProcess = c("center","scale"),
           trControl = train.control,tuneGrid=expand.grid(cost=seq(0.001,1,length.out = 10), Loss=c("L1","L2")))


#C = seq(0, 5, length = 50),sigma=seq(0, 1, length = 50)

svm
plot(svm)
svm$bestTune#0.03003003
svm <- caret::train(Survived ~., data =Train1 , 
             method = "svmLinear", trControl = train.control,  
             preProcess = c("center","scale"),tuneGrid=data.frame(svm$bestTune))
svm <- caret::train(Survived ~., data =Train1 , 
                    method = "svmLinear", trControl = train.control,  
                    preProcess = c("center","scale"),tuneGrid=data.frame(C=1))
svm$finalModel
plot(svm)
varImp(svm,scale=F)
#predict
pre.svm<-predict(svm,Test1)
confusionMatrix(pre.svm,Test1$Survived)#0.8127   

predictions_plot<-predict(svm,Test1,type="prob")
plotdata1<-cbind(predictions_plot,class=Test1$Survived)
ggplot(plotdata1,aes(y=`Die`,x=`Save`,col=factor(class)))+geom_point(alpha=0.5)
ggplot(plotdata1,aes(y=`Die`,x=`Save`,col=factor(class)))+geom_point(alpha=0.5,size=2)+geom_jitter(width=0.25,height=0.25)#发现有几个点即使设置阈值也难以区分的

#glm
glm <- caret::train(Survived ~.,data =Train1 , 
                    method = "glm", trControl = train.control,  
                    preProcess = c("center","scale"))
pre.glm<-predict(glm,Test1)
confusionMatrix(pre.glm,Test1$Survived)#0.8202 
#xgboost
xgparam<-expand.grid(nrounds=100,
                     subsample = seq(0.5, 1, length = 3), 
                     colsample_bytree = seq(0.5, 0.9, length = 3),
                     max_depth = c(6:15),
                     min_child_weight = c(1:3),
                     eta = seq(0, 1, length = 6),
                     gamma = seq(0, 5, length = 6))


#,alpha = seq(0, 3, length = 5)
#find hyperparameters
xg <- caret::train(Survived ~.,data =Train1 , metric="Accuracy",
                    method = "xgbTree", trControl = train.control,  
                     tuneGrid = xgparam,early_stopping_rounds = 15,verbose=T)

xg <- caret::train(x=input_x,y=input_y,
                   method = "xgbTree", trControl = train.control, tuneGrid = grid_default,
                   early_stopping_rounds = 15,metric="Accuracy",verbose=T)
input_x <- as.matrix(select(Train1, -Survived))
input_y <- Train1$Survived
grid_default <- expand.grid(
  nrounds = 40,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 0.5
)
#find best nround

#################
t1=proc.time()
LogitBoost_Adapt = caret::train(Survived ~ ., data = Train1,method = "LogitBoost",trControl = train.control,
                         preProc = c("center", "scale"), metric = "Accuracy",tuneLength = 540)
t2=proc.time()
(t2-t1)/60 
confusionMatrix(predict(LogitBoost_Adapt,Test1),Test1$Survived)






#knn
knnFit <- caret::train(Survived ~.,data =Train1 , 
                              method = "knn", trControl = train.control,  
                              preProcess = c("center","scale"), tuneGrid = data.frame(k = seq(1,200,by = 1)))
knnFit <- caret::train(Survived ~.,data =Train1 , 
                       method = "knn", trControl = train.control,  
                       preProcess = c("center","scale"),tuneGrid = data.frame(k = 2))
knnFit$bestTune
pre.knn<-predict(knnFit,Test1)
confusionMatrix(pre.knn,Test1$Survived)

#rf
rf <- caret::train(Survived ~.,data =Train1 , 
                       method = "rf", trControl = train.control,  
                       preProcess = c("center","scale"))
rf$bestTune
pre.rf<-predict(rf,Test1)
confusionMatrix(pre.rf,Test1$Survived)
#---------------------集成-------------------#
library(caretEnsemble)
bestparam
currentSubsampleRate, currentColsampleRate, currentDepth, 
currentMinChild,currentEta,currentgamma,numrounds

grid_default <- expand.grid(
  nrounds = numrounds,
  max_depth = getBestPars(opt_obj)[3],
  eta = getBestPars(opt_obj)[1],
  gamma = getBestPars(opt_obj)[2],
  min_child_weight = getBestPars(opt_obj)[4],
  subsample = getBestPars(opt_obj)[5],
  colsample_bytree=0.5
)
grid_default <- expand.grid(
  nrounds = 20,
  max_depth = 32,
  eta = 0.4259741,
  gamma = 0,
  min_child_weight = 4.413045,
  subsample = 1,
  colsample_bytree=0.81
)
grid_default <- expand.grid(
  nrounds = as.numeric(numrounds),
  max_depth = as.numeric(pa[1,5]),
  eta = as.numeric(pa[1,3]),
  gamma = as.numeric(pa[1,4]),
  min_child_weight = as.numeric(pa[1,6]),
  subsample = as.numeric(pa[1,7]),
  colsample_bytree=0.5
)
grid_default <- expand.grid(
  nrounds = 42,
  max_depth = 2,
  eta = 0.6122684,
  gamma = 0,
  min_child_weight = 1.456841,
  subsample = 1,
  colsample_bytree=0.5
)

model_list <- caretList(
  Survived~., data=Train1,
  trControl=train.control,
  tuneList=list(xgbTree=caretModelSpec(method="xgbTree", tuneGrid=grid_default,preProcess = c("center","scale")),
                svm=caretModelSpec(method="svmLinear",  preProcess = c("center","scale"),tuneGrid=data.frame(svm$bestTune)),
                knn=caretModelSpec(method="knn",  preProcess = c("center","scale"),tuneGrid=data.frame(knnFit$bestTune))))
                #rf=caretModelSpec(method="rf", tuneGrid=data.frame(rf$bestTune))))
#
#Make a greedy ensemble - currently can only use RMSE
glm_ensemble <- caretStack(
  model_list,
  method="glm",
  metric="Accuracy",
  trControl=trainControl(method='cv', 
                         number=10, 
                         savePredictions = "final", 
                         classProbs=TRUE, 
                         index=createResample(Train1$Survived)))

glm_ensemble$models
glm_ensemble$ens_model
glm_ensemble$ens_model$finalModel

confusionMatrix(predict(glm_ensemble,Test1),Test1$Survived)#0.8315          

predictions3<-as.factor(ifelse(predict(glm_ensemble,Test1,type="prob")>0.5,'Die',"Save"))
confusionMatrix(predictions3,Test1$Survived)#0.8652 

p<-predict(glm_ensemble,Test1,type="prob")
plotdata<-cbind(p,Test1)
plotdata<-plotdata%>%mutate(die=1-p)
ggplot(plotdata,aes(y=p,x=die,col=factor(Survived)))+geom_point(alpha=0.5,size=2)+geom_vline(xintercept = 0.5)+geom_jitter(width=0.25,height=0.25)#在图中显示
ggplot(plotdata,aes(y=p,x=die,col=factor(Survived)))+geom_point(alpha=0.5,size=2)

#保存错误
u<-ig.data[1:891,]

testerror<-u[ -trainIndex,]%>%cbind(pre=as.character(predictions3))
testerror$pre[which(testerror$pre=="Die")]<-0
testerror$pre[which(testerror$pre=="Save")]<-1
testerror<-testerror[which(testerror$Survived!=testerror$pre),]
write.csv(testerror,"testerror.csv")










#---------人工投票-------------#
vote<-function(xgboost,model2,test,i){
  #结果表格
  x<-data.frame(threshold1=c(NA),threshold2=c(NA),performance=c(NA))
  testtask <- makeClassifTask (data = test,target = "Survived")
  #
  prediction1 <- predict(xgboost,testtask)
  per1<-prediction1$data$prob.1
  prediction2<-predict(model2,test,type="prob")
  per2<-prediction2[,2]
  tmp<-data.frame(final=c(NA))
  result<-cbind(per1,per2,tmp)
  row<-1
  while(i<=1){
    
    #输出概率值
    for(j in c(1:nrow(result)))
    {
      result[j,3]<-result[j,1]*i+result[j,2]*(1-i)
      print(result[j,1]*i+result[j,2]*(1-i))
    }
    #检验结果
    finalpre<-as.factor(ifelse(result[,3]>0.5,1,0))
    h<-confusionMatrix(finalpre,test$Survived)
    print(h)
    x[row,3]<-h$overall[1]
    x[row,1]<-i
    x[row,2]<-1-i
    i=i+0.1
    row=row+1
  }
  
  return(x)
}
#xgboost svm
re<-vote(xgmodel,svm,Test,0)
get<-predictontest(xgmodel,svm,0.8,0.2,Test)
#xgboost glm
re<-vote(xgmodel,glm,Test,0)
get<-predictontest(xgmodel,svm,0.7,0.3,Test)
#基于已选择的最好算法，find threshold
pre<-cbind(get$final,Test$Survived)
x<-prediction(pre[,1],pre[,2])
pref<-ROCR::performance(x,"tpr","fpr")
plot(pref)
Auc<-auc(pre[,2],pre[,1])
cutoffs <- data.frame(cut=pref@alpha.values[[1]], fpr=pref@x.values[[1]], 
                      tpr=pref@y.values[[1]])

#use function to calculate best threshold
#根据距离点(0,1)最近的点作为阈值
threshold.four<-cutoffs[which.min((((1-cutoffs[,3])*(1-cutoffs[,3])+(cutoffs[,2])*(cutoffs[,2]))^(1/2))),1]

predictions2<-as.factor(ifelse(get$final>0.5,1,0))
cufs<-confusionMatrix(predictions2,factor(Test$Survived))
cufs
plotdata<-get%>%mutate(Save=1-final)%>%cbind(xgpred$data$truth)
ggplot(plotdata,aes(y=`final`,x=`Save`,col=factor(xgpred$data$truth)))+geom_point(alpha=0.5,size=2)+geom_vline(xintercept = threshold.four)
ggplot(plotdata,aes(y=`final`,x=`Save`,col=factor(xgpred$data$truth)))+geom_point(alpha=0.5,size=2)+geom_vline(xintercept = threshold.four)+geom_jitter(width=0.1,height=0.1)#在图中显示



predictontest<-function(model1,model2,threshold1,threshold2,data){
  #转换给xgboost的数据
  testtask<-makeClassifTask(data=data,target = "Survived")
  #predict
  prediction1 <- predict(model1,testtask)
  per1<-prediction1$data$prob.1
  prediction2<-predict(model2,data,type="prob")
  per2<-prediction2[,2]
  tmp<-data.frame(final=c(NA))
  result<-cbind(per1,per2,tmp)
  row<-1
  #输出概率值
  for(j in c(1:nrow(result)))
  {
    if(result[j,1]-result[j,2]>=0.3)
    {
      result[j,3]<-result[j,1]
    }
    else if(result[j,2]-result[j,1]>=0.3)
    {
      result[j,3]<-result[j,2]
    }
    else
      {
        result[j,3]<-result[j,1]*threshold1+result[j,2]*threshold2
      }
  }
  return(result)
}
get1<-predictontest(xgmodel,svm,0.6,0.4,ctest)
get1<-predictontest(xgmodel,glm,0.7,0.3,ctest)



predictions3<-as.factor(ifelse(get1$final>threshold.four,1,0))
result<-cbind(x,predictions3)
result<-result[,c(1,12)]
write.csv(result,"result.csv")



