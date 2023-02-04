#submit

#import data
ctest<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\data\\test.csv',header=T)
x<-ctest


ctest<-ig.data[892:1309,]
ctest$Sex[which(ctest$Sex=="male")]<-1
ctest$Sex[which(ctest$Sex=="female")]<-0
ctest$Sex<-as.numeric(ctest$Sex)
pred<-predict(glm_ensemble ,ctest)
#pred<-predict(svm ,ctest)
result<-cbind(x,pred)
result[which(result$Sex=="male"&result$pred=="Save"),]
nrow(result[which(result$Sex=="male"&result$pred=="Save"),])
nrow(result[which(result$pred=="Die"),])/nrow(result)
result<-result[,c(1,12)]
result$pred<-apply(result,1,function(x){
  if(x[2]=="Die"){
    x[2]=0
  }
  else{
    x[2]=1
  }
}
)
write.csv(result,"result0.8254.csv")


ans<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\data\\submission.csv',header=T)
confusionMatrix(factor(result$pred),factor(ans$Survived))

write.csv(pred,"pred.csv")

#ctest<-ctest%>%select(Sex,Pclass,Fare,Age,Parch)
#ctest$Embarked[which(ctest$Embarked=='S')]<-0
#ctest$Embarked[which(ctest$Embarked=='C')]<-1
#ctest$Embarked[which(ctest$Embarked=='Q')]<-2
ctest$Sex[which(ctest$Sex=="male")]<-1
ctest$Sex[which(ctest$Sex=="female")]<-0
ctest$Sex<-as.numeric(ctest$Sex)
ctest$Embarked<-as.numeric(ctest$Embarked)

7.7500

ctest<-data.frame(Title=c(NA))%>%cbind(ctest)
for (i in c(1:nrow(ctest))){
  #Mr
  if(grepl("Mr",ctest$Name[i]))
  {
    ctest$Title[i]<-0
  }
  
  #Miss
  else if(grepl("Ms",ctest$Name[i]) || grepl("Mile",ctest$Name[i]) ||grepl("Miss",ctest$Name[i]))
  {
    ctest$Title[i]<-1
  }
  else if(grepl("Master",ctest$Name[i]))
  {
    ctest$Title[i]<-2
  }
  #Mrs
  else if(grepl("Mme",ctest$Name[i]) || grepl("Mrs",ctest$Name[i]))
  {
    ctest$Title[i]<-4
  }
  else
  {
    ctest$Title[i]<-3
  }
  
}

ggplot()+geom_boxplot(ctest,mapping=aes(x=Title,y=Age,fill=factor(Title)))
ctest$Age[which(is.na(ctest$Age),arr.ind=T)]<-0
Age.mean<-na.omit(ctest)%>%group_by(Title)%>%summarise(mean_Age=mean(Age))  

for (i in c(1:nrow(ctest))){
  #Mr
  if(ctest$Title[i]==0 && is.na(ctest$Age[i]))
  {
    ctest$Age[i]<-33
  }
  
  #Miss
  else if(ctest$Title[i]==1 && is.na(ctest$Age[i]))
  {
    ctest$Age[i]<-22
  }
  else if(ctest$Title[i]==2 && is.na(ctest$Age[i]))
  {
    ctest$Age[i]<-5
  }
  #Mrs
  else if(ctest$Title[i]==3 && is.na(ctest$Age[i]))
  {
    ctest$Age[i]<-44
  }
  else if(ctest$Title[i]==4 && is.na(ctest$Age[i]))
  {
    ctest$Age[i]<-24   
  }
}

ctest$Age[which(ctest$Age<1)]<-1
ctest<-ctest[,-1]

ctest<-ctest%>%select(Sex,Pclass,Age,SibSp,Parch,Fare,Embarked)
#性别转换，0为女性，1为男性
#性别转换
tmp<-hot.enc.sex(ctest)
ctest<-tmp
ctest<-ctest[,-3]
#港口转换
#s=0,c=1,q=2
tmp<-hot.enc.Embarked2(ctest)
ctest<-tmp
ctest<-ctest[,-11]

ctest<-data.frame(Survived=c(0))%>%cbind(ctest)

#ctask <- na.omit(ctest)%>%select(Pclass,Sex,Age,SibSp,Parch,Fare,Survived)
#ctask<-ctest%>%select(Male,Female,Survived,Sex,Pclass,Age,SibSp,Parch,Fare)
#names(ctask)<-c("Male", "Female", "Survived","Sex1","Pclass","Age","SibSp","Parch","Fare")
#ctask$Survived<-as.factor(ctask$Survived)
ctest$Survived<-as.factor(ctest$Survived)
ctest[153,11]<-9.6875

ctest<-ctest%>%select(Sex,FareAdj,FamilySize)
ctest$Survived<-as.factor(ctest$Survived)
ctest1<-makeClassifTask(data=ctest,target = "Survived")
#pred<-predict(xgmodel ,ctest1)

predictions3<-as.factor(ifelse(pred$data$prob.1>0.5,1,0))
#check<-model.matrix(Survived~.,ctask)
#pred<-predict(fit,check,tpe="res")

pred$data$response

cbind(ctest,pred$data$response)

result<-cbind(x,predictions3)
result<-result[,c(1,12)]
write.csv(result,"result.csv")




######################################################################
ctest$TicketFreq <- ave(1:418,ctest$Ticket,FUN=length)
ctest$FareAdj <- ctest$Fare / ctest$TicketFreq+ctest$Age/70
ctest$FamilySize <- ctest$SibSp + ctest$Parch + 1


#
data$Embarked[which(data$Embarked=='S')]<-0
data$Embarked[which(data$Embarked=='C')]<-1
data$Embarked[which(data$Embarked=='Q')]<-2
data$Sex[which(data$Sex=="male")]<-1
data$Sex[which(data$Sex=="female")]<-0
data$Sex<-as.numeric(data$Sex)
data$Embarked<-as.numeric(data$Embarked)
