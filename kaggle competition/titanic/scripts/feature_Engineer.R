#import data
datatrain<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\data\\train.csv',header=T)
ctest<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\data\\test.csv',header=T)
ctest<-data.frame(Survived=c(NA))%>%cbind(ctest)
ig.data<-rbind(datatrain,ctest)
x<-ctest


#Title
ig.data<-data.frame(Title=c(NA))%>%cbind(ig.data)
for (i in c(1:nrow(ig.data))){
  #Mr
  if(grepl("Mr",ig.data$Name[i]))
  {
    ig.data$Title[i]<-0
  }
  
  #Miss
  else if(grepl("Ms",ig.data$Name[i]) || grepl("Mile",ig.data$Name[i]) ||grepl("Miss",ig.data$Name[i]))
  {
    ig.data$Title[i]<-1
  }
  else if(grepl("Master",ig.data$Name[i]))
  {
    ig.data$Title[i]<-2
  }
  #Mrs
  else if(grepl("Mme",ig.data$Name[i]) || grepl("Mrs",ig.data$Name[i]))
  {
    ig.data$Title[i]<-4
  }
  else
  {
    ig.data$Title[i]<-3
  }
  
}

#input missing data
#Age

library(rpart)
fit <- rpart(Age ~ Title + Pclass + SibSp + Parch,data=ig.data)
ig.data$Age[is.na(ig.data$Age)] <- predict(fit,newdata=ig.data[is.na(ig.data$Age),])
fit <- rpart(Fare ~ Title + Pclass + Embarked + Sex + Age,data=ig.data)
ig.data$Fare[is.na(ig.data$Fare)] <- predict(fit,newdata=ig.data[is.na(ig.data$Fare),])


#engineer Feature
ig.data$TicketFreq <- ave(1:1309,ig.data$Ticket,FUN=length)
#ig.data$FareAdj <- ig.data$Fare / ig.data$TicketFreq+ig.data$Age/70
ig.data$FareAdj <- ig.data$Fare / ig.data$TicketFreq
ig.data$FamilySize <- ig.data$SibSp + ig.data$Parch + 1
ig.data$FP<-100*(4-ig.data$Pclass)+(ig.data$FareAdj+1)

ig.data<-data.frame(Familysur=c(NA))%>%cbind(ig.data)
for(i in c(1:nrow(ig.data))){
  if(as.numeric(ig.data[i,"TicketFreq"]>=1))
  {
    tmp1<-ig.data%>%filter(Ticket==ig.data[i,"Ticket"],Survived==1)
    tmp2<-ig.data%>%filter(Ticket==ig.data[i,"Ticket"],Survived==0)
    {
      if(nrow(tmp1)>nrow(tmp2))
      {
        ig.data[i,"Familysur"]<-1#存活能力好
      }
      #if(nrow(tmp1)==nrow(tmp2)){
       # x[i,"Familysur"]<-2#存活能力一般
      #}
      else{
        ig.data[i,"Familysur"]<-0#存活能力差
      }
    }
  }
}

ig.data[which(is.na(ig.data),arr.ind=T),]

#妇孺
#ig.data$wc[which((ig.data$Sex=="female"| ig.data$Age<=15) &ig.data$Pclass!=3)]<-1
ig.data$wc[which(ig.data$Age<=15)]<-1
ig.data$wc[which(ig.data$Sex=="male" & ig.data$Age>15)]<-0
ig.data$wc[which(ig.data$Sex=="female" & ig.data$Age>15)]<-2
#ig.data$wc[which((ig.data$Sex=="female"|ig.data$Age<=15)& ig.data$Pclass==3)]<-2
#妇孺加年龄
ig.data$FA<-ig.data$Age*((-1)**(ig.data$wc))


#标记nogroup
ig.data$group<-1
#没有group
ig.data$group[which(ig.data$GroupId=="noGroup")]<-0
ig.data$group[which(ig.data$GroupId=="noGroup"&ig.data$Title=="woman")]<-2
#
data<-ig.data[1:891,]%>%select(FamilySize,Survived,FareAdj,Sex,FP)
data<-ig.data[1:891,]%>%select(FamilySize,Survived,wc,FP,FS)

#
data$Sex[which(data$Sex=="male")]<-1
data$Sex[which(data$Sex=="female")]<-0
data$Sex<-as.numeric(data$Sex)


trainIndex <- createDataPartition(data$Survived, p = .7, 
                                  list = FALSE, 
                                  times = 1)

Train <- data[ trainIndex,]
Test <- data[ -trainIndex,]
