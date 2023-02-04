ig.data$Surname = substring(ig.data$Name,0,regexpr(',',ig.data$Name)-1)
ig.data$GroupId = paste( ig.data$Surname, ig.data$Pclass, sub('.$','X',ig.data$Ticket), ig.data$Fare,ig.data$Embarked, sep='-')
ig.data[c(195,1067,59,473,1142),c('Name','GroupId')]

ig.data$Title <- 'man'
ig.data$Title[ig.data$Sex=='female'] <- 'woman'
ig.data$Title[grep('Master',ig.data$Name)] <- 'boy'

ig.data$Color <- ig.data$Survived
ig.data$GroupId[ig.data$Title=='man'] <- 'noGroup'
ig.data$GroupFreq <- ave(1:1309,ig.data$GroupId,FUN=length)
ig.data$GroupId[ig.data$GroupFreq<=1] <- 'noGroup'
ig.data$TicketId = paste( ig.data$Pclass,sub('.$','X',ig.data$Ticket),ig.data$Fare,ig.data$Embarked,sep='-')
count = 0
# add nannies and relatives to groups
for (i in which(ig.data$Title!='man' & ig.data$GroupId=='noGroup')){
  ig.data$GroupId[i] = ig.data$GroupId[ig.data$TicketId==ig.data$TicketId[i]][1]
  if (ig.data$GroupId[i]!='noGroup') {
    # color variable is used in plots below
    if (is.na(ig.data$Survived[i])) ig.data$Color[i] = 5
    else if (ig.data$Survived[i] == 0) ig.data$Color[i] = -1
    else if (ig.data$Survived[i] == 1) ig.data$Color[i] = 2
    count = count + 1
  }
}





savedata<-ig.data

ig.data<-read.csv('C:\\Users\\nigel\\Desktop\\titanic\\data\\ig.data.csv',header=T)


ctest<-ig.data[892:1309,]

ctest$pred<-pred
ig.data$pred<-ig.data$Survived
ctest$pred<-apply(ctest,1,function(x){
  if(x[27]=="Die"){
    x[27]=0
  }
  else{
    x[27]=1
  }
}
)

tmp<-rbind(ctest,ig.data[1:891,])


for (i in c(1:nrow(ctest))){
  id<-ctest$GroupId[i]
  tick<-ctest$TicketId[i]
  if(id!="noGroup"){
    if(nrow(filter(tmp,GroupId==id,Survived==0))==0 &nrow(filter(tmp,GroupId==id,Survived==1))>=1)
    {
      print("Save")
      print(i)
      ctest$pred[i]<-1
    }
      
    else if(nrow(filter(tmp,GroupId==id,Survived==1))==0 & nrow(filter(tmp,GroupId==id,Survived==0))>=1){
      print("die")
      print(i)
      ctest$pred[i]<-0
    }
  }

}
else if(id=="noGroup" & ctest$Sex[i]=="male"){
  if(nrow(filter(tmp,TicketId==tick,Survived==0,Title=="woman"))==0 &nrow(filter(tmp,TicketId==tick,Survived==0,Title=="boy"))==0&(nrow(filter(tmp,TicketId==tick,Survived==0))+nrow(filter(tmp,TicketId==tick,Survived==1)))>1){
    ctest$pred[i]<-1
  }
  else if(nrow(filter(tmp,TicketId==tick,Survived==1,Title=="woman"))==0 &nrow(filter(tmp,TicketId==tick,Survived==1,Title=="boy"))==0&(nrow(filter(tmp,TicketId==tick,Survived==0))+nrow(filter(tmp,TicketId==tick,Survived==1)))>1){
    ctest$pred[i]<-0
  }
}   

2-24873X-29-S

result<-data.frame(cbind(x$PassengerId,tmp$pred[892:1309]))
result<-ctest%>%select(pred,PassengerId)


result$pred<-apply(result,1,function(x){
  if(x[2]=="Die"){
    x[2]=0
  }
  else{
    x[2]=1
  }
}
)







pp<-ig.data
#全部都死、存活
for (i in c(1:nrow(ig.data))){
  id<-ig.data$GroupId[i]
  tick<-ig.data$TicketId[i]
  if(nrow(filter(ig.data,TicketId==tick,Survived==0))==0&(nrow(filter(ig.data,TicketId==tick,Survived==0))+nrow(filter(ig.data,TicketId==tick,Survived==1)))>1)
  {
    ig.data$FS[i]<-1
  }
  else if(nrow(filter(ig.data,TicketId==tick,Survived==1))==0&(nrow(filter(ig.data,TicketId==tick,Survived==0))+nrow(filter(ig.data,TicketId==tick,Survived==1)))>1)
  {
    ig.data$FS[i]<-0
  }
  else
    ig.data$FS[i]<-0.5
}







for (i in c(1:nrow(ig.data))){
  id<-ig.data$GroupId[i]
  tick<-ig.data$TicketId[i]
  if(id!="noGroup"){
    if(nrow(filter(ig.data,TicketId==tick,Survived==0,GroupId!="noGroup"))==0 &nrow(filter(ig.data,GroupId==id,Survived==1,GroupId!="noGroup"))>=1)
    {
      ig.data$FS[i]<-1
    }
    
    else if(nrow(filter(ig.data,TicketId==tick,Survived==1,GroupId!="noGroup"))==0 &nrow(filter(ig.data,GroupId==id,Survived==0,GroupId!="noGroup"))>=1)
    {
      ig.data$FS[i]<-0
    }
  }
  else
    ig.data$FS[i]<-0.5
}

