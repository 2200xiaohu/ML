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



