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



#imput missing value for Age
#Mr = 0;Miss =1;Master =2;Other=3;Mrs=4
#Mr = Mr; Mrs="Mme", "Mrs"; Miss="Ms","Mile","Miss"; Master=Master
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

ggplot()+geom_boxplot(ig.data,mapping=aes(x=Title,y=Age,fill=factor(Title)))
ig.data$Age[which(is.na(ig.data$Age),arr.ind=T)]<-0
Age.mean<-na.omit(ig.data)%>%group_by(Title)%>%summarise(mean_Age=mean(Age))  

for (i in c(1:nrow(ig.data))){
  #Mr
  if(ig.data$Title[i]==0 && is.na(ig.data$Age[i]))
  {
    ig.data$Age[i]<-32.368090
  }
  
  #Miss
  else if(ig.data$Title[i]==1 && is.na(ig.data$Age[i]))
  {
  ig.data$Age[i]<-21.816327
  }
  else if(ig.data$Title[i]==2 && is.na(ig.data$Age[i]))
  {
  ig.data$Age[i]<-4.574167 
  }
  #Mrs
  else if(ig.data$Title[i]==3 && is.na(ig.data$Age[i]))
  {
  ig.data$Age[i]<-43.750000
  }
  else if(ig.data$Title[i]==4 && is.na(ig.data$Age[i]))
  {
    ig.data$Age[i]<-35.788991
  }
}

ig.data$Age[which(ig.data$Age<1)]<-1
ig.data<-ig.data[,-1]



#计算 价格
ig.data$TicketFreq <- ave(1:891,ig.data$Ticket,FUN=length)
ig.data$FareAdj <- ig.data$Fare / ig.data$TicketFreq+ig.data$Age/70
ig.data$FamilySize <- ig.data$SibSp + ig.data$Parch + 1

