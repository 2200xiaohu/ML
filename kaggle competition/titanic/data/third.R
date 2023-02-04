#标记nogroup
ig.data$group<-1
#没有group
ig.data$group[which(ig.data$GroupId=="noGroup")]<-0
ig.data$group[which(ig.data$GroupId=="noGroup"&ig.data$Title=="woman")]<-2
#
data<-ig.data[1:891,]%>%select(FamilySize,Survived,wc,FP,FS)
trainIndex <- createDataPartition(data$Survived, p = .7, 
                                  list = FALSE, 
                                  times = 1)

Train <- data[ trainIndex,]
Test <- data[ -trainIndex,]



write.csv(ig.data,"ig.data.csv")
