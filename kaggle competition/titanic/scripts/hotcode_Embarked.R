#one hot encoding

hot.enc.Embarked2<-function(tmp){
  #港口转换
  #s=0,c=1,q=2
  tmp$Embarked[which(tmp$Embarked=='S')]<-0
  tmp$Embarked[which(tmp$Embarked=='C')]<-1
  tmp$Embarked[which(tmp$Embarked=='Q')]<-2
  tmp$Embarked<-as.numeric(tmp$Embarked)
  tmp<-data.frame(Embarked0=c(NA),Embarked1=c(NA),Embarked2=c(NA))%>%cbind(tmp)
  for(i in c(1:nrow(tmp))){
    print(tmp$Embarked[i])
    if(tmp$Embarked[i]==1){
      tmp$Embarked0[i]=0
      tmp$Embarked1[i]=1
      tmp$Embarked2[i]=0
    }
    else if(tmp$Embarked[i]==0){
      tmp$Embarked0[i]=1
      tmp$Embarked1[i]=0
      tmp$Embarked2[i]=0
    }
    else if(tmp$Embarked[i]==2){
      tmp$Embarked0[i]=0
      tmp$Embarked1[i]=0
      tmp$Embarked2[i]=1
    }
    else{
      tmp$Embarked0[i]=NA
      tmp$Embarked1[i]=NA
      tmp$Embarked2[i]=NA
    }
  }
  return(tmp)
}

hot.enc.Embarked1<-function(tmp){
  #港口转换
  #s=0,c=1,q=2
  tmp$Embarked[62]<-3
  tmp$Embarked[830]<-3
  tmp$Embarked[which(tmp$Embarked=='S')]<-0
  tmp$Embarked[which(tmp$Embarked=='C')]<-1
  tmp$Embarked[which(tmp$Embarked=='Q')]<-2
  tmp$Embarked<-as.numeric(tmp$Embarked)
  tmp<-data.frame(Embarked0=c(NA),Embarked1=c(NA),Embarked2=c(NA))%>%cbind(tmp)
  for(i in c(1:nrow(tmp))){
    print(tmp$Embarked[i])
    if(tmp$Embarked[i]==1){
      tmp$Embarked0[i]=0
      tmp$Embarked1[i]=1
      tmp$Embarked2[i]=0
    }
    else if(tmp$Embarked[i]==0){
      tmp$Embarked0[i]=1
      tmp$Embarked1[i]=0
      tmp$Embarked2[i]=0
    }
    else if(tmp$Embarked[i]==2){
      tmp$Embarked0[i]=0
      tmp$Embarked1[i]=0
      tmp$Embarked2[i]=1
    }
    else{
      tmp$Embarked0[i]=NA
      tmp$Embarked1[i]=NA
      tmp$Embarked2[i]=NA
    }
  }
  return(tmp)
}

