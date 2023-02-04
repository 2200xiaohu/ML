#one hot encoding

hot.enc.sex<-function(tmp){
  #性别转换，0为女性，1为男性
  tmp$Sex[which(tmp$Sex=="male")]<-1
  tmp$Sex[which(tmp$Sex=="female")]<-0
  tmp$Sex<-as.numeric(tmp$Sex)
  tmp<-data.frame(Male=c(NA),Female=c(NA))%>%cbind(tmp)
  for(i in c(1:nrow(tmp))){
    if(tmp$Sex[i]==1){
      tmp$Male[i]=1
      tmp$Female[i]=0
    }
    if(tmp$Sex[i]==0){
      tmp$Male[i]=0
      tmp$Female[i]=1
    }
  }
  return(tmp)
}



