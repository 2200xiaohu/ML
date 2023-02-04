ig.data

library(cluster)
library(Rtsne)


data.train<-ig.data%>%select(Sex,Age)
data.train$Sex[which(data.train$Sex=="male")]<-1
data.train$Sex[which(data.train$Sex=="female")]<-0
data.train$Sex<-as.numeric(data.train$Sex)
data.train<-scale(data.train)
gower_dist<-daisy(data.train,metric="gower")
gower_mat<-as.matrix(gower_dist)

sil_width<-c(NA)
for (i in 2:8){
  pam_fit<-pam(gower_dist,diss=T,k=i)
  sil_width[i]<-pam_fit$silinfo$avg.width
}
plot(1:8,sil_width,
     xlab="cluster number",
     ylab="silhouette width"
)
lines(1:8,sil_width)

k<-2
pam_fit<-pam(gower_dist,diss=T,k)
pam_results <- data.train %>%
  cbind(as.factor(pam_fit$clustering))

%>%
  group_by(cluster) %>%
  do(the_summary = summary(.))
pam_results$the_summary

tsne_obj <- Rtsne(gower_dist, is_distance = TRUE)
tsne_data <- tsne_obj$Y %>%
  data.frame() %>%
  setNames(c("X", "Y")) %>%
  mutate(cluster = factor(pam_fit$clustering))
ggplot(aes(x = X, y = Y), data = tsne_data) +
  geom_point(aes(color = cluster))




#
ig.data<-ig.data%>%cbind(w=as.numeric(pam_fit$clustering))
