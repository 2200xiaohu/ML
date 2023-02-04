##################################################################
feature  FamilySize,Survived,FareAdj,Sex
svmlinear    C=0.1
xgboost   grid_default <- expand.grid(
  nrounds = 11,
  max_depth = 7,
  eta = 0.8,
  gamma = 3,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.5
)

stack  glm
test Acc 0.7903 threshold 0.5
score 0.78947

########################################################
feature  FamilySize,Survived,FareAdj,Sex,FP
svmlinear    C=0.
xgboost   grid_default <- expand.grid(
  nrounds = 40,
  max_depth = 7,
  eta = 0.8,
  gamma = 3,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.5
)
stack  glm
test Acc 0.8277 threshold 0.5
score 0.7703

##########################################################
feature  FamilySize,Survived,FareAdj,FP,wc
svmlinear    C=0.
xgboost   grid_default <- expand.grid(
  nrounds = 40,
  max_depth = 7,
  eta = 0.8,
  gamma = 3,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.5
)
stack  glm
test Acc 0.809 threshold 0.5
score 0.74401 


##########################################################
feature  FamilySize,Survived,wc(15 age),FP,FS
svmlinear    C=1.3
xgboost   grid_default <- expand.grid(
  nrounds = 40,
  max_depth = 7,
  eta = 0.8,
  gamma = 3,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.5
)
stack  glm
test Acc 0.8464  threshold 0.5
score 0.77751 


##########################################################
feature  FamilySize,Survived,wc(15 age),FP
svmlinear    C=4.3
xgboost   grid_default <- expand.grid(
  nrounds = 126,
  max_depth = 7,
  eta = 0.8,
  gamma = 3,
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.5
)
stack  glm
test Acc 0.839    threshold 0.5
score 0.7919   

all center,scale

############################################
feature  FamilySize,Survived,wc(15 age),FP,FS
svmlinear    C=0.02002002
xgboost   grid_default <- expand.grid(
  nrounds = 20,
  max_depth = 32,
  eta = 0.4259741,
  gamma = 0,
  colsample_bytree =0.81,
  min_child_weight = 4.413045,
  subsample = 1
)
knn  k=44
stack  glm
test Acc 0.8427    threshold 0.5
score 0.82535

