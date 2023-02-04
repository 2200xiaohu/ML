library(ParBayesianOptimization)
x<-model.matrix(Survived~.,Train)#为什么变小了
y<-model.frame(Survived~.,Train)[,"Survived"]
scoring_function <- function(
  eta, gamma, max_depth, min_child_weight, subsample, nfold) {
  
  dtrain <- xgb.DMatrix(x, label = y, missing = NA)
  
  pars <- list(
    eta = eta,
    gamma = gamma,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    verbosity = 0
  )
  
  xgbcv <- xgb.cv(
    params = pars,
    data = dtrain,
    
    nfold = nfold,
    
    nrounds = 200,
    prediction = TRUE,
    showsd = TRUE,
    early_stopping_rounds = 10,
    maximize = TRUE,
    stratified = TRUE
  )
  
  # required by the package, the output must be a list
  # with at least one element of "Score", the measure to optimize
  # Score must start with capital S
  # For this case, we also report the best num of iteration
  return(
    list(
      Score = max(xgbcv$evaluation_log$test_auc_mean),
      nrounds = xgbcv$best_iteration
    )
  )
}


bounds <- list(
  eta = c(0, 1),
  gamma =c(0, 100),
  max_depth = c(2L, 50L), # L means integers
  min_child_weight = c(1, 25),
  subsample = c(0.25, 1),
  nfold = c(3L, 10L)
)

library(doParallel)
no_cores <- detectCores() -2 # use half my CPU cores to avoid crash  
cl <- makeCluster(no_cores) # make a cluster
registerDoParallel(cl) # register a parallel backend
clusterExport(cl, c('x','y')) # import objects outside
clusterEvalQ(cl,expr= { # launch library to be used in FUN
  library(xgboost)
})

time_withparallel <- system.time(
  opt_obj <- bayesOpt(
    FUN = scoring_function,
    bounds = bounds,
    initPoints = 20,
    iters.n = 50,
    parallel = TRUE
  ))
stopCluster(cl) # stop the cluster
registerDoSEQ() # back to serial computing
opt_obj$scoreSummary
pa<-opt_obj$scoreSummary[which(opt_obj$scoreSummary$Score==min(opt_obj$scoreSummary$Score))]
pa<-opt_obj$scoreSummary[29,]

getBestPars(opt_obj)

params <- list(eta = getBestPars(opt_obj)[1],
               gamma = getBestPars(opt_obj)[2],
               max_depth = getBestPars(opt_obj)[3],
               min_child_weight = getBestPars(opt_obj)[4],
               subsample = getBestPars(opt_obj)[5],
               nfold = getBestPars(opt_obj)[6],
               objective = "binary:logistic")
params <- list(eta = pa[1,3],
               gamma = pa[1,4],
               max_depth = pa[1,5],
               min_child_weight = pa[1,6],
               subsample = pa[1,7],
               nfold = pa[1,8],
               objective = "binary:logistic")
# the numrounds which gives the max Score (auc)
numrounds <- opt_obj$scoreSummary$nrounds[
  which(opt_obj$scoreSummary$Score
        == max(opt_obj$scoreSummary$Score))]
numrounds <-pa[1,14]

fit_tuned <- xgboost(params = params,
                     data = x,
                     label = y,
                     nrounds = as.numeric(pa[1,14]),
                     eval_metric = "auc")
xvals<-model.matrix(Survived~.,Test)
yvals<-model.frame(Survived~.,Test)[,"Survived"]

pre<-predict(fit_tuned,xvals,type="res")
predictions2<-as.factor(ifelse(predict(fit_tuned,xvals)>0.5,1,0))
performance.nottune<-confusionMatrix(predictions2,factor(yvals))
performance.nottune
