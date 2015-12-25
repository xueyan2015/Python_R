


library(randomForest)
library(glmnet)
library(snowfall)


setwd("D:/Project Files2/Python Efficiency/test RF using another dataset")

bootstrap_T<- c(TRUE)
bootstrap_F<- c(FALSE)
ntree <- c(300, 500, 800, 1500)
mtry <- c(3, 5, 10, 20, 30, 40, 50, 60, 70, 80) #-- range of mtry has been extended
nodesize <- c(1, 5, 10, 20, 30, 50)
maxnodes <- c(NA, 3, 6 ,10, 15, 20) 

# this is only for bootstrap sampling...
sampsize_T <- c(1)
# this is only for sampling with no replacement...
sampsize_F <- c(0.632, 1) 

#class weight
class_weight <- list(NULL, c('0'=1,'1'=1), c('0'= 0.202689, '1'= 0.797311))
class_weight_index <- c(1:length(class_weight))





#design
#parm_combos <- expand.grid(bootstrap = bootstrap ,ntree = ntree , mtry = mtry, nodesize = nodesize, maxnodes=maxnodes) 
parm_combos_T <- expand.grid(bootstrap = bootstrap_T ,ntree = ntree , mtry = mtry, nodesize = nodesize, maxnodes=maxnodes, sampsize= sampsize_T, v_class_wt= class_weight_index)
parm_combos_F <- expand.grid(bootstrap = bootstrap_F ,ntree = ntree , mtry = mtry, nodesize = nodesize, maxnodes=maxnodes, sampsize= sampsize_F, v_class_wt= class_weight_index)
# is.data.frame(parm_combos)
parm_combos <- rbind(parm_combos_T, parm_combos_F)
# table(parm_combos$bootstrap, parm_combos$sampsize)
head(parm_combos)
# table(parm_combos$bootstrap, parm_combos$v_class_wt)



Run_RandomForest_v2 <- function(file_id, parm_combos,class_weight,n_processors=3){
  
  X_train_file <- paste("X_train_", file_id, ".csv",sep="")
  y_train_file <- paste("y_train_", file_id, ".csv",sep="")
  
  X_test_file <- paste("X_test_", file_id, ".csv",sep="")
  y_test_file <- paste("y_test_", file_id, ".csv", sep="")
  
  X_train <- read.csv(X_train_file, header=FALSE)
  y_train <- read.csv(y_train_file, header=FALSE)
  colnames(y_train) <- "response"
  
  X_test <- read.csv(X_test_file, header=FALSE)
  y_test <- read.csv(y_test_file, header=FALSE)
  colnames(y_test) <- "response"
  
  
  #loop thru parm_combos using thread processing...
  sfInit(parallel=TRUE, cpus=n_processors)
  
  sfLibrary("randomForest",character.only = TRUE)
  sfLibrary("glmnet",character.only = TRUE)
  #data
  sfExport('X_train','y_train','X_test','y_test','parm_combos','class_weight')
  #key functions
  sfExport("auc", namespace = "glmnet")
  sfExport("randomForest", namespace = "randomForest")
 
  #run 
  outcomes <- sfClusterApplyLB(1:nrow(parm_combos), quick_rf_v2)
  sfStop()
  
  results_dframe <- matrix(nrow = nrow(parm_combos), ncol = 10)
#   results_dframe <- as.data.frame(results_dframe)
  for(j in 1:nrow(parm_combos)){
  results_dframe[j,] <- outcomes[[j]]
  
  }


  colnames(results_dframe)<- c('Loop ID','bootstrap','ntree', 'mtry', 'nodesize', 'maxnodes', 'sampsize', 'v_class_wt', 'auc_training','auc_testing')
  
  write.csv(results_dframe, paste('R_RF_Performance_File', file_id, '.csv',sep=''),row.names=F)
  
  return(results_dframe)

  
}


quick_rf_v2 <- function(inner_loop){
  
  
  
  rf_model<- randomForest(x=X_train, y=as.factor(y_train[,1])
                         , ntree= parm_combos[inner_loop,'ntree']
                         , mtry=  parm_combos[inner_loop,'mtry']
                         , nodesize = parm_combos[inner_loop,'nodesize']
                         , replace= parm_combos[inner_loop,'bootstrap']
                         , maxnodes = if(is.na(parm_combos[inner_loop,'maxnodes'])) NULL else parm_combos[inner_loop,'maxnodes']
                         , sampsize = nrow(X_train) * parm_combos[inner_loop, 'sampsize']
                         , classwt = class_weight[[parm_combos[inner_loop,'v_class_wt']]]
                         )
  
  pred_training <- predict(rf_model, X_train, type='prob') 
  pred_testing <-  predict(rf_model, X_test, type='prob')
  
  auc_training <- auc(y_train[,1], pred_training[,2])
  auc_testing <-  auc(y_test[,1], pred_testing[,2])
  
  return(c(inner_loop ,parm_combos[inner_loop,'bootstrap'], parm_combos[inner_loop,'ntree'], parm_combos[inner_loop,'mtry'], parm_combos[inner_loop,'nodesize'], parm_combos[inner_loop,'maxnodes'], parm_combos[inner_loop, 'sampsize'], parm_combos[inner_loop,'v_class_wt'], auc_training, auc_testing))
  
  
  
}


for(outer_loop in 1:2){
  
  Run_RandomForest_v2(outer_loop, parm_combos, class_weight, n_processors=3)
  
}



