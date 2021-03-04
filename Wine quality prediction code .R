

################################
## Installation des librairies nécessaires
list.of.packages <- c('glmnet','splitstackshape','caret','corrplot','MASS','tidyverse','e1071','LiblineaR','kernlab','ordinalForest','xgboost','ranger','gbm','VGAM','rpartScore','randomForest','ordinalNet','kknn','h2o','caretEnsemble','haven','factoextra')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)



library(glmnet)
library(splitstackshape)
library(caret)
library(corrplot)
library(MASS)
library(tidyverse)
library(e1071)
library(LiblineaR)
library(kernlab)
library(ordinalForest)
library(xgboost)
library(ranger)
library(gbm)
library(VGAM)
library(rpartScore)
library(randomForest)
library(ordinalNet)
library(kknn)
library(h2o)
library(caretEnsemble)
library(haven)
library(factoextra)


################################
## Assignation du chemin pour le répertoire et importation des jeux de données
setwd("C:/Users/BP/Desktop/projets pour github/Wine quality prediction")

df = read.table("datrain.txt", header = TRUE)


################################
## Informations sur le jeu de données / statistiques descriptives

# Convertir la variable y en facteur
df$y = as.factor(df$y)

# Dimensions du jeu de données
dim(df)

# Présences de valeurs manquantes é
sum(is.na(df))

# Statistiques descriptives du jeu de données
summary(df)


# Distribution des différentes variables du jeu de données
df %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram(bins = 50) +
  ggtitle("Distribution des variables du jeu de données")

# Boxplot pour la variable fixedacidity
df %>%
  select(y,fixedacidity)%>%
  ggplot(aes(x = y,y = fixedacidity, color = y)) +
  geom_boxplot() 

# Boxplot pour la variable volatileacidity
df %>%
  select(y,volatileacidity)%>%
  ggplot(aes(x = y,y = volatileacidity, color = y)) +
  geom_boxplot()

# Boxplot pour la variable citricacid
df %>%
  select(y,citricacid)%>%
  ggplot(aes(x = y,y = citricacid, color = y)) +
  geom_boxplot()

# Boxplot pour la variable residualsugar
df %>%
  select(y,residualsugar)%>%
  ggplot(aes(x = y,y = residualsugar, color = y)) +
  geom_boxplot()

# Boxplot pour la variable chlorides
df %>%
  select(y,chlorides)%>%
  ggplot(aes(x = y,y = chlorides, color = y)) +
  geom_boxplot()

# Boxplot pour la variable freesulfurdioxide
df %>%
  select(y,freesulfurdioxide)%>%
  ggplot(aes(x = y,y = freesulfurdioxide, color = y)) +
  geom_boxplot()

# Boxplot pour la variable totalsulfurdioxide
df %>%
  select(y,totalsulfurdioxide)%>%
  ggplot(aes(x = y,y = totalsulfurdioxide, color = y)) +
  geom_boxplot()

# Boxplot pour la variable density
df %>%
  select(y,density)%>%
  ggplot(aes(x = y,y = density, color = y)) +
  geom_boxplot()

# Boxplot pour la variable pH
df %>%
  select(y,pH)%>%
  ggplot(aes(x = y,y = pH, color = y)) +
  geom_boxplot()

# Boxplot pour la variable sulphates
df %>%
  select(y,sulphates)%>%
  ggplot(aes(x = y,y = sulphates, color = y)) +
  geom_boxplot()

# Boxplot pour la variable alcohol
df %>%
  select(y,alcohol)%>%
  ggplot(aes(x = y,y = alcohol, color = y)) +
  geom_boxplot()

# Distribution de la variable y
ggplot(df, aes(y)) + geom_bar() +ggtitle("Distribution de Y")
table(df$y)

# Corrélation entre les variables
cor(df[, c(1:11)])
res = cor(df[, c(1:11)])

corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)




#######################################################
#######################################################
# Préparation du jeu de données

## On cap les valeurs extrémes en les égalant au 1er et dernier centile
fun <- function(x){
  quantiles <- quantile( x, c(.01, .99 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  x
}

for (col in names(df)[names(df) %in% c("freesulfurdioxide",
                                       "residualsugar",
                                       "chlorides",
                                       "totalsulfurdioxide",
                                       "volatileacidity")]) {
  
  df[,col] = fun(df[,col])
}
  

# seed
set.seed(100)

# on divise le dataset en train / test et on le standardise
# on utilise la fonction createDataPartition() pour garder les proportions de Y dans le train et le test
datatrain.index <- createDataPartition(df$y, p = 0.8, list = FALSE)
datatrain <- df[ datatrain.index,]
datatest  <- df[-datatrain.index,]

datatrain_y = datatrain[,"y"]
datatest_y = datatest[,"y"]

## On centre et on standardise les variables
preProcValues <- preProcess(datatrain[ , -which(names(datatrain) %in% 'y')], 
                            method = c("center", "scale"))

datatrain <- predict(preProcValues, datatrain[ , -which(names(datatrain) %in% 'y')])
datatest <- predict(preProcValues, datatest[ , -which(names(datatest) %in% 'y')])


## On ajoute la variable réponse aux jeu de données train et test
datatrain = cbind(datatrain,datatrain_y)
names(datatrain)[names(datatrain) == 'datatrain_y'] <- 'y'
datatest = cbind(datatest,datatest_y)
names(datatest)[names(datatest) == 'datatest_y'] <- 'y'



# Création d'un tableau qui va regrouper l'ensemble des performances des différents modéles
performance =  data.frame("Modéle" = NA, "Taux_de_bonne_classification" = NA)




#######################################################
#######################################################
# Développement de modèles de base


###################
###################
## Approche naive : on classe tout dans la classe majoritaire (y = 2)

datatest$y_naif = 2
acc_naif = sum(datatest$y_naif == datatest$y) / nrow(datatest) *100
performance[nrow(performance),] = list("Approche naive",acc_naif)


## La validation croisée 5 folds sera utilisée car le jeu de données est relativement petit
fitControl <- trainControl(method = "cv",number = 5, search="grid")



##########
# Régression linéaire : ici on considére la variable Y comme continue.

set.seed(100)
datatrain$y = as.numeric(datatrain$y)
datatest$y = as.numeric(datatest$y)

# Entrainement
model_lin_reg <- train(y ~ ., data = datatrain, 
                 method = "lm", 
                 trControl = fitControl)

# Prédictions & performance sur échantillon test
datatest$prediction_reg_lin = round(predict(model_lin_reg, newdata = datatest,type = "raw")) # il est nécessaire d'arrondir les prédictions
acc_lin_reg = sum(datatest$prediction_reg_lin == datatest$y) / nrow(datatest) *100 # taux de bonne classification
performance[nrow(performance)+1,] = list("Régression linéaire BASE",acc_lin_reg)



##########
# Régression logistique - "proportional odds logistic regression"
set.seed(100)

# On convertit la variable réponse en facteur ordonné
datatrain$y <- factor(datatrain$y, levels = c("1", "2", "3"),ordered = T)
datatest$y <- factor(datatest$y, levels = c("1", "2", "3"),ordered = T)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           search = "grid",
                           savePredictions="final", 
                           classProbs = FALSE)
# Grid search 
log_reg_grid <- expand.grid(method = c("logistic", "probit", "loglog", "cloglog", "cauchit"))

# Entrainement
model_log_reg <- train(y ~ ., data = datatrain, 
                       method = "polr",
                       metric = "Accuracy", # On veut maximiser le taux de bonne classification
                       maximize = T, # On veut maximiser le taux de bonne classification
                       trControl = fitControl,
                       tuneGrid=log_reg_grid)


# Prédictions & performance sur échantillon test
datatest$prediction_log_reg = predict(model_log_reg,newdata = datatest)
acc_log_reg = sum(datatest$prediction_log_reg == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("Régression logistique BASE",acc_log_reg)


#########################
# Modèles SVM


# SVM polynomial
set.seed(100)
svm_poly_grid <- expand.grid(scale = TRUE,
                             degree = c(2,3,4,5),
                             C = 1)
# Entrainement
model_svm_poly <- train(y ~ .,
                        data = datatrain,
                        method="svmPoly",
                        trControl = fitControl,
                        tuneGrid=svm_poly_grid)

# Prédictions & performance sur échantillon test
datatest$prediction_svm_poly = predict(model_svm_poly,newdata = datatest)
acc_svm_poly = sum(datatest$prediction_svm_poly == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("SVM polynomial BASE",acc_svm_poly)



# SVM radial
set.seed(100)

svm_radial_grid <- expand.grid(sigma = c(.01, .015, 0.2),
                    C = c(0.75, 0.9, 1, 1.1, 1.25))

# Entrainement
model_svm_radial <- train(y ~ .,
                          data = datatrain,
                          method="svmRadial",
                          trControl = fitControl,
                          tuneGrid=svm_radial_grid)

# Prédictions & performance sur échantillon test
datatest$prediction_svm_radial = predict(model_svm_radial,newdata = datatest)
acc_svm_radial = sum(datatest$prediction_svm_radial == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("SVM radial BASE",acc_svm_radial)



##########
# Modèle : Adjacent Categories Probability Model for Ordinal Data
set.seed(100)

vglmAdjCat_grid <- expand.grid(parallel = c(T,F),
                               link = c("extlogitlink", 
                                        "logofflink",
                                        "identitylink",
                                        "negidentitylink", 
                                        "reciprocallink", 
                                        "negreciprocallink"))

# Entrainement
model_vglmAdjCat <- train(y ~ ., data = datatrain, 
                       method = "vglmAdjCat",
                       metric = "Accuracy",
                       maximize = T, 
                       trControl = fitControl,
                       tuneGrid=vglmAdjCat_grid
                       )

# Prédictions & performance sur échantillon test
datatest$prediction_vglmAdjCat = predict(model_vglmAdjCat,newdata = datatest)
acc_vglmAdjCat = sum(datatest$prediction_vglmAdjCat == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("Adjacent Categories Probability BASE",acc_vglmAdjCat)



########
# Modèle CART Ordinal Responses
set.seed(100)

# Entrainement
model_CART_ord_resp <- train(y~., data = datatrain,
                          method = "rpartScore",
                          metric = "Accuracy",
                          maximize = T, 
                          trControl = fitControl)

# Prédictions & performance sur échantillon test
datatest$prediction_CART_ord_resp = predict(model_CART_ord_resp,newdata = datatest)
acc_CART_ord_resp = sum(datatest$prediction_CART_ord_resp == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("CART ordinal response BASE",acc_CART_ord_resp)


##########
# Modèle Continuation Ratio Model for Ordinal Data
set.seed(100)

vglmContRatio_grid <- expand.grid(parallel = c(T,F),
                               link = c("extlogitlink", 
                                        "logofflink",
                                        "identitylink",
                                        "negidentitylink", 
                                        "reciprocallink", 
                                        "negreciprocallink"))


# Entrainement
model_vglmContRatio <- train(y ~ ., data = datatrain, 
                          method = "vglmContRatio",
                          metric = "Accuracy",
                          maximize = T, 
                          trControl = fitControl,
                          tuneGrid=vglmContRatio_grid)

# Prédictions & performance sur échantillon test
datatest$prediction_vglmContRatio = predict(model_vglmContRatio,newdata = datatest)
acc_vglmContRatio = sum(datatest$prediction_vglmContRatio == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("Continuation Ratio Model BASE",acc_vglmContRatio)



##########
# Modèle Cumulative Probability Model for Ordinal Data
set.seed(100)

vglmCumulative_grid <- expand.grid(parallel = c(T,F),
                                  link = c("extlogitlink", 
                                           "logofflink",
                                           "identitylink",
                                           "negidentitylink", 
                                           "reciprocallink", 
                                           "negreciprocallink"))

# Entrainement
model_vglmCumulative <- train(y ~ ., data = datatrain, 
                          method = "vglmCumulative",
                          metric = "Accuracy",
                          maximize = T, 
                          trControl = fitControl,
                          tuneGrid=vglmCumulative_grid)

# Prédictions & performance sur échantillon test
datatest$prediction_vglmCumulative = predict(model_vglmCumulative,newdata = datatest)
acc_vglmCumulative = sum(datatest$prediction_vglmCumulative == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("Cumulative Probability Model BASE",acc_vglmCumulative)



##########
# Modèle Penalized_Ordinal_Regression
set.seed(100)

Penalized_Ordinal_Regression_grid <- expand.grid(alpha = c(0.1,0.5,0.9), 
                                                 criteria = c("aic", "bic"), 
                                                 link = c("logit"))


# Entrainement
model_Penalized_Ordinal_Regression <- train(y ~ ., data = datatrain, 
                              method = "ordinalNet",
                              metric = "Accuracy",
                              maximize = T, 
                              trControl = fitControl,
                              tuneGrid=Penalized_Ordinal_Regression_grid)

# Prédictions & performance sur échantillon test
datatest$prediction_Penalized_Ordinal_Regression = predict(model_Penalized_Ordinal_Regression,newdata = datatest)
acc_Penalized_Ordinal_Regression = sum(datatest$prediction_Penalized_Ordinal_Regression == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("Penalized_Ordinal_Regression BASE",acc_Penalized_Ordinal_Regression)



##########
# Modèle de forêt aléatoire ordinale - ordinalRF
set.seed(100)

# Entrainement
model_Random_Forest_ordinalRF <- train(y ~ ., data = datatrain, 
                                            method = "ordinalRF",
                                            metric = "Accuracy",
                                            maximize = T, 
                                            trControl = fitControl,
                                       tuneLength=2)
                                       

# Prédictions & performance sur échantillon test
datatest$prediction_Random_Forest_ordinalRF = predict(model_Random_Forest_ordinalRF,newdata = datatest)
acc_Random_Forest_ordinalRF = sum(datatest$prediction_Random_Forest_ordinalRF == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("Ordinal Random Forest BASE",acc_Random_Forest_ordinalRF)

model_Random_Forest_ordinalRF$results$AccuracySD

#############
# Modèle KNN
set.seed(100)

kknn_grid <- expand.grid(kmax = seq(5,25,2), distance = c(1, 2,3,4,5),
                         kernel = c("rectangular","triangular","biweight",
                                    "epanechnikov", "gaussian", "cos",
                                    "inv", "optimal"))

# Entrainement
model_knn <- train(y ~ ., 
                   data = datatrain,
                   method = "kknn",
                   trControl = fitControl,
                   tuneGrid = kknn_grid)


# Prédictions & performance sur échantillon test
datatest$prediction_knn = predict(model_knn,newdata = datatest)
acc_knn = sum(datatest$prediction_knn == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("KNN BASE",acc_knn)




##########
# Modèle de forêt aléatoire classique avec grid search
set.seed(100)

RF_tunegrid <- expand.grid(mtry = seq(1,length(names(datatrain))-1),
                           splitrule = c("gini", "extratrees") ,
                           min.node.size = c(1:10))

# Entrainement
model_RF <- train(y ~ ., data = datatrain, 
                                       method = "ranger",
                                       metric = "Accuracy",
                                       maximize = T, 
                                       trControl = fitControl,
                                       tuneGrid = RF_tunegrid)

# Prédictions & performance sur échantillon test
datatest$prediction_RF = predict(model_RF,newdata = datatest)
acc_RF = sum(datatest$prediction_RF == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("RF BASE",acc_RF)





##########
# Modèle Gradient Boosting
set.seed(100)

gbm_tunegrid <- expand.grid(n.trees = seq(100,500,100),
                            interaction.depth = c(1,2,3),
                            shrinkage = seq(0.001, 0.1,0.01),
                            n.minobsinnode = 10)

# Entrainement
model_gbm <- train(y ~ ., data = datatrain, 
                   method = "gbm",
                   metric = "Accuracy",
                   maximize = T, 
                   trControl = fitControl,
                   tuneGrid = gbm_tunegrid)

# Prédictions & performance sur échantillon test
datatest$prediction_gbm = predict(model_gbm,newdata = datatest)
acc_gbm = sum(datatest$prediction_gbm == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("GBM BASE",acc_gbm)



##########
# Modèle de Réseau de neurones
set.seed(100)

nn_grid = expand.grid(decay = seq(0.01,0.3,0.05), size = seq(4,12,1))

# Entrainement
model_nn <- train(y ~ ., data = datatrain, 
                   method = "nnet",
                   metric = "Accuracy",
                   maximize = T, 
                   trControl = fitControl,
                  maxit = 1000,
                  linout = 0,
                  tuneGrid = nn_grid)

# Prédictions & performance sur échantillon test
datatest$prediction_nn = predict(model_nn,newdata = datatest)
acc_nn = sum(datatest$prediction_nn == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("nn BASE",acc_nn)


# On compare tous les modéles précédents
set.seed(100)
resamps <- resamples(list(Reg_log = model_log_reg,
                          SVM_Poly = model_svm_poly,
                          SVM_Radial = model_svm_radial,
                          Adjacent_Categories_Probability_Model_Ordinal_Data = model_vglmAdjCat,
                          CART_ordinal_response = model_CART_ord_resp,
                          Continuation_Ratio_Model_Ordinal_Data = model_vglmContRatio,
                          Cumulative_Probability_Model_Ordinal_Data = model_vglmCumulative,
                          Penalized_Ordinal_Regression = model_Penalized_Ordinal_Regression,
                          ordinalRF = model_Random_Forest_ordinalRF,
                          KNN = model_knn,
                          Random_Forest_classique = model_RF,
                          Gradient_Boosting = model_gbm,
                          Neural_net = model_nn))

summary(resamps)
bwplot(resamps, metric = "Accuracy")




################################################################################
################################################################################
### H2O
################################################################################
################################################################################




################################################################################
df = read.table("datrain.txt", header = TRUE)


## On cap les valeurs extrémes en les égalant au 1er et dernier centile
fun <- function(x){
  quantiles <- quantile( x, c(.01, .99 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  x
}

for (col in names(df)[names(df) %in% c("freesulfurdioxide",
                                       "residualsugar",
                                       "chlorides",
                                       "totalsulfurdioxide",
                                       "volatileacidity")]) {
  
  df[,col] = fun(df[,col])
}


# seed
set.seed(100)

# on divise le dataset en train / test et on le standardise
# on utilise la fonction createDataPartition() pour garder les proportions de Y dans le train et le test
datatrain.index <- createDataPartition(df$y, p = 0.8, list = FALSE)
datatrain <- df[ datatrain.index,]
datatest  <- df[-datatrain.index,]

datatrain_y = datatrain[,"y"]
datatest_y = datatest[,"y"]

## On standardise les variables
preProcValues <- preProcess(datatrain[ , -which(names(datatrain) %in% 'y')], method = c("center", "scale"))

datatrain <- predict(preProcValues, datatrain[ , -which(names(datatrain) %in% 'y')])
datatest <- predict(preProcValues, datatest[ , -which(names(datatest) %in% 'y')])

## On ajoute la variable réponse aux jeu de données train et test
datatrain = cbind(datatrain,datatrain_y)
names(datatrain)[names(datatrain) == 'datatrain_y'] <- 'y'
datatest = cbind(datatest,datatest_y)
names(datatest)[names(datatest) == 'datatest_y'] <- 'y'

################################################################################


##########
# H2O - GBM

# Initialisation de H2O
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain), y)

# On convertit les jeux de données en H2O frame
datatrain_h2o = as.h2o(datatrain)
datatest_h2o = as.h2o(datatest)

# En classification la variable réponse est un facteur
datatrain_h2o[,y] <- as.factor(datatrain_h2o[,y])
datatest_h2o[,y] <- as.factor(datatest_h2o[,y])



# Liste d'hyperparamétres pour GBM
gbm_params <- list(learn_rate = seq(0.01, 0.2, 0.01),
                    max_depth = seq(2, 20, 1),
                    sample_rate = seq(0.5, 1.0, 0.1),
                    col_sample_rate = seq(0.1, 1.0, 0.1),
                   col_sample_rate_per_tree = seq(0.3, 1, 0.05), 
                   ntrees = seq(100, 1000, 50),
                   nbins = seq(20, 500, 10))

# On fait du random search pour le temps de calcul
search_criteria <- list(strategy = "RandomDiscrete", max_models = 30, seed = 100,
                        stopping_tolerance = 0.001,
                        stopping_rounds = 5)


# Validation croisée du modéle
gbm_grid <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid",
                      training_frame = datatrain_h2o,
                      balance_classes = T, # On ré-équilibre les classes de la variables réponse
                      nfolds =  5,
                      seed =100,
                      max_runtime_secs = 900,
                      fold_assignment = 'Stratified', # les folds ont les bonnes proportions de la variable réponse
                      hyper_params = gbm_params,
                      search_criteria = search_criteria)

# On ordonne les modéles par rapport au taux de bonne classification
gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid",
                             sort_by = "accuracy",
                             decreasing = TRUE)


# On prend le meilleur modéle GBM
best_gbm_h2o <- h2o.getModel(gbm_gridperf@model_ids[[1]])


# Prédictions & performance sur échantillon test
prediction_gbm_h2o = h2o.predict(best_gbm_h2o, datatest_h2o)
datatest_h2o = h2o.cbind(datatest_h2o,prediction_gbm_h2o[, "predict"])
datatest = as.data.frame(datatest_h2o)
names(datatest)[names(datatest) == 'predict'] <- 'prediction_gbm_h2o'
acc_gbm_h2o = sum(datatest$prediction_gbm_h2o == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("GBM H2O BASE",acc_gbm_h2o)

h2o.shutdown(prompt = F)




##########
# H2O - Forét aléatoire

# Initialisation de H2O
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain), y)

# On convertit les jeux de données en H2O frame
datatrain_h2o = as.h2o(datatrain)
datatest_h2o = as.h2o(datatest)

# En classification la variable réponse est un facteur
datatrain_h2o[,y] <- as.factor(datatrain_h2o[,y])
datatest_h2o[,y] <- as.factor(datatest_h2o[,y])



# Liste d'hyperparamétres pour RF
RF_params <- list(mtries = seq(1,length(names(datatrain))-1),
                  max_depth = seq(2, 15, 1),
                  sample_rate = seq(0.5, 1.0, 0.1),
                  col_sample_rate_per_tree = seq(0.3, 1, 0.05), 
                  ntrees = seq(100, 200, 50),
                  nbins = seq(20, 100, 10))

search_criteria <- list(strategy = "RandomDiscrete", max_models = 30, seed = 100,
                        stopping_tolerance = 0.001,
                        stopping_rounds = 5)


# Validation croisée du modéle
RF_grid <- h2o.grid("drf", x = x, y = y,
                    grid_id = "RF_grid",
                    training_frame = datatrain_h2o,
                    balance_classes = T,
                    nfolds =  5,
                    seed =100,
                    max_runtime_secs = 900,
                    fold_assignment = 'Stratified',
                    hyper_params = RF_params,
                    search_criteria = search_criteria)

# On ordonne les modéles par rapport au taux de bonne classification
RF_gridperf <- h2o.getGrid(grid_id = "RF_grid",
                           sort_by = "accuracy",
                           decreasing = TRUE)


# On prend le meilleur modéle RF
best_RF_h2o <- h2o.getModel(RF_gridperf@model_ids[[1]])


# Prédictions & performance sur échantillon test
prediction_RF_h2o = h2o.predict(best_RF_h2o, datatest_h2o)
datatest_h2o = h2o.cbind(datatest_h2o,prediction_RF_h2o[, "predict"])
datatest = as.data.frame(datatest_h2o)
names(datatest)[names(datatest) == 'predict'] <- 'prediction_RF_h2o'
acc_RF_h2o = sum(datatest$prediction_RF_h2o == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("RF H2O BASE",acc_RF_h2o)

h2o.shutdown(prompt = F)




##########
# H2O - AutoML

# On initialise H2O
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain), y)

# On convertit les jeux de données en H2O frame
datatrain_h2o = as.h2o(datatrain)
datatest_h2o = as.h2o(datatest)

# En classification la variable réponse est un facteur
datatrain_h2o[,y] <- as.factor(datatrain_h2o[,y])
datatest_h2o[,y] <- as.factor(datatest_h2o[,y])

# Entrainement par validation croisée
model_automl_h2o <- h2o.automl(x = x, 
                               y = y, 
                               training_frame = datatrain_h2o,
                               nfolds =  5,
                               balance_classes = T,
                               max_runtime_secs = 900,
                               seed = 100)

model_automl_h2o@leaderboard

# Prédictions & performance sur échantillon test
prediction_automl_h2o = h2o.predict(model_automl_h2o@leader, datatest_h2o)
datatest_h2o = h2o.cbind(datatest_h2o,prediction_automl_h2o[, "predict"])
datatest = as.data.frame(datatest_h2o)
names(datatest)[names(datatest) == 'predict'] <- 'prediction_automl_h2o'
acc_automl_h2o = sum(datatest$prediction_automl_h2o == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("H2O AutoML BASE",acc_automl_h2o )


h2o.shutdown(prompt = F)




#####################################
#####################################
# H2O - méthode d'ensemble forêt aléatoire + GBM - Pas de grid search
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain), y)

# On convertit les jeux de données en H2O frame
datatrain_h2o = as.h2o(datatrain)
datatest_h2o = as.h2o(datatest)

# Variable réponse en facteur
datatrain_h2o[,y] <- as.factor(datatrain_h2o[,y])
datatest_h2o[,y] <- as.factor(datatest_h2o[,y])


# Nombre de folds en validation croisée
nfolds <- 5

# GBM + RF

# Entrainement GBM
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = datatrain_h2o,
                  distribution = "multinomial",
                  balance_classes = T,
                  seed =100,
                  max_runtime_secs = 900,
                  fold_assignment = 'Stratified',
                  nfolds = nfolds,
                  keep_cross_validation_predictions = TRUE)

# Entrainement RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = datatrain_h2o,
                          ntrees = 100,
                          balance_classes = T,
                          nfolds = nfolds,
                          max_runtime_secs = 900,
                          fold_assignment = 'Stratified',
                          keep_cross_validation_predictions = TRUE,
                          seed = 100)

# Entrainement modéle stacked
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                seed = 100,
                                training_frame = datatrain_h2o,
                                model_id = "my_ensemble",
                                base_models = list(my_gbm, my_rf))


# Prédictions & performance sur échantillon test
prediction_ensemble_h2o <- h2o.predict(ensemble, newdata = datatest_h2o)
datatest_h2o = h2o.cbind(datatest_h2o,prediction_ensemble_h2o[, "predict"])
datatest = as.data.frame(datatest_h2o)
names(datatest)[names(datatest) == 'predict'] <- 'prediction_ensemble_h2o_bis'
acc_ensemble_h2o = sum(datatest$prediction_ensemble_h2o == datatest$y) / nrow(datatest) *100
performance[nrow(performance)+1,] = list("Ensemble H2O bis base",acc_ensemble_h2o)

h2o.shutdown(prompt = F)







###############################################################################################################
###############################################################################################################
###############################################################################################################
##############################################################################################################################################################################################################################
##############################################################################################################################################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
### Création de nouvelles variables 

### on reprend le jeu de données initial
df = read.table("datrain.txt", header = TRUE)




########################################################################################
# Création d'une variable qui indique les valeurs extrémes - Isolation Forest

h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(df), y)

# On convertit les jeux de données en H2O frame
df_h2o = as.h2o(df)


# On construit une forét d'isolation
isol_forest <- h2o.isolationForest(training_frame=df_h2o[,names(df)[names(df) !='y']],
                             seed = 100,
                             max_runtime_secs = 500)


## Ajout du score d'anomalie dans le dataset df
score_isolation_forest_train = h2o.predict(isol_forest, df_h2o[,names(df)[names(df) != 'y']])
score_isolation_forest_train = as.data.frame(score_isolation_forest_train[, "predict"])
names(score_isolation_forest_train)[names(score_isolation_forest_train) == 'predict'] <- 'isolation_forest'




h2o.shutdown(prompt = F)




#####################################
### Intéractions

## jeu de données train
df_temp = df[,names(df)[names(df) !='y']]
df_inter_train=data.frame(model.matrix(~(.)^2-1,df_temp))

df_inter_train = df_inter_train[ , -which(names(df_inter_train) %in% c("fixedacidity",
                                                      "volatileacidity",
                                                      "citricacid",      
                                                      "residualsugar",   
                                                      "chlorides",
                                                      "freesulfurdioxide",
                                                      "totalsulfurdioxide", 
                                                      "density",        
                                                      "pH",        
                                                      "sulphates",
                                                      "alcohol"))]



## jeu de données test        

df_inter_test=data.frame(model.matrix(~(.)^2-1,df_temp))

df_inter_test = df_inter_test[ , -which(names(df_inter_test) %in% c("fixedacidity",
                                                                       "volatileacidity",
                                                                       "citricacid",      
                                                                       "residualsugar",   
                                                                       "chlorides",
                                                                       "freesulfurdioxide",
                                                                       "totalsulfurdioxide", 
                                                                       "density",        
                                                                       "pH",        
                                                                       "sulphates",
                                                                       "alcohol"))] 
 

#####################################
##### LOG : 

## Jeu de données train
df_temp = df[ , -which(names(df) %in% c('y',"citricacid"))]
df_log_train =data.frame(log(df_temp))
names(df_log_train)=paste(names(df_log_train),"LOG", sep="_")






#####################################
##### SQRT : 

## Jeu de données train
df_temp = df[ , -which(names(df) %in% c('y'))]
df_sqrt_train =data.frame((df_temp)^(1/2))
names(df_sqrt_train)=paste(names(df_sqrt_train),"SQRT", sep="_")





#####################################
##### Au carré :

## Jeu de données train
df_temp = df[ , -which(names(df) %in% c('y'))]
df_carre_train =data.frame((df_temp)^(2))
names(df_carre_train)=paste(names(df_carre_train),"carre", sep="_")




#####################################
##### Fraction :

## Jeu de données train
df_temp = df[ , -which(names(df) %in% c('y',"citricacid"))]

colonnes_1 = names(df)[!names(df) %in% c("y","citricacid")]
colonnes_2 = names(df)[!names(df) %in% c("y","citricacid")]

df_div_train = df_temp


for (col1 in colonnes_1){
  colonnes_2 = names(df_temp)
  colonnes_2 = colonnes_2[! colonnes_2 %in% col1]
  
  for (col2 in colonnes_2){
    
    df_div_train[paste('DIV',col1,col2,sep = "_")] = df_div_train[col2]/df_div_train[col1]
      
  }
  
}

df_div_train = df_div_train[ , -which(names(df_div_train) %in% c("fixedacidity",
                                                  "volatileacidity",
                                                  "citricacid",      
                                                  "residualsugar",   
                                                  "chlorides",
                                                  "freesulfurdioxide",
                                                  "totalsulfurdioxide", 
                                                  "density",        
                                                  "pH",        
                                                  "sulphates",
                                                  "alcohol"))]





##### AJOUT DE TOUTES LES VARIABLES AU TRAIN ET AU TEST
## Train
df_feat_eng = cbind(df,
           df_carre_train,
           df_div_train,
           df_inter_train,
           df_log_train,
           df_sqrt_train,
           score_isolation_forest_train)





############################################
# Création d'une nouvelle variable : molecular S02 (intuition basée sur ce lien : http://srjcstaff.santarosa.edu/~jhenderson/SO2.pdf)

df_feat_eng$molecular_so2 = df_feat_eng$freesulfurdioxide/(1+10^(df_feat_eng$pH-1.8))


# Convertir la variable y en facteur
df_feat_eng$y = as.factor(df_feat_eng$y)



## On cap les valeurs extrémes en les égalant au 1er et dernier centile
fun <- function(x){
  quantiles <- quantile( x, c(.01, .99 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  x
}

for (col in names(df_feat_eng)[!names(df_feat_eng) %in% "y"]) {
  
  df_feat_eng[,col] = fun(df_feat_eng[,col])
}






####################################
####################################
## Division du jeu de données en train/test

# seed
set.seed(100)

# on divise le dataset en train / test et on le standardise
# on utilise la fonction createDataPartition() pour garder les proportions de Y dans le train et le test
datatrain_feat_eng.index <- createDataPartition(df_feat_eng$y, p = 0.8, list = FALSE)
datatrain_feat_eng <- df_feat_eng[ datatrain_feat_eng.index,]
datatest_feat_eng  <- df_feat_eng[-datatrain_feat_eng.index,]

## On standardise les variables
preProcValues <- preProcess(datatrain_feat_eng, method = c("center", "scale"))

datatrain_feat_eng <- predict(preProcValues, datatrain_feat_eng)
datatest_feat_eng <- predict(preProcValues, datatest_feat_eng)



######################################
######################################
## Méthodes de sélection de variables

## Backward selection
set.seed(100)

subsets <- seq(40,120,5)

ctrl <- rfeControl(functions=rfFuncs,
                   method = "cv",
                   number = 5,
                   verbose = FALSE)

rfe_model <- rfe(datatrain_feat_eng[ , -which(names(datatrain_feat_eng) %in% c("y"))],
                 datatrain_feat_eng[,"y"],
                 sizes = subsets,
                 metric = "Accuracy",
                 maximize = T,
                 rfeControl = ctrl)


# Sommaire des résultats
print(rfe_model)

# On trace la courbe des résultats
plot(rfe_model, type=c("g", "o"))


## On crée le jeu de données avec les variables sélectionnées - datatrain et datatest
datatrain_feat_eng_rfe = datatrain_feat_eng[,rfe_model$optVariables]
datatrain_feat_eng_rfe = cbind(datatrain_feat_eng_rfe,datatrain_feat_eng$y)
names(datatrain_feat_eng_rfe)[names(datatrain_feat_eng_rfe) == 'datatrain_feat_eng$y'] <- 'y'

datatest_feat_eng_rfe = datatest_feat_eng[,rfe_model$optVariables]
datatest_feat_eng_rfe = cbind(datatest_feat_eng_rfe,datatest_feat_eng$y)
names(datatest_feat_eng_rfe)[names(datatest_feat_eng_rfe) == 'datatest_feat_eng$y'] <- 'y'



##########
# Forêt aléatoire classique avec grid-search
set.seed(100)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           search = "grid",
                           savePredictions="final", 
                           classProbs = FALSE)

RF_tunegrid <- expand.grid(mtry = seq(3,length(rfe_model$optVariables),10),
                           splitrule = c("gini", "extratrees") ,
                           min.node.size = c(1:10))

# Entrainement
model_RF_rfe <- train(y ~ ., data = datatrain_feat_eng_rfe, 
                  method = "ranger",
                  metric = "Accuracy",
                  maximize = T, 
                  trControl = fitControl,
                  tuneGrid = RF_tunegrid)

# Prédictions & performance sur échantillon test
datatest_feat_eng_rfe$prediction_RF_rfe = predict(model_RF_rfe,newdata = datatest_feat_eng_rfe)
acc_RF_rfe = sum(datatest_feat_eng_rfe$prediction_RF_rfe == datatest_feat_eng_rfe$y) / nrow(datatest_feat_eng_rfe) *100
performance[nrow(performance)+1,] = list("RF Feat Eng RFE",acc_RF_rfe)


##############################
##############################
# H2O - AutoML

# On initialise H2O
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain_feat_eng_rfe), y)

# On convertit les jeux de données en H2O frame
datatrain_feat_eng_rfe_h2o = as.h2o(datatrain_feat_eng_rfe)
datatest_feat_eng_rfe_h2o = as.h2o(datatest_feat_eng_rfe)

# En classification la variable réponse est un facteur
datatrain_feat_eng_rfe_h2o[,y] <- as.factor(datatrain_feat_eng_rfe_h2o[,y])
datatest_feat_eng_rfe_h2o[,y] <- as.factor(datatest_feat_eng_rfe_h2o[,y])

# Entrainement par validation croisée
model_automl_h2o_rfe <- h2o.automl(x = x, 
                               y = y, 
                               training_frame = datatrain_feat_eng_rfe_h2o,
                               nfolds =  5,
                               balance_classes = T,
                               max_runtime_secs = 900,
                               #max_models = 30,
                               seed = 100)

model_automl_h2o_rfe@leaderboard

# Prédictions & performance sur échantillon test
prediction_automl_rfe_h2o = h2o.predict(model_automl_h2o_rfe@leader, datatest_feat_eng_rfe_h2o)
datatest_feat_eng_rfe_h2o = h2o.cbind(datatest_feat_eng_rfe_h2o,prediction_automl_rfe_h2o[, "predict"])
datatest_feat_eng_rfe_automl = as.data.frame(datatest_feat_eng_rfe_h2o)
names(datatest_feat_eng_rfe_automl)[names(datatest_feat_eng_rfe_automl) == 'predict'] <- 'prediction_automl_rfe_h2o'
acc_automl_rfe_h2o = sum(datatest_feat_eng_rfe_automl$prediction_automl_rfe_h2o == datatest_feat_eng_rfe_automl$y) / nrow(datatest_feat_eng_rfe_automl) *100
performance[nrow(performance)+1,] = list("H2O AutoML RFE",acc_automl_rfe_h2o )


h2o.shutdown(prompt = F)







#####################################
#####################################
# H2O - ensemble bis - No Grid Search
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain_feat_eng_rfe), y)

# On convertit les jeux de données en H2O frame
datatrain_feat_eng_rfe_h2o = as.h2o(datatrain_feat_eng_rfe)
datatest_feat_eng_rfe_h2o = as.h2o(datatest_feat_eng_rfe)

# Variable réponse en facteur
datatrain_feat_eng_rfe_h2o[,y] <- as.factor(datatrain_feat_eng_rfe_h2o[,y])
datatest_feat_eng_rfe_h2o[,y] <- as.factor(datatest_feat_eng_rfe_h2o[,y])


# Nombre de folds en validation croisée
nfolds <- 5

# GBM + RF

# Entrainement GBM
my_gbm_rfe <- h2o.gbm(x = x,
                  y = y,
                  training_frame = datatrain_feat_eng_rfe_h2o,
                  distribution = "multinomial",
                  balance_classes = T,
                  seed =100,
                  max_runtime_secs = 900,
                  fold_assignment = 'Stratified',
                  nfolds = nfolds,
                  keep_cross_validation_predictions = TRUE)

# Entrainement RF
my_rf_rfe <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = datatrain_feat_eng_rfe_h2o,
                          ntrees = 100,
                          balance_classes = T,
                          nfolds = nfolds,
                          max_runtime_secs = 900,
                          fold_assignment = 'Stratified',
                          keep_cross_validation_predictions = TRUE,
                          seed = 100)

# Entrainement modéle stacked
ensemble_rfe <- h2o.stackedEnsemble(x = x,
                                y = y,
                                seed = 100,
                                training_frame = datatrain_feat_eng_rfe_h2o,
                                model_id = "my_ensemble_rfe",
                                base_models = list(my_gbm_rfe, my_rf_rfe))


# Prédictions & performance sur échantillon test
prediction_ensemble_rfe_h2o <- h2o.predict(ensemble_rfe, newdata = datatest_feat_eng_rfe_h2o)
datatest_feat_eng_rfe_h2o = h2o.cbind(datatest_feat_eng_rfe_h2o,prediction_ensemble_rfe_h2o[, "predict"])
datatest_feat_eng_rfe_ensemble = as.data.frame(datatest_feat_eng_rfe_h2o)
names(datatest_feat_eng_rfe_ensemble)[names(datatest_feat_eng_rfe_ensemble) == 'predict'] <- 'prediction_ensemble_rfe_h2o'
acc_ensemble_rfe_h2o = sum(datatest_feat_eng_rfe_ensemble$prediction_ensemble_rfe_h2o == datatest_feat_eng_rfe_ensemble$y) / nrow(datatest_feat_eng_rfe_ensemble) *100
performance[nrow(performance)+1,] = list("Ensemble H2O RFE",acc_ensemble_rfe_h2o)

h2o.shutdown(prompt = F)





################################################
################################################
################################################
################################################
## Variable importance - Random Forest - H2O

h2o.shutdown(prompt = F)
h2o.init()

y <- "y"
x <- setdiff(names(datatrain_feat_eng), y)

# On convertit les jeux de données en H2O frame
datatrain_feat_eng_h2o = as.h2o(datatrain_feat_eng)
datatest_feat_eng_h2o = as.h2o(datatest_feat_eng)

# For classification, response should be a factor
datatrain_feat_eng_h2o[,y] <- as.factor(datatrain_feat_eng_h2o[,y])
datatest_feat_eng_h2o[,y] <- as.factor(datatest_feat_eng_h2o[,y])



# Hyperparamètres de la forét aléatoire 

seed_opts = 1234
max_models_opts = 20
ntrees_opts = seq(50, 1000, 50)
min_rows_opts = c(1, 5, 10, 20, 50, 100)
sample_rate_opts = seq(0.3, 1, 0.05)
col_sample_rate_per_tree_opts = seq(0.3, 1, 0.05)
nbins_opts = seq(20, 100, 10)
mtries_opts = seq(10, length(names(datatrain))-1, 10) 


hyper_params_rf_varSelect = list(ntrees = ntrees_opts,
                                 min_rows = min_rows_opts,
                                 sample_rate = sample_rate_opts,
                                 col_sample_rate_per_tree = col_sample_rate_per_tree_opts,
                                 nbins = nbins_opts,
                                 mtries = mtries_opts)
  
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_models = max_models_opts,
                        seed = seed_opts,
                        max_runtime_secs= 300,
                        stopping_tolerance = 0.001,
                        stopping_rounds = 5)  
  
# Train random grid random forest
RF_grid <- h2o.grid("drf", x = x, y = y,
                      grid_id = "RF_grid",
                      training_frame = datatrain_feat_eng_h2o,
                      balance_classes=T,
                      seed =100,
                      nfolds=5,
                      fold_assignment = 'Stratified',
                      hyper_params = hyper_params_rf_varSelect,
                      search_criteria = search_criteria)


RF_grid_perf <- h2o.getGrid(grid_id = "RF_grid",sort_by = "accuracy",decreasing = TRUE)
print(RF_grid_perf)


# Choix du meilleur modéle par rapport é l'accuracy
best_RF <- h2o.getModel(RF_grid_perf@model_ids[[1]])

# On crée un data frame avec les variables
var_imp_RF = as.data.frame(h2o.varimp(best_RF))

h2o.shutdown(prompt = F)


# On trace le bar plot de l'importance des variables : on garde celle dont l'importance est >= 10%
var_imp_RF %>%
  select(variable,scaled_importance)%>%
  filter(scaled_importance >= 0.3406487) %>%
  ggplot(aes(x= reorder(variable,  scaled_importance), y=scaled_importance)) +
  geom_bar(stat = "identity")+
  coord_flip()
  

# On garde les variables dont l'importance est >= 10% (arbitraire)
temp = subset(var_imp_RF,scaled_importance >= 0.1)
 

## On crée le jeu de données avec les variables sélectionnées - datatrain et datatest
datatrain_feat_eng_var_imp_rf = datatrain_feat_eng[,temp$variable]
datatrain_feat_eng_var_imp_rf = cbind(datatrain_feat_eng_var_imp_rf,datatrain_feat_eng$y)
names(datatrain_feat_eng_var_imp_rf)[names(datatrain_feat_eng_var_imp_rf) == 'datatrain_feat_eng$y'] <- 'y'

datatest_feat_eng_var_imp_rf = datatest_feat_eng[,temp$variable]
datatest_feat_eng_var_imp_rf = cbind(datatest_feat_eng_var_imp_rf,datatest_feat_eng$y)
names(datatest_feat_eng_var_imp_rf)[names(datatest_feat_eng_var_imp_rf) == 'datatest_feat_eng$y'] <- 'y'



##########
# Random Forest classique avec grid search
set.seed(100)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           search = "grid",
                           savePredictions="final", 
                           classProbs = FALSE)

RF_tunegrid <- expand.grid(mtry = seq(2,length(temp$variable),5),
                           splitrule = c("gini", "extratrees") ,
                           min.node.size = c(1:10))

# Entrainement
model_RF_var_imp <- train(y ~ ., data = datatrain_feat_eng_var_imp_rf, 
                          method = "ranger",
                          metric = "Accuracy",
                          maximize = T, 
                          trControl = fitControl,
                          tuneGrid = RF_tunegrid)


# Prédictions & performance sur échantillon test
datatest_feat_eng_var_imp_rf$prediction_RF_var_imp_rf = predict(model_RF_var_imp,newdata = datatest_feat_eng_var_imp_rf)
acc_RF_var_imp_rf = sum(datatest_feat_eng_var_imp_rf$prediction_RF_var_imp_rf == datatest_feat_eng_var_imp_rf$y) / nrow(datatest_feat_eng_var_imp_rf) *100
performance[nrow(performance)+1,] = list("RF Feat Eng VAR IMP RF",acc_RF_var_imp_rf)





##############################
##############################
# H2O - AutoML

# On initialise H2O
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain_feat_eng_var_imp_rf), y)

# On convertit les jeux de données en H2O frame
datatrain_feat_eng_var_imp_rf_h2o = as.h2o(datatrain_feat_eng_var_imp_rf)
datatest_feat_eng_var_imp_rf_h2o = as.h2o(datatest_feat_eng_var_imp_rf)

# En classification la variable réponse est un facteur
datatrain_feat_eng_var_imp_rf_h2o[,y] <- as.factor(datatrain_feat_eng_var_imp_rf_h2o[,y])
datatest_feat_eng_var_imp_rf_h2o[,y] <- as.factor(datatest_feat_eng_var_imp_rf_h2o[,y])

# Entrainement par validation croisée
model_automl_h2o_var_imp_rf <- h2o.automl(x = x, 
                                   y = y, 
                                   training_frame = datatrain_feat_eng_var_imp_rf_h2o,
                                   nfolds =  5,
                                   balance_classes = T,
                                   max_runtime_secs = 900,
                                   seed = 100)

model_automl_h2o_var_imp_rf@leaderboard

# Prédictions & performance sur échantillon test
prediction_automl_var_imp_rf_h2o = h2o.predict(model_automl_h2o_var_imp_rf@leader, datatest_feat_eng_var_imp_rf_h2o)
datatest_feat_eng_var_imp_rf_h2o = h2o.cbind(datatest_feat_eng_var_imp_rf_h2o,prediction_automl_var_imp_rf_h2o[, "predict"])
datatest_feat_eng_var_imp_rf_automl = as.data.frame(datatest_feat_eng_var_imp_rf_h2o)
names(datatest_feat_eng_var_imp_rf_automl)[names(datatest_feat_eng_var_imp_rf_automl) == 'predict'] <- 'prediction_automl_var_imp_rf_h2o'
acc_automl_var_imp_rf_h2o = sum(datatest_feat_eng_var_imp_rf_automl$prediction_automl_var_imp_rf_h2o == datatest_feat_eng_var_imp_rf_automl$y) / nrow(datatest_feat_eng_var_imp_rf_automl) *100
performance[nrow(performance)+1,] = list("H2O AutoML VAR IMP RF",acc_automl_var_imp_rf_h2o )


h2o.shutdown(prompt = F)








#####################################
#####################################
# H2O - méthode d'ensemble forêt aléatoire + GBM -Sans Grid Search
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain_feat_eng_var_imp_rf), y)

# On convertit les jeux de données en H2O frame
datatrain_feat_eng_var_imp_rf_h2o = as.h2o(datatrain_feat_eng_var_imp_rf)
datatest_feat_eng_var_imp_rf_h2o = as.h2o(datatest_feat_eng_var_imp_rf)

# Variable réponse en facteur
datatrain_feat_eng_var_imp_rf_h2o[,y] <- as.factor(datatrain_feat_eng_var_imp_rf_h2o[,y])
datatest_feat_eng_var_imp_rf_h2o[,y] <- as.factor(datatest_feat_eng_var_imp_rf_h2o[,y])


# Nombre de folds en validation croisée
nfolds <- 5

# GBM + RF

# Entrainement GBM
my_gbm_var_imp_rf <- h2o.gbm(x = x,
                      y = y,
                      training_frame = datatrain_feat_eng_var_imp_rf_h2o,
                      distribution = "multinomial",
                      balance_classes = T,
                      seed =100,
                      max_runtime_secs = 900,
                      fold_assignment = 'Stratified',
                      nfolds = nfolds,
                      keep_cross_validation_predictions = TRUE)

# Entrainement RF
my_rf_var_imp_rf <- h2o.randomForest(x = x,
                              y = y,
                              training_frame = datatrain_feat_eng_var_imp_rf_h2o,
                              ntrees = 100,
                              balance_classes = T,
                              nfolds = nfolds,
                              max_runtime_secs = 900,
                              fold_assignment = 'Stratified',
                              keep_cross_validation_predictions = TRUE,
                              seed = 100)

# Entrainement modéle stacked
ensemble_var_imp_rf <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    seed = 100,
                                    training_frame = datatrain_feat_eng_var_imp_rf_h2o,
                                    model_id = "my_ensemble_var_imp_rf",
                                    base_models = list(my_gbm_var_imp_rf, my_rf_var_imp_rf))


# Prédictions & performance sur échantillon test
prediction_ensemble_var_imp_rf_h2o <- h2o.predict(ensemble_var_imp_rf, newdata = datatest_feat_eng_var_imp_rf_h2o)
datatest_feat_eng_var_imp_rf_h2o = h2o.cbind(datatest_feat_eng_var_imp_rf_h2o,prediction_ensemble_var_imp_rf_h2o[, "predict"])
datatest_feat_eng_var_imp_rf_ensemble = as.data.frame(datatest_feat_eng_var_imp_rf_h2o)
names(datatest_feat_eng_var_imp_rf_ensemble)[names(datatest_feat_eng_var_imp_rf_ensemble) == 'predict'] <- 'prediction_ensemble_var_imp_rf_h2o_bis'
acc_ensemble_var_imp_rf_h2o = sum(datatest_feat_eng_var_imp_rf_ensemble$prediction_ensemble_var_imp_rf_h2o == datatest_feat_eng_var_imp_rf_ensemble$y) / nrow(datatest_feat_eng_var_imp_rf_ensemble) *100
performance[nrow(performance)+1,] = list("Ensemble H2O VAR IMP RF",acc_ensemble_var_imp_rf_h2o)

h2o.shutdown(prompt = F)




















#############################################
#############################################
# RéDUCTION DE LA DIMENSIONALITé - PCA

datpca=as.matrix(df_feat_eng[ , -which(names(df_feat_eng) %in% 'y')])
datpca=apply(datpca,2,scale)
pca=prcomp(datpca)

# On garde les composantes principales qui expliquent plus de 90% de la variance
pca_temp = as.data.frame(get_eig(pca))
df_pca = as.data.frame(pca$x[,1:25])
df_pca = cbind(df_pca,df_feat_eng$y)
names(df_pca)[names(df_pca) == 'df_feat_eng$y'] <- 'y'


##### On divise le jeu de données en train / test
# seed
set.seed(100)

# on divise le dataset en train / test et on le standardise
# on utilise la fonction createDataPartition() pour garder les proportions de Y dans le train et le test
datatrain_pca.index <- createDataPartition(df_pca$y, p = 0.8, list = FALSE)
datatrain_pca <- df_pca[ datatrain_pca.index,]
datatest_pca  <- df_pca[-datatrain_pca.index,]




##########
# Random Forest classique avec grid search
set.seed(100)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           search = "grid",
                           savePredictions="final", 
                           classProbs = FALSE)

RF_tunegrid <- expand.grid(mtry = seq(2,length(datatrain_pca)-1,1),
                           splitrule = c("gini", "extratrees") ,
                           min.node.size = c(1:10))

# Entrainement
model_RF_pca <- train(y ~ ., data = datatrain_pca, 
                      method = "ranger",
                      metric = "Accuracy",
                      maximize = T, 
                      trControl = fitControl,
                      tuneGrid = RF_tunegrid)

# Prédictions & performance sur échantillon test
datatest_pca$prediction_RF_pca = predict(model_RF_pca,newdata = datatest_pca)
acc_RF_pca = sum(datatest_pca$prediction_RF_pca == datatest_pca$y) / nrow(datatest_pca) *100
performance[nrow(performance)+1,] = list("RF PCA",acc_RF_pca)





##############################
##############################
# H2O - AutoML

# On initialise H2O
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain_pca), y)

# On convertit les jeux de données en H2O frame
datatrain_pca_h2o = as.h2o(datatrain_pca)
datatest_pca_h2o = as.h2o(datatest_pca)

# En classification la variable réponse est un facteur
datatrain_pca_h2o[,y] <- as.factor(datatrain_pca_h2o[,y])
datatest_pca_h2o[,y] <- as.factor(datatest_pca_h2o[,y])

# Entrainement par validation croisée
model_automl_h2o_pca <- h2o.automl(x = x, 
                                   y = y, 
                                   training_frame = datatrain_pca_h2o,
                                   nfolds =  5,
                                   balance_classes = T,
                                   max_runtime_secs = 900,
                                   #max_models = 30,
                                   seed = 100)

model_automl_h2o_pca@leaderboard

# Prédictions & performance sur échantillon test
prediction_automl_pca_h2o = h2o.predict(model_automl_h2o_pca@leader, datatest_pca_h2o)
datatest_pca_h2o = h2o.cbind(datatest_pca_h2o,prediction_automl_pca_h2o[, "predict"])
datatest_pca_automl = as.data.frame(datatest_pca_h2o)
names(datatest_pca_automl)[names(datatest_pca_automl) == 'predict'] <- 'prediction_automl_pca_h2o'
acc_automl_pca_h2o = sum(datatest_pca_automl$prediction_automl_pca_h2o == datatest_pca_automl$y) / nrow(datatest_pca_automl) *100
performance[nrow(performance)+1,] = list("H2O AutoML PCA",acc_automl_pca_h2o )


h2o.shutdown(prompt = F)







#####################################
#####################################
# H2O - méthode d'ensemble forêt aléatoire + GBM - Sans Grid Search
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(datatrain_pca), y)

# On convertit les jeux de données en H2O frame
datatrain_pca_h2o = as.h2o(datatrain_pca)
datatest_pca_h2o = as.h2o(datatest_pca)

# Variable réponse en facteur
datatrain_pca_h2o[,y] <- as.factor(datatrain_pca_h2o[,y])
datatest_pca_h2o[,y] <- as.factor(datatest_pca_h2o[,y])


# Nombre de folds en validation croisée
nfolds <- 5

# GBM + RF

# Entrainement GBM
my_gbm_pca <- h2o.gbm(x = x,
                      y = y,
                      training_frame = datatrain_pca_h2o,
                      distribution = "multinomial",
                      balance_classes = T,
                      seed =100,
                      max_runtime_secs = 900,
                      fold_assignment = 'Stratified',
                      nfolds = nfolds,
                      keep_cross_validation_predictions = TRUE)

# Entrainement RF
my_rf_pca <- h2o.randomForest(x = x,
                              y = y,
                              training_frame = datatrain_pca_h2o,
                              ntrees = 100,
                              balance_classes = T,
                              nfolds = nfolds,
                              max_runtime_secs = 900,
                              fold_assignment = 'Stratified',
                              keep_cross_validation_predictions = TRUE,
                              seed = 100)

# Entrainement modéle stacked
ensemble_pca <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    seed = 100,
                                    training_frame = datatrain_pca_h2o,
                                    model_id = "my_ensemble_pca",
                                    base_models = list(my_gbm_pca, my_rf_pca))


# Prédictions & performance sur échantillon test
prediction_ensemble_pca_h2o <- h2o.predict(ensemble_pca, newdata = datatest_pca_h2o)
datatest_pca_h2o = h2o.cbind(datatest_pca_h2o,prediction_ensemble_pca_h2o[, "predict"])
datatest_pca_ensemble = as.data.frame(datatest_pca_h2o)
names(datatest_pca_ensemble)[names(datatest_pca_ensemble) == 'predict'] <- 'prediction_ensemble_pca_h2o'
acc_ensemble_pca_h2o = sum(datatest_pca_ensemble$prediction_ensemble_pca_h2o == datatest_pca_ensemble$y) / nrow(datatest_pca_ensemble) *100
performance[nrow(performance)+1,] = list("Ensemble H2O PCA",acc_ensemble_pca_h2o)

h2o.shutdown(prompt = F)






###################################################################################################################
###################################################################################################################
###################################################################################################################
## On entraine le meilleur modéle : H2O AutoML sur l'ensemble du jeu de données.

## On centre et on standardise les variables
preProcValues <- preProcess(df[ , -which(names(df) %in% 'y')], 
                            method = c("center", "scale"))

df_final <- predict(preProcValues, df[ , -which(names(df) %in% 'y')])



## On ajoute la variable réponse au jeu de données
df_final = cbind(df_final,df$y)
names(df_final)[names(df_final) == 'df$y'] <- 'y'


##########
# H2O - AutoML

# On initialise H2O
h2o.shutdown(prompt = F)
h2o.init()

# Variables indépendantes et variable réponse
y <- "y"
x <- setdiff(names(df_final), y)

# On convertit les jeux de données en H2O frame
df_final_h2o = as.h2o(df_final)


# En classification la variable réponse est un facteur
df_final_h2o[,y] <- as.factor(df_final_h2o[,y])

# Entrainement par validation croisée
model_automl_h2o_final <- h2o.automl(x = x, 
                               y = y, 
                               training_frame = df_final_h2o,
                               nfolds =  5,
                               balance_classes = T,
                               max_runtime_secs = 3600,
                               seed = 100)

model_automl_h2o_final@leaderboard


