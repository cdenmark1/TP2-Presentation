
######################################################################################################################
#################################  Porto Seguro Safe Driver Prediction competition ##################################
#################################     By: Xiaohui Ling & Christian Denmark        ###################################                              
#####################################################################################################################
rm(list=ls())

#Let's set the seed for reproducibility
set.seed(123)

library(dplyr)
train <- read.csv(file='train.csv')
summary(train)
str(train)
# First, let's check for missing values!
sum(is.na(train))

# Missing data in this dataset is encoded as -1 rather than the standard NA.
# We want to convert these -1s to NAs so that any missing categorical 
# data does not get encoded as an additional factor level.
train[train == -1] <- NA
sum(is.na(train))

# Now, let's trun the categorial variable into factors
# and binary factors into logical values
cat_vars <- names(train)[grepl('_cat$', names(train))]
print('those are categorical variables:')
print(cat_vars)

train <- train %>%
  mutate_at(.vars = cat_vars, .funs = as.factor)
summary(train)
sum(is.na(train))

############
# Heat map #
############
library(corrplot)
par(mfrow = c(1, 1))
train %>%
  select(-starts_with("ps_calc"), -ps_ind_10_bin, -ps_ind_11_bin, -ps_car_10_cat, -id) %>%
  mutate_at(vars(ends_with("cat")), funs(as.integer)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.integer)) %>%
  mutate(target = as.integer(target)) %>%
  cor(use="complete.obs", method = "spearman") %>%
  corrplot(type="lower", tl.col = "black",  diag=FALSE)

# One hot encode the factor variables
# Here, we transform a categorical variable, or variables, to a format that works better with both
# classification and regression algorithms

# ! Notice that model.matrix() automatically ignores the rows with missing values or in this case, NAs !

# Since this data set has a lot of missing values, we believe that deleting these records outright would 
#affect the integrity of the data set. So, instead of deleting them, we decided to apply and use 
#the mice package to discover the patten of missing values and do some imputation.

trainmm <- model.matrix(~ . - 1, data = train)
str(trainmm)

# Making a train index
train_index <- sample(c(TRUE, FALSE), replace = TRUE, size = nrow(trainmm), prob = c(0.2, 0.8))

# Split the data according to the train index
training <- as.data.frame(trainmm[train_index, ])
testing <- as.data.frame(trainmm[!train_index, ])
summary(training)

# Let's find any linear combos in features
# install.packages('caret')
library(caret)
lin_comb <- findLinearCombos(training)
lin_comb
# Take the set difference of feature names and linear combos
d <- setdiff(seq(1:ncol(training)), lin_comb$remove)
# Remove the linear combo columns
training <- training[, d]

# Without this line of code, there will be a warning message of: 'Rank deficient fit'. 
training <- training[, setdiff(names(training), 'ps_ind_02_cat4')]

#################################################
###################  Logistic Regression ########
################################################# 

# Here we use a logistic regression model on the training data. Family set to binomial thus link=logit.
logmod <- glm(target ~ . - id, data = training, family = binomial(link = 'logit'))

# Here, we make predictions on the test set
preds <- predict(logmod, newdata = testing, type = "response")
summary(preds)

logistic.YesNo = ifelse (preds > .05, "Yes" , "No" )
cm_logistic = table(logistic.YesNo, testing$target)
cm_logistic
(errorrate = 1 - (cm_logistic[1,1] + cm_logistic[2,2])/ sum(cm_logistic))

# Plot the histogram of the predictions
data.frame(preds = preds) %>%
  ggplot(aes(x = preds)) + 
  geom_histogram(bins = 50, fill = 'grey50') +
  labs(title = 'Histogram of Predictions') +
  theme_bw()

# Let's print the range of the predictions
print(round(range(preds),2))

# Here, we print the median of the predictions
print(median(preds))

#################################################
###################  Roc plot ###################
################################################# 
library(pROC)
test_prob = predict(logmod, newdata = testing, type = "response")
test_roc = roc(testing$target ~ test_prob, plot = TRUE, print.auc = TRUE)
# A good ROC curve will has a high AUC. Ideally, we want high sensitivity (True positive rate) while also having low specificity, 
#(False positive rate).Thus, a very good ROC curve hugs the upper left hand corner of the chart. The AUC we got from this model 
#is only 0.585, which is not great. Although, this model is slightly better than just flipping a coin with random chance.

#################################################
###################  Boosting ###################
################################################# 
library(gbm)
boost=gbm(training$target~.,data=training,distribution="bernoulli",n.trees=300,interaction.depth=2)
pred_probs <- predict(object = boost, newdata = testing, n.trees = 300, type = 'response')
summary(boost)
par(mfrow=c(1,2))
plot(boost)
pred_labels <- ifelse(pred_probs>0.04,"Yes","No")
CM <- (table(ActualValues=testing$target, Predictions=pred_labels))
(Error <- 1 - (CM[1,1] + CM[2,2])/sum(CM))
#Here, we can see that the overall error rate is 43.26%. That means that our boosting model is okay. It is not as good as the ROC model 
#but it is accurate 56.27% of the time.
