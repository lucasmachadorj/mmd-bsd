######   TENTATIVA 1   ######
install.packages('neuralnet')
install.packages('tm')

library("neuralnet") 

traininginput <- read.csv(file='/Users/Bianca/Documents/MMD/mmd-bsd/datasets/X_train.csv')
trainingoutput <- read.csv(file='/Users/Bianca/Documents/MMD/mmd-bsd/datasets/y_train_registered.csv')

trainingdata <- cbind(traininginput,trainingoutput)
colnames(trainingdata) <- c('season','holiday','workingday','weather','temp','atemp','humidity','windspeed','year','month','day','hour','weekday','registered')

traininginput$season <- as.factor(traininginput$season)
traininginput$holiday <- as.factor(traininginput$holiday)
traininginput$workingday <- as.factor(traininginput$workingday)
traininginput$day <- as.factor(traininginput$day)
traininginput$month <- as.factor(traininginput$month)
traininginput$year <- as.factor(traininginput$year)
traininginput$weekday <- as.factor(traininginput$weekday)
traininginput$weather <- as.factor(traininginput$weather)
 
#Train the neural network
#Going to have 13 hidden layers
#Threshold is a numeric value specifying the threshold for the partial
#derivatives of the error function as stopping criteria.
form.in <- as.formula('registered~season+holiday+workingday+weather+temp+atemp+humidity+windspeed+year+month+day+hour+weekday') 

sle <- function(actual,predicted) (log(1+actual)-log(1+predicted))^2
msle <- function(actual, predicted) mean(sle(actual,predicted))
rmsle <- function(actual, predicted) sqrt(msle(actual,predicted))

net.bsd <- neuralnet(formula=form.in,data=trainingdata, hidden=5, threshold=0.01)
print(net.bsd)
 
#Plot the neural network
plot(net.bsd)
 
#Test the neural network on some training data
testdata <- read.csv(file='/Users/Bianca/Documents/MMD/mmd-bsd/datasets/X_test.csv')
net.results <- compute(net.bsd, testdata) #Run them through the neural network
resulttestdata <- read.csv(file='/Users/Bianca/Documents/MMD/mmd-bsd/datasets/y_test_registered.csv')

#Lets see what properties net.sqrt has
ls(net.results)
 
#Lets see the results
#print(net.results$net.result)
 
#Lets display a better version of the results
cleanoutput <- cbind(testdata,resulttestdata,
                         as.data.frame(net.results$net.result))
colnames(cleanoutput) <- c("Input","Expected Output","Neural Net Output")
print(cleanoutput)

######   TENTATIVA 2   ######
install.packages('mlbench')
install.packages('caret')

library(mlbench)
traininginput <- read.csv(file='/Users/Bianca/Documents/MMD/mmd-bsd/datasets/X_train.csv')
trainingoutput <- read.csv(file='/Users/Bianca/Documents/MMD/mmd-bsd/datasets/y_train_registered.csv')

trainingdata <- cbind(traininginput,trainingoutput)
colnames(trainingdata) <- c('season','holiday','workingday','weather','temp','atemp','humidity','windspeed','year','month','day','hour','weekday','registered')

require(caret)

mygrid <- expand.grid(.decay=c(0.5, 0.1), .size=c(4,5,6))
nnetfit <- train(registered/50 ~ ., data=trainingdata, method="nnet", maxit=10000, tuneGrid=mygrid, trace=F)
print(nnetfit)