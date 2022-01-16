data <- read.csv ("train.csv")
dim(data)

#===============================================================================
#We have 42,000 labeled samples available, and each sample has 784 features, 
#which means each digit image has 784 (28 * 28) pixels. Take a look at the label 
#and the first 5 features (pixels) for each of the first 6 data samples:
#===============================================================================

head(data[1:6])

#===============================================================================
#The target label ranging from 0 to 9 denotes 10 digits:
#===============================================================================

unique(unlist(data[1]))

#===============================================================================
#The pixel variable ranges from 0 to 255, representing the brightness 
#of the pixel,
#for example 0 means black and 255 stands for white:
#===============================================================================

min(data[2:785])
max(data[2:785])

#===============================================================================
#Now let's take a look at two samples, first, the fourth image:
#===============================================================================
sample_4 <- matrix(as.numeric(data[4,-1]), nrow = 28, byrow = TRUE)
image(sample_4, col = grey.colors(255))

#===============================================================================
#Where we reshaped the feature vector of length 784 into a matrix of 28 * 28.
#Second, the 7th image:
#===============================================================================

sample_7 <- matrix(as.numeric(data[7,-1]), nrow = 28, byrow = TRUE)
image(sample_7, col = grey.colors(255))

#===============================================================================
#We noticed that the images are rotated 90 degrees to the left. To better view 
#the images, 
#a rotation of 90 degrees clockwise is required. 
#We simply need to reserve elements in each column of an image matrix:
#===============================================================================

rotate <- function(x) t(apply(x, 2, rev))

#===============================================================================
#Now visualize the rotated images:
#===============================================================================

image(rotate(sample_4), col = grey.colors(255))
#===============================================================================
image(rotate(sample_7), col = grey.colors(255))

#===============================================================================
#After viewing what the data and images behind look like, 
#we do more exploratory analysis on the labels and features. 
#Firstly, because it is a classification problem
#we inspect whether the classes from the data are balanced or unbalanced as a 
#good practice.
#But before doing so, 
#we should transform the label from integer to factor:
#===============================================================================

# Transform target variable "label" from integer to factor, in order to perform classification
is.factor(data$label)

data$label <- as.factor(data$label)
is.factor(data$label)

#===============================================================================
#Now, we can summarize the label distribution in counts:
#===============================================================================

summary(data$label)

#===============================================================================
#Or combined with proportion (%):
#===============================================================================

proportion <- prop.table(table(data$label)) * 100
cbind(count=table(data$label), proportion=proportion)

#===============================================================================
#Classes are balanced.
#Now, we explore the distribution of features,the pixels. 
#As an example, we take the 4 pixels from the central 2*2 block 
#(that is, pixel376, pixel377, pixel404, and pixel405) in each image 
#and display the histogram for each of the 9 digits:
#===============================================================================

central_block <- c("pixel376", "pixel377", "pixel404", "pixel405")
par(mfrow=c(2, 2))
for(i in 1:9) { hist(c(as.matrix(data[data$label==i, central_block])),
         main=sprintf("Histogram for digit %d", i),
         xlab="Pixel value")}
#===============================================================================
#The brightness of central pixels is distributed differently among 9 digits. 
#For instance, the majority of the central pixels are bright,
#as digit 8 is usually written in a way that strokes go through the center;
#while digit 7 is not written in this way, hence most of the central pixels are dark.
#Pixels taken from other positions can also be distinctly distributed among different digits.
#The exploratory analysis we just conducted helps move us forward with building 
#classification models based on pixels.
#===============================================================================

#We first split the dataset into two subsets for training and testing 
#respectively using the caret package
#===============================================================================
library(caret)

#===============================================================================
#We first split the data into two partitions, 
#75% for training and 25% for testing, using the createDataPartition function:
#===============================================================================

set.seed(42)
train_perc = 0.75
train_index <- createDataPartition(data$label, p=train_perc, list=FALSE)

#===============================================================================

data_train <- data[train_index,]
data_test <- data[-train_index,]

#===============================================================================
#Then, we implement the multinomial logistic regression model using the nnet package.
#The package contains functions for feed-forward single-layer neural networks as
#well as multinomial logistic regression models. 
#More details can be found in 
# https://cran.r-project.org/web/packages/nnet/nnet.pdf:
#===============================================================================

library(nnet)

model_lr <- multinom(label ~ ., data=data_train, MaxNWts=10000,
                     decay=5e-3, maxit=100)
prediction_lr <- predict(model_lr, data_test, type = "class")
prediction_lr[1:5]
data_test$label[1:5]

#We can also obtain the confusion matrix by:

cm_lr = table(data_test$label, prediction_lr)
cm_lr

#And the classification accuracy:

accuracy_lr = mean(prediction_lr == data_test$label)
accuracy_lr

#Going from logistic regression to single-layer neural networks
#Again, we use the nnet package to implement our single-layer network:

model_nn <- nnet(label ~ ., data=data_train, size=50, maxit=300, MaxNWts=100000, decay=1e-4)


prediction_nn <- predict(model_nn, data_test, type = "class")
cm_nn = table(data_test$label, prediction_nn)
cm_nn
#===============================================================================
accuracy_nn = mean(prediction_nn == data_test$label)
accuracy_nn
#===============================================================================
#MXNet (https://mxnet.incubator.apache.org/) is a deep learning framework 
#that supports programming languages include R, Scala, Python, Julia, C++, and Perl.
#===============================================================================

cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
if (!require("mxnet"))
  install.packages("mxnet")
#===============================================================================

require(mxnet)
data_train <- data.matrix(data_train)
data_train.x <- data_train[,-1]
data_train.x <- t(data_train.x/255)
data_train.y <- data_train[,1]

#===============================================================================
#Now that the training dataset is ready,
#we can start constructing the network by defining its architecture as follows:
#===============================================================================

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

#===============================================================================
#After building the bone, it is time to train the model.
#We can choose our computation device, CPU and/or GPU-here is a CPU example:
#===============================================================================

devices <- mx.cpu()

#===============================================================================

mx.set.seed(42)
model_dnn <- mx.model.FeedForward.create(softmax, X=data_train.x,
                                           y=data_train.y, ctx=devices, num.round=30, array.batch.size=100,
                                           learning.rate=0.01, momentum=0.9, eval.metric=mx.metric.accuracy,
                                           initializer=mx.init.uniform(0.1),
                                           epoch.end.callback=mx.callback.log.train.metric(100))
#===============================================================================

data_test.x <- data_test[,-1]
data_test.x <- t(data_test.x/255)

prob_dnn <- predict(model_dnn, data_test.x)
prediction_dnn <- max.col(t(prob_dnn)) - 1
cm_dnn = table(data_test$label, prediction_dnn)

#===============================================================================

cm_dnn
accuracy_dnn = mean(prediction_dnn == data_test$label)
accuracy_dnn
#===============================================================================
#Extracting richer representation with CNNs
#===============================================================================

conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5),
                                   num_filter=20)
act1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=act1, pool_type="max",
                             kernel=c(2,2), stride=c(2,2))

#===============================================================================

conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5),
                                   num_filter=50)
act2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=act2, pool_type="max",
                             kernel=c(2,2), stride=c(2,2))

#===============================================================================
flatten <- mx.symbol.Flatten(data=pool2)

#===============================================================================
# first fully connected layer
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
act3 <- mx.symbol.Activation(data=fc1, act_type="relu")
# second fully connected layer
fc2 <- mx.symbol.FullyConnected(data=act3, num_hidden=10)

#===============================================================================

# softmax output
softmax <- mx.symbol.SoftmaxOutput(data=fc2, name="sm")

#===============================================================================

train.array <- data_train.x
dim(train.array) <- c(28, 28, 1, ncol(data_train.x))
mx.set.seed(42)
model_cnn <- mx.model.FeedForward.create(softmax, X=train.array,
                                           y=data_train.y, ctx=devices, num.round=30,
                                           array.batch.size=100, learning.rate=0.05,
                                           momentum=0.9, wd=0.00001,
                                           eval.metric=mx.metric.accuracy,
                                           epoch.end.callback=mx.callback.log.train.metric(100))

#===============================================================================

test.array <- data_test.x
dim(test.array) <- c(28, 28, 1, ncol(data_test.x))

#===============================================================================

prob_cnn <- predict(model_cnn, test.array)
prediction_cnn <- max.col(t(prob_cnn)) - 1
cm_cnn = table(data_test$label, prediction_cnn)
cm_cnn

#===============================================================================
accuracy_cnn = mean(prediction_cnn == data_test$label)
accuracy_cnn
#===============================================================================
#Our CNN model further boosts the accuracy to close to 99%
#===============================================================================
graph.viz(model_cnn$symbol)

#===============================================================================
data_test.y <- data_test[,1]
logger <- mx.metric.logger$new()
model_cnn <- mx.model.FeedForward.create(softmax, X=train.array,
                                           y=data_train.y,eval.data=list(data=test.array,
                                                                         label=data_test.y), ctx=devices, num.round=30,
                                           array.batch.size=100, learning.rate=0.05,
                                           momentum=0.9, wd=0.00001,eval.metric=
                                             mx.metric.accuracy, epoch.end.callback =
                                             mx.callback.log.train.metric(1, logger))
#===============================================================================
#We can get the performance on the training set after each round of training:
#===============================================================================

logger$train

#===============================================================================
#As well as the performance on the testing set after each round of training:
#===============================================================================

logger$eval

#===============================================================================
#The learning curve can be visualized by the following codes:
#===============================================================================

plot(logger$train,type="l",col="red", ann=FALSE)
lines(logger$eval,type="l", col="blue")
title(main="Learning curve")
title(xlab="Iterations")
title(ylab="Accuary")
legend(20, 0.5, c("training","testing"), cex=0.8,
         col=c("red","blue"), pch=21:22, lty=1:2)

#===============================================================================

par(mfrow=c(1,2))
test_1 <- matrix(as.numeric(data_test[1,-1]), nrow = 28,
                   byrow = TRUE)
image(rotate(test_1), col = grey.colors(255))
test_2 <- matrix(as.numeric(data_test[2,-1]), nrow = 28,
                   byrow = TRUE)
image(rotate(test_2), col = grey.colors(255))

#===============================================================================

layerss_for_viz <- mx.symbol.Group(mx.symbol.Group(c(conv1, act1, pool1, conv2, act2, pool2, fc1, fc2)))
executor <- mx.simple.bind(symbol=layerss_for_viz,
                             data=dim(test.array), ctx=mx.cpu())
#===============================================================================

mx.exec.update.arg.arrays(executor, model_cnn$arg.params, match.name=TRUE)
mx.exec.update.aux.arrays(executor, model_cnn$aux.params,
                            match.name=TRUE)
#===============================================================================

mx.exec.update.arg.arrays(executor,
                            list(data=mx.nd.array(test.array)), match.name=TRUE)
mx.exec.forward(executor, is.train=FALSE)

#===============================================================================
names(executor$ref.outputs)

#===============================================================================

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$activation15_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n',
          col=grey.colors(255)
          )
   }

#===============================================================================

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$activation15_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n',
          col=grey.colors(255)
          )
}

#===============================================================================

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <-as.array
  (executor$ref.outputs$pooling10_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n',
          col=grey.colors(255)
          )
  }
par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$pooling10_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n',
          col=grey.colors(255)
          )
  }

#===============================================================================

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$convolution11_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n',
          col=grey.colors(255)
          )
  }
par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$convolution11_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n',
          col=grey.colors(255)
          )
}

#===============================================================================

validation_perc = 0.4
validation_index <- createDataPartition(data_test.y, p=validation_perc, list=FALSE)

validation.array <- test.array[, , , validation_index]
dim(validation.array) <- c(28, 28, 1,
                             length(validation.array[1,1,]))
data_validation.y <- data_test.y[validation_index]
final_test.array <- test.array[, , , -validation_index]
dim(final_test.array) <- c(28, 28, 1,
                             length(final_test.array[1,1,]))
data_final_test.y <- data_test.y[-validation_index]

#===============================================================================

mx.callback.early.stop <- function(eval.metric) {
  function(iteration, nbatch, env, verbose) {
    if (!is.null(env$metric)) {
      if (!is.null(eval.metric)) {
        result <- env$metric$get(env$eval.metric)
        if (result$value >= eval.metric) {
           return(FALSE)
           }
         }
       }
     return(TRUE)
     }
}

#===============================================================================

model_cnn_earlystop <- mx.model.FeedForward.create(softmax,
                                                     X=train.array, y=data_train.y,
                                                     eval.data=list(data=validation.array, label=data_validation.y),
                                                     ctx=devices, num.round=30, array.batch.size=100,
                                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                                     eval.metric=mx.metric.accuracy,
                                                     epoch.end.callback = mx.callback.early.stop(0.985))

#===============================================================================

prob_cnn <- predict(model_cnn_earlystop, final_test.array)
prediction_cnn <- max.col(t(prob_cnn)) - 1
cm_cnn = table(data_final_test.y, prediction_cnn)
cm_cnn

accuracy_cnn = mean(prediction_cnn == data_final_test.y)
accuracy_cnn


#===============================================================================
















