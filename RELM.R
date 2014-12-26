calcActivationFunction <- function(ActivationFunction,tempH){
  switch(lower(ActivationFunction),
         "sig" = ,
         "sigmoid" =  1 ./ (1 + exp(-tempH)),# Sigmoid 
         "sin" = ,
         "sine" =  sin(tempH),              # Sine
         "hardlim" =  double(hardlim(tempH)),# Hard Limit
         "tribas" =  tribas(tempH), # Triangular basis function
         "radbas" =  radbas(tempH) # Radial basis function
         # More activation functions can be added here
  )
}

#' @param obs observed values
#' @param pred predicted values
calcMultiLabelAccuracy(obs,pred){
  label_index_expected <- apply(obs,1,which.max)
  label_index_actual <- apply(pred,1,which.max)
  TrainingAccuracy <- sum(label_index_expected == label_index_actual)/
                    length(label_index_expected)
}

#' R version of Extreme Learning Machine ,for ELM with random hidden nodes and random hidden neurons 
#' @usage relm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
#'
#' @param TrainingData_File     - Filename of training data set
#' @param TestingData_File      - Filename of testing data set
#' @param Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
#' @param NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
#' @param ActivationFunction    - Type of activation function:
#'                           'sig' for Sigmoidal function
#'                           'sin' for Sine function
#'                           'hardlim' for Hardlim function
#'                           'tribas' for Triangular basis function
#'                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
#'
#' @return a relm object
#' @describeIn TrainingTime          - Time (seconds) spent on training ELM
#' @describeIn TestingTime           - Time (seconds) spent on predicting ALL testing data
#' @describeIn TrainingAccuracy      - Training accuracy: 
#'                           RMSE for regression or correct classification rate for classification
#' @describeIn TestingAccuracy       - Testing accuracy: 
#'                           RMSE for regression or correct classification rate for classification
#'
#' @details  MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
#' FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
#' neurons; neuron 5 has the highest output means input belongs to 5-th class
#' @references See http://www.ntu.edu.sg/home/egbhuang/index.html
#' @examples
#' Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy]  <-  elm('sinc_train', 'sinc_test', 0, 20, 'sig')
#' Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
#'

relm <- function(TrainingData_File, TestingData_File,sep <- "\t", Elm_Type, 
                 NumberofHiddenNeurons, ActivationFunction){

  
  ########### Macro definition
  REGRESSION <- 0
  CLASSIFIER <- 1
  
  ########### Load training dataset
  train_data <- read.table(TrainingData_File,sep = sep)
  class.index <- ncol(train_data)
  y <- train_data[,class.index]  # class label index
  X <- train_data[,-class.index]
  rm(train_data) #   Release raw training data array

  ########### Load testing dataset
  test_data <- read.table(TestingData_File,sep = sep)
  TV.y <- test_data[,class.index]
  TV.X <- test_data[,class.index]
  rm(test_data) #   Release raw testing data array
  
  NumberofTrainingData <- nrow(y)
  NumberofTestingData <- nrow(y)
  NumberofInputNeurons <- class.index-1
  
  if (Elm_Type == CLASSIFIER){
    
    ############ Preprocessing the data of classification(multi-classification)
    label=unique(c(unique(y),unique(TV.y)) #   Find and save in 'label' class label from training and testing data sets
    
    number_class=length(label)
    NumberofOutputNeurons=number_class
    
    multi.label.matrix <- diag(x=1,NumberofOutputNeurons)
    
    ########## Processing the targets of training 
    temp_y <- multi.label.matrix[y,]
    y=temp_y*2-1
    
    ########## Processing the targets of testing
    temp_TV_y <- multi.label.matrix[TV.y,]
    TV.y=temp_TV_y*2-1
    
  }
  
  ########### Calculate weights & biases
  start_time_train=Sys.time()
  
  ########### Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
  InputWeight <- matrix(runif(NumberofHiddenNeurons*NumberofHiddenNeurons),
                        NumberofHiddenNeurons)*2-1
  BiasofHiddenNeurons=runif(NumberofHiddenNeurons)
  tempH=X*InputWeight
  rm(X)                                            #   Release input of training data 
  tempH=tempH+BiasofHiddenNeurons ;
  
  ########### Calculate hidden neuron output matrix H
  H = calcActivationFunction(ActivationFunction,tempH)
  
  rm(tempH) #   Release the temparary array for calculation of hidden neuron output matrix H
  
  ########### Calculate output weights OutputWeight (beta_i)
  OutputWeight=ginv(H) * y;                        # implementation without regularization factor //refer to 2006 Neurocomputing paper
  #OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   # faster method 1 //refer to 2012 IEEE TSMC-B paper
  #implementation; one can set regularizaiton factor C properly in classification applications 
  #OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      # faster method 2 //refer to 2012 IEEE TSMC-B paper
  #implementation; one can set regularizaiton factor C properly in classification applications
  
  #If you use faster methods or kernel method, PLEASE CITE in your paper properly: 
  
  #Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010. 
  
  end_time_train=Sys.time();
  TrainingTime=end_time_train-start_time_train        #   Calculate CPU time (seconds) spent for training ELM
  
  Y=H * OutputWeight   #   Y: the actual output of the training data
  rm(H)
  
  ########### Calculate the output of testing input
  start_time_test=Sys.time()
  tempH_test=TV.X * InputWeight
  rm(TV.P)            #   Release input of testing data             
  tempH_test=tempH_test + BiasofHiddenNeurons;
  H_test = calcActivationFunction(ActivationFunction,tempH_test)
  TY=H_test * OutputWeight                       #   TY: the actual output of the testing data
  end_time_test=Sys.time();
  TestingTime=end_time_test-start_time_test           #   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
  
  if (Elm_Type == REGRESSION){
    TrainingAccuracy=RMSE(Y,y)#   Calculate training accuracy (RMSE) for regression case
    TestingAccuracy=RMSE(TY,TV.y)#   Calculate testing accuracy (RMSE) for regression case
  }
  
  if (Elm_Type == CLASSIFIER){
    ########## Calculate training & testing classification accuracy
    
    TrainingAccuracy <- calcMultiLabelAccuracy(y,Y)
    TestingAccuracy <- calcMultiLabelAccuracy(TV.y,TY)
  }
  list(predictedValue = TY,
       TrainingTime = TrainingTime,
       TestingTime = TestingTime,
       TrainingAccuracy = TrainingAccuracy,
       TestingAccuracy = TestingAccuracy
       )

}