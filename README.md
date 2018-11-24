# Poker-Hand-Predictor
University of Hull Deep learning Winter School Group Project  
Feed-forward network developed in Python with Keras + TensorFlow backend used to predict poker outcomes when given a hand of 5 cards. Achieves ~90% accuracy. Uses the 'Poker Hand Data Set' from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Poker+Hand)


Network Architecture:  
  2 Hidden layers, 512 nodes in each  
  Sigmoid activation function on hidden layers, softmax used on output layer  
  2 dropout layers used, although have little effect given the application  
  Loss function: binary crossentropy  
  Optimiser: RMSprop  
  

Designed and developed with:  
  Aidan Fray  
  Adam Tunnicliffe  
  Lydia Maisanda 
