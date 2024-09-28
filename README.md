Good morning! 
In this repo, i'm going to detect and classify which images contain brain tumor or not. 
Talking a little bit about the method, 

First, we collected data and preprocessed it. (the data was collected from Kaggle and Google) 
-> We realized that the pics are almost in reactangular shape. So that, when we scale it for model learning, it will create wrong answers. 
+ so we developed a tool which could locate the brain, draw a line convering all the brain and cropped the picture into square (the brain needs to be in the central part of the pic) 
+  so the pictures were converted into square shape which produced much more better results when resizing. 

Secondly, we trained the modal using CNN (Converlution Neuron Network) 
+ we created 3 2D-layers for the modal
+ used activation relu - softmax
+ dense to get the wanted result.

Last, we used K-fold validation method to divide train-test folds and runned training process. 
-> the result was quite bad at first. 
-> to solve this problem, we decided to use early-stopping and learning-rate-adaptive-reschedule to increase the accuracy. 
=> finally, we got 99,97% for accuracy. 

This project followed a repo of a fellow, thanks to him. 
All the process I mentioned above is the improvement of mine. 
Thanks for your time! 
