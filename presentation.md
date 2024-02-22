# Group5

## Challenge
our task is brain tumor segmentation

There are 2 choice to make it: 
1. cut the brain into slices
2. use 3D operations: e.g. 3D convolution, which is similar to 2D convolution but much more time-consuming
   
reason: the size of tumor is much smaller than the brain, and if you cut the brain in a particular position, it's hard to cut exactly where the tumor is.

## U-Net
we use the unet whose the encoder is consist of convolution layers, and  decoder:conv-transpose, symetric, the structure looks like letter U.

there're 2 main advantage of unet: 
1. symrtric structure: 
2. skip-connection: **ppt**


## Modify
In the early stage of project, we scratch a unet on our own, but encountered problems like gradient vanishing and the net is hard to convengence. 

We modified our network by referring to Fabian's work in 2018. If you have heard about the famous Res-Net, you may be familiar with this residual connection.

Another place we modify is replacing the relu function with the LeakyRelu, which is not zero when input is minus.

After all of that, we solve the problem of gradient vanishing.

## Data Augmentation
we apply two kinds of data augmentation: rotation and scaling,

before we feed the image into model, we **ppt**. 

in this process, some background noises will be generated, so **ppt**.

finally, we apply a normalization before training to balance the parameters at the both side of the unet, since the output is 0 or 1 but the input is 0 to positive infinity.

## Dataset
here is our training process, the blueline is the train loss and the orange line is the validate loss, as you can see, the model converges very fast at the first 10 epoch, learn steady, and than finally, overfit itself. 

the fast convergence and steady learning can be attributed to two factors: the use of residual connections, and the use of a scheduler.

as you can see in this figure, we use a scheduler to adjust the learning rate through the training process, high at the beginning and low in the end.

## Loss
Here is our result
**ppt**
the prediction is nearly perfect, and our model achieve a 92% dice score on the training set, 87% on the validate set and 84% on the final test set.

## Improvement
we do another experiment to see if our data augmentation really matters. Here is the result. As you can see in this picture **ppt**, when we comparing the result at the same epoch, the model with data augmentation perform better.

## Conclusion
In conclusion, we scatch and modify a unet structure, design a data augmentation process, and train the model with the enhanced data and a scheduler.

Here is the end! thanks for listening!