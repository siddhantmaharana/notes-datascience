# Deep Learning

Deep Learning is nothing but a paradigm of machine learning which has shown incredible promise in the recent years. This is because of the fact that Deep Learning shows great analogy with the functioning of the human brain.

Now although Deep Learning has been around for many years, the major breakthroughs from these techniques came just in the recent years. This is because of two main reasons – 
- The first and foremost, as we saw before, is the increase of data generated through various sources.
- The second is the growth in hardware resources required to run these models. GPUs, which are becoming a requirement to run deep learning models, are multiple times faster and they help us build bigger and deeper deep learning models in comparatively less time than we required previously.

### What is the difference between Deep Learning and Machine Learning?

Data dependencies: The most important difference between deep learning and traditional machine learning is its performance as the scale of data increases. When the data is small, deep learning algorithms don’t perform that well. This is because deep learning algorithms need a large amount of data to understand it perfectly. On the other hand, traditional machine learning algorithms with their handcrafted rules prevail in this scenario. 

Feature engineering: In Machine learning, most of the applied features need to be identified by an expert and then hand-coded as per the domain and data type. Deep learning algorithms try to learn high-level features from data. Like, Convolutional NN will try to learn low-level features such as edges and lines in early layers then parts of faces of people and then high-level representation of a face.

Interpretability: Machine learning algorithms are interpretable and thus used in industry to explain the reasoning behind any predictions or causations.

### Why are deep networks better than shallow ones?

There are studies which say that both shallow and deep networks can fit at any function, but as deep networks have several hidden layers often of different types so they are able to build or extract better features than shallow models with fewer parameters.

### What is a cost function?

A cost function is a measure of the accuracy of the neural network with respect to given training sample and expected output. It is a single value, nonvector as it gives the performance of the neural network as a whole. It can be calculated as below Mean Squared Error function:-
MSE=1n∑i=0n(Y^i–Yi)^2
Where Y^ and desired value Y is what we want to minimize

### What is a gradient descent?

Gradient descent is basically an optimization algorithm, which is used to learn the value of parameters that minimizes the cost function. It is an iterative algorithm which moves in the direction of steepest descent as defined by the negative of the gradient.

### What is a backpropagation?

Backpropagation is training algorithm used for multilayer neural network. In this method, we move the error from an end of the network to all weights inside the network and thus allowing efficient computation of the gradient. It can be divided into several steps as follows:-

- Forward propagation of training data in order to generate output.
- Then using target value and output value error derivative can be computed with respect to output activation.
- Then we back propagate for computing derivative of error with respect to output activation on previous and continue this for all the hidden layers.
- Using previously calculated derivatives for output and all hidden layers we calculate error derivatives with respect to weights.
- And then we update the weights.

### Explain the following three variants of gradient descent: batch, stochastic and mini-batch?

Stochastic Gradient Descent: Here we use only single training example for calculation of gradient and update parameters.
Batch Gradient Descent: Here we calculate the gradient for the whole dataset and perform the update at each iteration.
Mini-batch Gradient Descent: It’s one of the most popular optimization algorithms. It’s a variant of Stochastic Gradient Descent and here instead of single training example, mini-batch of samples is used.

### What is weight initialization in neural networks?

Weight initialization is one of the very important steps. A bad weight initialization can prevent a network from learning but good weight initialization helps in giving a quicker convergence and a better overall error. Biases can be generally initialized to zero. The rule for setting the weights is to be close to zero without being too small.

### What is the role of the activation function?

The activation function is used to introduce non-linearity into the neural network helping it to learn more complex function. Without which the neural network would be only able to learn linear function which is a linear combination of its input data.

### What Is A Dropout?

Dropout is a regularization technique for reducing overfitting in neural networks. At each training step we randomly drop out (set to zero) set of nodes, thus we create a different model for each training case, all of these models share weights. It’s a form of model averaging.

### CNN
The Conv layer is the building block of a Convolutional Network. The Conv layer consists of a set of learnable filters (such as 5 * 5 * 3, width * height * depth). During the forward pass, we slide (or more precisely, convolve) the filter across the input and compute the dot product. Learning again happens when the network back propagate the error layer by layer.

Initial layers capture low-level features such as angle and edges, while later layers learn a combination of the low-level features and in the previous layers and can therefore represent higher level feature, such as shape and object parts.

### RNN and LSTM
RNN is another paradigm of neural network where we have difference layers of cells, and each cell not only takes as input the cell from the previous layer, but also the previous cell within the same layer. This gives RNN the power to model sequence.

This seems great, but in practice RNN barely works due to exploding/vanishing gradient, which is cause by a series of multiplication of the same matrix. To solve this, we can use a variation of RNN, called long short-term memory (LSTM), which is capable of learning long-term dependencies.

The math behind LSTM can be pretty complicated, but intuitively LSTM introduce

input gate
output gate
forget gate
memory cell (internal state)
LSTM resembles human memory: it forgets old stuff (old internal state * forget gate) and learns from new input (input node * input gate)

