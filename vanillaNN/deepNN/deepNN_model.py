# NN_model
from utils import layer_sizes, sigmoid, relu, sigmoid_backward, relu_backward, epsilon
from layer import Layer
import numpy as np
import tensorflow as tf
class deepNN_model(Layer):
    def __init__(self, layer_dims, learning_rate, num_iterations = 10000, print_cost=False):
        self.parameters = self.initialize_parameters_deep(layer_dims)
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        ### START CODE HERE ### (「 1 line of code)
        Z = np.dot(W,A) + b
        ### END CODE HERE ###

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (「 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b) # This "linear_cache" contains (A_prev, W, b)
            A, activation_cache = sigmoid(Z) # This "activation_cache" contains "Z"
            ### END CODE HERE ###

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (「 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b) # This "linear_cache" contains (A_prev, W, b)
            A, activation_cache = relu(Z) # This "activation_cache" contains "Z"
            ### END CODE HERE ###

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(self.parameters) // 2                  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            ### START CODE HERE ### (「 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], "relu")
            caches.append(cache)
            ### END CODE HERE ###

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (「 2 lines of code)
        AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], "sigmoid")
        caches.append(cache)
        ### END CODE HERE ###

        assert(AL.shape == (1,X.shape[1]))

        return AL, caches

    def clip(self, AL):
        # Get the epsilon value and output data type
        epsilon_value = tf.constant(1e-7, dtype=tf.float64)  # Adjust epsilon value as needed
        output_dtype = tf.float64  # Replace with the actual dtype you need
        _epsilon = tf.convert_to_tensor(epsilon_value, dtype=output_dtype)
        AL = tf.clip_by_value(AL, _epsilon, 1 - _epsilon)
        return AL

    def train(self, X, Y):
        costs = []
        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache"
            AL, caches = self.L_model_forward(X)
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost"
            AL = self.clip(AL)
            cost = self.compute_cost(AL, Y)
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads"
            grads = self.L_model_backward(AL, Y, caches)
            # Update rule for each parameter
            self.update_parameters(grads, self.learning_rate)
            # If print_cost=True, Print the cost every 1000 iterations
            if self.print_cost and i % 100 == 0:
            # if self.print_cost and i == 1:
                print ("Cost after iteration %i: %f" %(i, cost))
                # print ("Paramenters after a after iteration %i: " %(i))
                # print(self.parameters)
                # print(grads)
                costs.append(cost)
        return costs


    # GRADED FUNCTION: compute_cost

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        ### START CODE HERE ### (「 1 lines of code)
        cost = (-1/m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
        ### END CODE HERE ###

        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())

        return cost
    # GRADED FUNCTION: update_parameters





    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        ### START CODE HERE ### (「 2 lines of code)
        A2, cache = self.L_model_forward(X)
        predictions = (A2 > 0.5)
        ### END CODE HERE ###

        return predictions

    def initialize_parameters_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(2)
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            ### START CODE HERE ### (「 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
            ### END CODE HERE ###

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        # print("parameter init:")
        # print(parameters)
        return parameters

    def linear_backward(self, dZ, cache):
        # Here cache is "linear_cache" containing (A_prev, W, b) coming from the forward propagation in the current layer
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        ### START CODE HERE ### (「 3 lines of code)
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        ### END CODE HERE ###

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db


    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            ### START CODE HERE ### (「 1 line of code)
            dZ = relu_backward(dA, activation_cache)
            ### END CODE HERE ###

        elif activation == "sigmoid":
            ### START CODE HERE ### (「 1 line of code)
            dZ = sigmoid_backward(dA, activation_cache)
            ### END CODE HERE ###

        ### START CODE HERE ### (「 1 line of code)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        ### END CODE HERE ###

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L-1] # Last Layer
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        ### END CODE HERE ###

        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            ### START CODE HERE ### (approx. 5 lines)
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            ### END CODE HERE ###

        return grads


    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
        """

        L = len(self.parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (「 3 lines of code)
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        # ### END CODE HERE ###
        # return parameters


