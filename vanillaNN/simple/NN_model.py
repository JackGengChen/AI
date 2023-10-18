# NN_model
from utils import layer_sizes, sigmoid
from layer import Layer
import numpy as np

class vannila_model(Layer):
    def __init__(self, n_x, n_h, n_y, learning_rate, num_iterations = 10000, print_cost=False):
        self.parameters = self.initialize_parameters(n_x, n_h, n_y)
        self.W1 = self.parameters["W1"]
        self.b1 = self.parameters["b1"]
        self.W2 = self.parameters["W2"]
        self.b2 = self.parameters["b2"]
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output


    def forward_propagation(self, X):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        ### START CODE HERE ### (「 4 lines of code)
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        ### END CODE HERE ###

        # Implement Forward Propagation to calculate A2 (probabilities)
        ### START CODE HERE ### (「 4 lines of code)
        Z1 = np.dot(W1,X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = sigmoid(Z2)
        # print("Z1 shape", Z1.shape)
        ### END CODE HERE ###

        assert(A2.shape == (1, X.shape[1]))

        # Values needed in the backpropagation are stored in "cache". This will be given as an input to the backpropagation
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache

    def backward_propagation(self, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]

        # First, retrieve W1 and W2 from the dictionary "parameters".
        ### START CODE HERE ### (「 2 lines of code)
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
        ### END CODE HERE ###

        # Retrieve also A1 and A2 from dictionary "cache".
        ### START CODE HERE ### (「 2 lines of code)
        A1 = cache["A1"]
        A2 = cache["A2"]
        Z1 = cache["Z1"]
        Z2 = cache["Z2"]
        ### END CODE HERE ###

        # Backward propagation: calculate dW1, db1, dW2, db2.
        ### START CODE HERE ### (「 6 lines of code, corresponding to 6 equations on slide above)
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2,A1.T)
        db2 = (1/m) *(np.sum(dZ2,axis=1,keepdims=True))
        dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1,2))
        dW1 = (1/m) *(np.dot(dZ1,X.T))
        db1 = (1/m) *(np.sum(dZ1, axis=1, keepdims=True))
        ### END CODE HERE ###

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads
    def train(self, X, Y):
        n_x = layer_sizes(X, Y)[0]
        n_y = layer_sizes(X, Y)[2]

        # Initialize parameters
        # parameters = initialize_parameters(n_x, n_h, n_y)
        parameters = self.parameters

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache"
            A2, cache = self.forward_propagation(X)
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost"
            cost = self.compute_cost(A2, Y)
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads"
            grads = self.backward_propagation( cache, X, Y)
            # Update rule for each parameter
            self.parameters = self.update_parameters(self.parameters, grads, self.learning_rate)
            # If print_cost=True, Print the cost every 1000 iterations
            # if self.print_cost and i % 1000 == 0:
            if self.print_cost and i == 1:
            #     print ("Cost after iteration %i: %f" %(i, cost))
                print ("Paramenters after a after iteration %i: " %(i))
                # print(self.parameters)
                print(grads)
        # Returns parameters learnt by the model. They can then be used to predict output
        return parameters


    # GRADED FUNCTION: compute_cost

    def compute_cost(self, A2, Y):
        """
        Computes the cross-entropy cost given in equation (13)

        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        [Note that the parameters argument is not used in this function,
        but the auto-grader currently expects this parameter.
        Future version of this notebook will fix both the notebook
        and the auto-grader so that `parameters` is not needed.
        For now, please include `parameters` in the function signature,
        and also when invoking this function.]

        Returns:
        cost -- cross-entropy cost given equation (13)

        """

        m = Y.shape[1] # number of example

        # Compute the cross-entropy cost
        ### START CODE HERE ### (「 2 lines of code)
        logprobs = logprobs = np.multiply(Y ,np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
        cost = (-1/m) * np.sum(logprobs)
        ### END CODE HERE ###

        cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
        # E.g., turns [[17]] into 17
        assert(isinstance(cost, float))

        return cost

    # GRADED FUNCTION: update_parameters

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Updates parameters using the gradient descent update rule given above

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
        # Retrieve each parameter from the dictionary "parameters"
        ### START CODE HERE ### (「 4 lines of code)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        ### END CODE HERE ###

        # Retrieve each gradient from the dictionary "grads"
        ### START CODE HERE ### (「 4 lines of code)
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        ## END CODE HERE ###

        # Update rule for each parameter
        ### START CODE HERE ### (「 4 lines of code)
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        ### END CODE HERE ###

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.

        ### START CODE HERE ### (「 4 lines of code)
        W1 = np.random.randn(n_h,n_x) * 0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h) * 0.01
        b2 = np.zeros((n_y,1))
        ### END CODE HERE ###

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        # print("parameter init:")
        # print(parameters)
        return parameters
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
        A2, cache = self.forward_propagation(X)
        predictions = (A2 > 0.5)
        ### END CODE HERE ###

        return predictions
