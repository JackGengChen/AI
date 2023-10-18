from deepNN_model import deepNN_model
from utils import layer_sizes, load_planar_dataset, plot_decision_boundary
import numpy as np
import matplotlib.pyplot as plt
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X, Y = load_planar_dataset()
    n_x, n_h, n_y = layer_sizes(X, Y)
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    # [2,50, 30, 20, 1]
    model = deepNN_model([2,100, 100, 100, 50, 20, 1], 1.25, num_iterations = 50000, print_cost = True)
    costs = model.train(X, Y)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    print(X.shape)
    predictions = model.predict(X)
    print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')

    # Plot the decision boundary
    plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))

