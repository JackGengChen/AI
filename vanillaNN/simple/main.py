from NN_model import vannila_model
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

    model = vannila_model(n_x, n_h, n_y, 1.2, num_iterations = 20000, print_cost = True)
    parameters = model.train(X, Y)
    print(X.shape)
    predictions = model.predict(X)
    print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')

    # Plot the decision boundary
    plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))

