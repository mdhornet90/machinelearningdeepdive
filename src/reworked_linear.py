from data_population_playground import getModel
import matplotlib.pyplot as plt
import numpy as np

def plot_model(model, comparison_idx):
    train_data = np.array(model.train_data)
    sorted_data = train_data[train_data[:,11].argsort()]
    x_data = np.arange(0, len(sorted_data))
    fig, ax1 = plt.subplots()
    ax1.plot(x_data, sorted_data[:,11], 'ro')
    ax1.set_xlabel('sample')
    ax1.set_ylabel('quality', color='r')
    ax1.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    ax2.plot(x_data, sorted_data[:,comparison_idx], 'b-')
    ax2.set_ylabel(model.headers[comparison_idx], color='b')
    ax2.tick_params('y', colors='b')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = getModel('winequality-red.csv')
    for i in range(len(model.train_data[0]) - 1):
        plot_model(model, i)