import numpy as np
import matplotlib.pyplot as plt

from GradientDescent import Plain, Stochastic

plt.rcParams.update({
        # Matplotlib style settings similar to seaborn's default style
        "axes.facecolor": "#eaeaf2",
        "axes.edgecolor": "white",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 1,
        "axes.axisbelow": True,
        "xtick.color": "gray",
        "ytick.color": "gray",

        # Additional stylistic settings
        "figure.facecolor": "white",
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.fancybox": True,
        "legend.edgecolor": 'lightgray',
    })

#Test this a few times for a few different fixed_lr
def mse_func_of_epochs(epochs: np.ndarray, 
                           mse: np.ndarray, 
                           mse_mom: np.ndarray, 
                           analytical: np.ndarray, 
                           fixed_lr: float, 
                           sgd: bool = True,
                           filename: str | None = None) -> None:
    
    if sgd:
        labels = ["SGD",
                  "SGD with Momentum", 
                  rf"MSE of OLS Using Stochastic Gradient Descent with $\eta_0 = {fixed_lr}$"]
    else:
        labels = ["PGD", 
                 "PGD with Momentum", 
                 rf"MSE of OLS Using Plain Gradient Descent with $\eta = {fixed_lr}$"]
    
    plt.plot(epochs, mse, label=labels[0])
    plt.plot(epochs, mse_mom, label=labels[1])

    plt.plot(epochs, analytical, label="Analytical Solution")
    plt.title(labels[2])
    if sgd:
        plt.xlabel("Epochs")
    else:
        plt.xlabel("Iterations")
    plt.yscale("log")
    plt.ylabel("MSE")
    plt.legend()

    if filename is not None:
        if sgd:
            plt.savefig(filename)
        else:
            plt.savefig(filename)
    else:
        plt.show()

def three_stacked_subplots(epochs: np.ndarray, 
                           sgd: np.ndarray,
                           sgd_mom: np.ndarray,
                           adagrad: np.ndarray, 
                           adagrad_mom: np.ndarray, 
                           rms: np.ndarray, 
                           rms_mom: np.ndarray, 
                           adam: np.ndarray,
                           lr: float,
                           plot_lr: bool = True,
                           save: bool = False):
    
    # Create a figure and an array of 3 subplots stacked vertically
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 9))

    # Plot on the first subplot
    axs[0].plot(epochs, sgd, label="SGD")
    axs[0].plot(epochs, sgd_mom, label="SGD with Momentum")
    axs[0].plot(epochs, adagrad, label="AdaGrad")
    axs[0].plot(epochs, adagrad_mom, label="AdaGrad with Momentum")
    axs[0].legend()
    if plot_lr:
        axs[0].set_ylabel(r"$\eta$")
    else:
        axs[0].set_ylabel("MSE")

    # Plot on the second subplot
    axs[1].plot(epochs, sgd, label="SGD")
    axs[1].plot(epochs, sgd_mom, label="SGD with Momentum")
    axs[1].plot(epochs, rms, label="RMSProp")
    axs[1].plot(epochs, rms_mom, label="RMSProp with Momentum")
    axs[1].legend()
    if plot_lr:
        axs[1].set_ylabel(r"$\eta$")
    else:
        axs[1].set_ylabel("MSE")

    # Plot on the third subplot
    axs[2].plot(epochs, sgd, label="SGD")
    axs[2].plot(epochs, sgd_mom, label="SGD with Momentum")
    axs[2].plot(epochs, adam, label="Adam")
    axs[2].legend()
    if plot_lr:
        axs[2].set_ylabel(r"$\eta$")
    else:
        axs[2].set_ylabel("MSE")
    axs[2].set_xlabel("Epochs")

    if plot_lr:
        fig.suptitle(fr"Learning Rates of Adaptive Methods ($\eta = {lr}$)")
    else:
        fig.suptitle("MSE of Adaptive Methods")

    # Adjust layout to avoid overlap
    plt.tight_layout()

    if save:
        if plot_lr:
            plt.savefig("learningsratesGD.png")
        else:
            plt.savefig("MSEGD.png")
    else:
        plt.show()

def gridsearch(parameter1, parameter2):
    ...

def mse_linreg_comparison(
        epochs: np.ndarray, 
        ols: np.ndarray, 
        ridge: np.ndarray, 
        nn: np.ndarray, 
        sklearn: np.ndarray,
        save: bool = False):
    
    plt.plot(epochs, ols, label = "OLS")
    plt.plot(epochs, ridge, label = "Ridge Regression")
    plt.plot(epochs, nn, label = "Neural Network")
    plt.plot(epochs, sklearn, label = "Scikit-Learn")

    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.title("Comparison of Optimal Methods for Polynomial Approximation")

    if save:
        plt.savefig("numericalprediction.png")
    else:
        plt.show()

def accuracy_with_different_activationfuncs_in_hidden_layers(
        epochs: np.ndarray, 
        sigmoid: np.ndarray, 
        relu: np.ndarray, 
        leakyrelu: np.ndarray,
        save: bool = False,
        linear: bool = True
        ) -> None:

    plt.plot(epochs, sigmoid, label = "Sigmoid")
    plt.plot(epochs, relu, label = "ReLU")
    plt.plot(epochs, leakyrelu, label = "Leaky ReLU")
    plt.legend()

    plt.xlabel("Epochs")

    if linear:
        plt.ylabel("MSE")
        plt.title("Different Activations of Hidden Layers in Numerical Prediction Task")

        if save:
            plt.savefig("activationfunctions_cost_numpred.png")
        else:
            plt.show()
    else:
        plt.ylabel("Accuracy")
        plt.title("Different Activations of Hidden Layers in Classification Task")
        
        if save:
            plt.savefig("figures/all_plots/activationfunctions_cost.png")
        else:
            plt.show()

def prediction_comparison(
        input: np.ndarray, 
        target: np.ndarray, 
        ols_pred: np.ndarray, 
        ridge_pred: np.ndarray,
        nn_pred: np.ndarray, 
        sklearn_pred: np.ndarray,
        save: bool = False
        ) -> None:

    plt.plot(input, target, label = "Exact Solution")
    plt.plot(input, ols_pred, label = "OLS")
    plt.plot(input, ridge_pred, label = "Ridge Regression")
    plt.plot(input, nn_pred, label = "Neural Network")
    plt.plot(input, sklearn_pred, label = "Scikit-Learn")

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Polynomial Approximation with Various Methods")
    
    if save:
        plt.savefig("predictioncomparison.png")
    else:
        plt.show()
