[![FYS-STK4155_Project2](https://github.com/emmastorberg/FYS-STK4155_Project2/actions/workflows/pytest.yml/badge.svg)](https://github.com/emmastorberg/FYS-STK4155_Project2/actions/workflows/pytest.yml)
# FYS-STK4155_Project2

´

    ├── data                            # Contains other datasets used for testing our network
    ├── data_analysis
    │   ├── load_data.py                  # Loads breast_cancer_data and explores distributions and correlation with target
    │   ├── logreg.yaml                   # Config for sklearn logreg
    │   ├── test_sklearn_logreg.py        # Runs sklearn logistic regression to compare later on and get feature importance
    │   └── utils.py              
    ├── docs                            # includes bibliography and latex files from overleaf
    ├── figures                         # Images of the plotted results
    ├── relevant_weekly_exercises       # Not in the project delivery
    ├── breast_cancer.ipynb              
    ├── gradient_descent_examples.ipynb
    ├── logistic_regression.py
    ├── main.py
    ├── neural_network.py
    ├── plot.ipynb
    ├── plots.ipynb
    ├── plotting.py
    ├── main.py                        
    ├── project2_report.pdf             # The report to be submitted for this project
    ├── tests.py
    ├── torch_plot.py                  # Classification with pytorch 
    └── requirements.txt                # Python libraries needed for running the code. Install with ´pip -r requirements.txt´
´
already tested:
- the tests in tests.py
- correct ouput shape
- predict has values between 0 and 1
- gradient descent plain/stochasitc works for linear regression
- neural network (with gradient descent) works for iris data + XOR dataset
- gradients with correct shape
