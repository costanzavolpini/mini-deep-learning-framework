# MINI DEEP-LEARNING FRAMEWORK
## Project 2
**EPFL | Deep Learning | EE559**

Project realized in context of the master course EE-559 Deep Learning at EPFL (Summer Semester 2018/2019).

Professor: François **Fleuret**

Students: Francis **Damachi**, Costanza **Volpini**

### DESCRIPTION:
In the last few years, Neural Networks have proved to be one of the most effective solutions to tackle a wide range problems (e.g. image and speech recognition, language processing). The aim of this project is to design a multi-layer perceptron capable using the standard math library and the basic tensor operations of Pytorch.

### CODE STRUCTURE:
- rocket_deepl/core/layers.py: contain class for fully connected layer.
- rocket_deepl/core/activations/relu.py: contain class for non-linear function relu.
- rocket_deepl/core/activations/tanh.py: contain class for non-linear function tanh.
- rocket_deepl/core/losses/l_mse.py: contain class for Mean Square error.
- rocket_deepl/optimizer/sgd.py: contain class for stochastic gradient descent.
- rocket_deepl/module.py: contain abstract class for module.
- rocket_deepl/sequential.py: contain a class that handles different modules. As input it takes a list of layers that composes the neural net.
- rocket_deepl/utils.py: contain compute_nb_errors(), train_model() methods.
- generator_training_test.py: contain function to generate train and target.
- test.py: main code to run a network with two input units, two output units, three hidden layers of 25 units.
- comparison_report.ipynb: jupyter notebook used for comparing model (report purpose).


### TO RUN THE CODE:
From the root of the project: ``` python test.py ```