# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:17:56 2019

@author: lasse
"""

# Multilayer perceptron with 1 hidden layer. 
import numpy as np
import math


# Making a class neuron which holds all the information for each neuron
class neuron:
    
    def __init__(self, neuron_type):
        
        self.neuron_type = neuron_type
#        if neuron_type == 'input':
#            self.activation_function = 'identity'
#        if neuron_type =='hidden' or neuron_type == 'output':
#            self.activation_function == 'sigmoid'
        if self.neuron_type == 'bias':
            self.activation = 1.0
        
    
    # Function to initialize the weights coming in to the neuron
    def initialize_weights(self, n_input_neurons):
        
        # Initialzing weights to be between -0.3 and 0.3
        weights = np.random.uniform(low=-0.3, high=0.3, size= n_input_neurons)
        # Checks if any weights are exactly 0, changes them to 0.1 (very arbitrarily) if they are
        for i in range(np.size(weights)):
            if weights[i] == 0:
                weights[i] = 0.1
        
        self.weights = weights
        
        # Setting up for weight change at t-1 for use in adding momentum
        weight_change_t_minus_one = []
        for i in range(np.size(weights)):
            weight_change_t_minus_one.append(0)
        
        self.weights_change_t_minus_one = weight_change_t_minus_one
    
    # Calculate activation of the neuron
    def calculate_activation(self, inputs):
        # If input neuron, activation is just equal to the input
        if self.neuron_type == 'input':
            self.activation = inputs
        # Using sigmoid activation function for hidden and output layers
        if self.neuron_type == 'hidden' or self.neuron_type == 'output':
            net_input = sum(inputs * self.weights)
            activation = 1 / (1 + math.exp(-net_input))
            
            self.net_input = net_input
            self.activation = activation
        
        # If the neuron is a bias neuron activation should simply always be 1
        if self.neuron_type == 'bias':
            self.activation = 1.0
        

            
    # Calculate delta of the neuron        
    def calculate_delta(self, target, unit_n = None):
        
        # First calculating the derivative of the activation function
        # Easy to add possibility to change to different activation functions
        activation_derivative = self.activation*(1-self.activation)
        
        # Formula for output neurons
        if self.neuron_type == 'output':

            self.delta = (target - self.activation)* activation_derivative
        
            # Calculating delta * weights for use in hidden layer
            self.delta_times_weight = self.delta * self.weights
            
            # Besides calculating delta, output neurons should also calculate the error 
            self.squared_error = (target - self.activation)**2
            
            
        if self.neuron_type == 'hidden' or self.neuron_type == 'bias':
            
            # Calculating the sum of the errors for the output units
            next_layer_delta_times_weight = []
            for d in range(np.size(target)):
                error_times_weight = target[d].delta_times_weight[unit_n] * target[d].weights[unit_n]
                next_layer_delta_times_weight.append(error_times_weight)
                
            sum_delta_times_weight = sum(next_layer_delta_times_weight)
            
            # Calculating delta
            self.delta = activation_derivative * sum_delta_times_weight
            
            
    
    # Updates the weights coming in to a neuron with the delta rule
    def update_weight(self, activation, eta, momentum):
            
       for w in range(np.size(self.weights)): 
           weight_change = eta * self.delta * activation[w] + momentum * self.weights_change_t_minus_one[w]
           self.weights_change_t_minus_one[w] = weight_change
           self.weights[w] += weight_change

    # For offline learning, calculates the change in weights, but doesn't apply it               
#    def calculate_weight_change(self, activation, eta, momentum):
#        
#        for w in range(np.size(self.weights)): 
#           weight_change = eta * self.delta * activation[w] + momentum * self.weights_change_t_minus_one[w]
# 


# ------------------------------------------------------------------------------------------

# Class to store information

class network:
    
    def __init__(self, input_units, hidden_units, output_units):
        
        self.input = input_units
        self.hidden = hidden_units
        self.output = output_units




# -------------------------------------------------------------------------------------------


# Print activation and weights for a given input
def test_performance(network, test_input, target = "not specified"):

    # Setting activation for the input units and saving activations in a variable
    input_activations =  []
    for u in range(np.size(network.input)-1):
        network.input[u].calculate_activation(test_input[u])
        input_activations.append(network.input[u].activation)
        print("Activation for input unit {} is: {}".format(u, network.input[u].activation))
        
    print("\n")
    # Adding activation for the bias unit to the list
    input_activations.append(network.input[-1].activation)
    # Saving the activations as an array for easier calculations
    input_activations = np.array(input_activations)
        
    
    # Calculating activation of the hidden units
    hidden_activations = []
    for u in range(network.hidden.size):
        network.hidden[u].calculate_activation(input_activations)
        hidden_activations.append(network.hidden[u].activation)
        
    # Printing activations for the units in the hidden layer (not printing activation for bias, since it's just 1)    
    for u in range(network.hidden.size - 1):
        print("Weights coming in to hidden unit {} are {}".format(u, network.hidden[u].weights))        
        print("Activation for hidden unit {} is {}".format(u, network.hidden[u].activation))
     
    print("\n")
    hidden_activations = np.array(hidden_activations)
             
    
    # For all neurons in output layer, calculate activation, and calculate delta.
    output_activations = []
    for u in range(network.output.size):
        network.output[u].calculate_activation(hidden_activations)
        print("Weights coming in to output unit {} are {}".format(u, network.output[u].weights))
        print("Activation for output unit {} is {}".format(u, network.output[u].activation))
        output_activations.append(network.output[u].activation)

    print("\nWith the input {}, output is {}".format(test_input, output_activations))
    print("Target was {}".format(target))




# --------------------------------------------------------------------------

def train_nn(data, target, params):

    # Reading param file and saving to variables
    with open(params, 'r') as f:
        args = f.read().split("\n")
         
    
    n_input = int(args[0])
    n_hidden = int(args[1])
    n_output = int(args[2])
    eta = float(args[3])
    momentum = float(args[4])
    error_criterion = float(args[5])
    
    # Reading input file and removing empty entries in the list if any (a \n at the end of document makes an empty entry)
    with open(data, 'r') as f:
        in_pairs = f.read().split('\n')
      
      
    
    in_pairs = list(filter(None, in_pairs))
    
    # Splitting the pairs into paired groups
    split_pairs = []
    for i in range(len(in_pairs)):
        one_pair = in_pairs[i].split(" ")
        split_pairs.append(one_pair)
    
    # Turning into a numpy array
    d = np.array(split_pairs)
    d = d.astype(np.float)
    
        
    # Same thing for teaching input
    with open(target, 'r') as f:
        target = f.read().split('\n')
    
    target = list(filter(None, target))
    
    target_pairs = []
    for i in range(len(target)):
        one_pair = target[i].split(" ")
        target_pairs.append(one_pair)
    
    t = np.array(target_pairs)
    t = t.astype(np.float)
    
    # ----------------------------------------------------------------------------------------
    
    
    # Setting up network architecture
    
    # Creating array of input units for as many units as specified in param.txt + a bias unit
    inputs = []
    for n in range(n_input):
        inputs.append(neuron('input'))
    inputs.append(neuron('bias'))
    inputs = np.array(inputs)
    
    
    # Making an array of the hidden layer; repeating for as many units specified + a bias unit
    hidden = []
    for n in range(n_hidden):
        hidden.append(neuron('hidden'))
    hidden.append(neuron('bias'))
    hidden = np.array(hidden)
    
    # Initializing weights for the hidden layer
    for u in range(hidden.size):
        hidden[u].initialize_weights(inputs.size)
    
    
    
    # Creating array of output units
    outputs = []
    for n in range(n_output):
        outputs.append(neuron('output'))
    outputs = np.array(outputs) 
    
    # Initializing weights for links to the output layer
    for u in range(outputs.size):
        outputs[u].initialize_weights(hidden.size)
    
    
    # Initializing population error with an arbitrarily high value
    population_error = 10000.0
    n_epochs = 0
    
    # Looping for each training input until error is below threshold
    while population_error > error_criterion:
    
        for p in range(len(d)):
            
            # for online learning, randomizing the order the data is presented in     
            index = np.random.choice(d.shape[0], len(d), replace = False)  
    
            # Setting activation for the input units (first training pair) and saving activations in a variable
            input_activations =  []
            for u in range(n_input):
                inputs[u].calculate_activation(d[index[p],u])
                input_activations.append(inputs[u].activation)
            
            # Adding activation for the bias unit to the list
            input_activations.append(inputs[-1].activation)
            # Saving the activations as an array for easier calculations
            input_activations = np.array(input_activations)
            
            
            # Calculating activation of the units in the hidden layer
            hidden_activations = []
            for u in range(hidden.size):
                hidden[u].calculate_activation(input_activations)
                hidden_activations.append(hidden[u].activation)
            
            hidden_activations = np.array(hidden_activations)
            
                     
            
            # Variables to store activation and delta
    #        outputs_activation = []
    #        outputs_delta = []
            p_squared_error = []
            # For all neurons in output layer, calculate activation, and calculate delta.
            for u in range(outputs.size):
                outputs[u].calculate_activation(hidden_activations)
                outputs[u].calculate_delta(t[index[p],u])
                
    #            outputs_activation.append(outputs[u].activation)
    #            outputs_delta.append(outputs[u].delta)
                p_squared_error.append(outputs[u].squared_error)
            
            p_sum_squared_error = sum(p_squared_error)
    
            '''
            FOR ONLINE LEARNING
            ''' 
            # Update weights coming to the output(s)
            for u in range(outputs.size):
                outputs[u].update_weight(hidden_activations, eta, momentum)
               
            '''
            FOR ONLINE LEARNING
            '''
            
            # Calculate delta for hidden unit(s) and update incoming weights
            for u in range(hidden.size):
                hidden[u].calculate_delta(outputs, u)
                
                #FOR ONLINE LEARNING
                hidden[u].update_weight(input_activations, eta, momentum)
                
        population_error = p_sum_squared_error / (n_output * len(d))
        n_epochs += 1
        if n_epochs % 100 == 0: 
            print("Number of epochs: {}".format(n_epochs))
            print("Pop error is {}".format(population_error))
    
    print('\nTraining done.\nNumber of epochs: {}, population error: {}'.format(n_epochs, population_error))
    output = network(inputs, hidden, outputs)
    return(output)
            
        
    
    

tester = train_nn('in.txt', 'teach.txt', 'param.txt')


with open('in.txt', 'r') as f:
    in_pairs = f.read().split('\n')
 
  

in_pairs = list(filter(None, in_pairs))

# Splitting the pairs into paired groups
split_pairs = []
for i in range(len(in_pairs)):
    one_pair = in_pairs[i].split(" ")
    split_pairs.append(one_pair)

# Turning into a numpy array
d = np.array(split_pairs)
d = d.astype(np.float)

    
# Same thing for teaching input
with open('teach.txt', 'r') as f:
    target = f.read().split('\n')

target = list(filter(None, target))

target_pairs = []
for i in range(len(target)):
    one_pair = target[i].split(" ")
    target_pairs.append(one_pair)

t = np.array(target_pairs)
t = t.astype(np.float)


test_input = d[0]
test_output = t[0]    

for i in range(len(d)):
    test_performance(tester, d[i], t[i])

test_performance(tester, test_input, test_output)


"""
WHAT TO TEST?
    - DIFFERENT MOMENTUM VALUES
    - DIFFERENT VALUES FOR INITIALIZING WEIGHTS (UNIFORM, GAUSSIAN..)
    - DIFFERENT LEARNING CONSTANTS
    - DIFFERENT ACTIVATION FUNCTIONS? (RELU? https://jamesmccaffrey.wordpress.com/2017/06/23/two-ways-to-deal-with-the-derivative-of-the-relu-function/)
    - DIFFERENT RULES OF THUMB FOR NUMBER OF HIDDEN UNITS
    - DIFFERENT TECHNIQUES FOR SPEEDING UP NETWORKS
"""
