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
    
    def __init__(self, neuron_type, activation_fun = 'sigmoid'):
        
        self.neuron_type = neuron_type
        if neuron_type == 'input':
            self.activation_function = 'identity'
            
        if neuron_type == 'hidden' or neuron_type == 'output':
            #Setting activation function (either sigmoid or relu)
            self.activation_function = activation_fun
        if self.neuron_type == 'bias':
            self.activation = 1.0
            self.activation_function = activation_fun
         
    
    # Function to initialize the weights coming in to the neuron
    def initialize_weights(self, n_input_neurons, w_distribution = 'uniform', w_range = 0.3):
        
        
        # Initializing weights with uniform distribution between -0.3 and 0.3 (or user specified)
        if w_distribution == 'uniform':             
            weights = np.random.uniform(low=-w_range, high=w_range, size= n_input_neurons)
        
        # Initializing weights with normal distribution mean 0, sd 1
        if w_distribution == 'gaussian':
            weights = np.random.normal(0, w_range, n_input_neurons)

        
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
        # Using specified activation function for hidden and output layers
        if self.neuron_type == 'hidden' or self.neuron_type == 'output':
            net_input = sum(inputs * self.weights)
            
            # If sigmoid activation using sigmoid function
            if self.activation_function == 'sigmoid':
                activation = 1 / (1 + math.exp(-net_input))
                
            # If ReLU activation function
            elif self.activation_function == 'relu':
                activation = max(0, net_input)
            
            self.net_input = net_input
            self.activation = activation
        
        # If the neuron is a bias neuron activation should simply always be 1
        if self.neuron_type == 'bias':
            self.activation = 1.0
        

            
    # Calculate delta of the neuron        
    def calculate_delta(self, target, unit_n = None):
        
        # First calculating the derivative of the activation function
        # For sigmoid derivative
        if self.activation_function == 'sigmoid':
            self.activation_derivative = self.activation*(1-self.activation)
        # For ReLU, derivative is 1 if activation is greater than 0, otherwise it's 0
        if self.activation_function == 'relu':
            if self.activation > 0:
                self.activation_derivative = 1
            else:
                self.activation_derivative = 0
                     
        
        # Formula for output neurons
        if self.neuron_type == 'output':
            #print("output {}".format(self.activation_derivative))
            self.delta = (target - self.activation) * self.activation_derivative
        
            # Calculating delta * weights for use in hidden layer
            self.delta_times_weight = self.delta * self.weights
            
            # Besides calculating delta, output neurons should also calculate the error 
            self.squared_error = (target - self.activation)**2
            
            
        if self.neuron_type == 'hidden' or self.neuron_type == 'bias':
            #print("hidden {}".format(self.activation_derivative))
            # Calculating the sum of the errors for the hidden units
            next_layer_delta_times_weight = []
            for d in range(np.size(target)):
                error_times_weight = target[d].delta_times_weight[unit_n] * target[d].weights[unit_n]
                next_layer_delta_times_weight.append(error_times_weight)
                
            sum_delta_times_weight = sum(next_layer_delta_times_weight)
            
            # Calculating delta
            self.delta = self.activation_derivative * sum_delta_times_weight
            
            
    
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
    
    def __init__(self, input_units, hidden_units, output_units, pattern_error, population_error):
        
        self.input = input_units
        self.hidden = hidden_units
        self.output = output_units
        self.pat_error = pattern_error
        self.population = population_error




# -------------------------------------------------------------------------------------------


# Print activation and weights for a given input
def test_performance(network, test_input, target = "not specified", verbose = False):

    # Setting activation for the input units and saving activations in a variable
    input_activations =  []
    for u in range(np.size(network.input)-1):
        network.input[u].calculate_activation(test_input[u])
        input_activations.append(network.input[u].activation)
        if verbose == True:
            print("Activation for input unit {} is: {}".format(u, network.input[u].activation))
        
    
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
    if verbose == True:    
        for u in range(network.hidden.size - 1):
            print("Weights coming in to hidden unit {} are {}".format(u, network.hidden[u].weights))        
            print("Activation for hidden unit {} is {}".format(u, network.hidden[u].activation))
         
        print("\n")
    hidden_activations = np.array(hidden_activations)
             
    
    # For all neurons in output layer, calculate activation, and calculate delta.
    output_activations = []
    for u in range(network.output.size):
        network.output[u].calculate_activation(hidden_activations)
        output_activations.append(network.output[u].activation)
        if verbose == True:
            print("Weights coming in to output unit {} are {}".format(u, network.output[u].weights))
            print("Activation for output unit {} is {}".format(u, network.output[u].activation))

    if verbose == True:
        print("\nWith the input {}, output is {}".format(test_input, output_activations))
        print("Target was {}".format(target))
        
    # Calculating whether the prediction was correct or not
    # For only 1 output:
    

    if len(output_activations) == 1:
        if output_activations[0] > 0.5:
            output_activations[0] = 1
        else:
            output_activations[0] = 0
        
        if output_activations[0] == target:
            correct = True
        else:
            correct = False
    
    
    # For more than 1 output
    else:    
        # Making output_activaitons a np array so the argmax function can be used to find the highest value (index)
        output_activations = np.array(output_activations)
        # If the highest predicted value is at the same place as the target, mark as correct
        if np.argmax(output_activations) == np.argmax(target):
            correct = True
        else:
            correct = False
    
    return(correct)



def test_multi(network, data, target):
    # Using the test_performance method to accuracy on a dataset
    correct = []
    for i in range(len(data)):
        correct.append(test_performance(network, data[i], target[i]))
    
    accuracy = (sum(correct) / len(data)) * 100
    return(accuracy)


# --------------------------------------------------------------------------
def read_input(input_file):
    
    #Reading input file and removing empty entries in the list if any (a \n at the end of document makes an empty entry)
    with open(input_file, 'r') as f:
        in_pairs = f.read().split('\n')
      
    in_pairs = list(filter(None, in_pairs))
    
    # Splitting the pairs into paired groups
    split_pairs = []
    for i in range(len(in_pairs)):
        one_pair = in_pairs[i].split(" ")
        # Removing empty elements from list
        one_pair = list(filter(None, one_pair))
        split_pairs.append(one_pair)
    
    # Turning into a numpy array
    d = np.array(split_pairs)
    d = d.astype(np.float)
    return(d)    
    

def read_teacher(teacher_file):
        
    # Same thing for teaching input
    with open(teacher_file, 'r') as f:
        target = f.read().split('\n')
    
    target = list(filter(None, target))
    
    target_pairs = []
    for i in range(len(target)):
        one_pair = target[i].split(" ")
        target_pairs.append(one_pair)
    
    t = np.array(target_pairs)
    t = t.astype(np.float)
    return(t)


# --------------------------------------------------------------------------

def init_weights(params, w_distribution, w_range, activation_func):

    # Reading param file and saving to variables
    with open(params, 'r') as f:
        args = f.read().split("\n")
         
    
    n_input = int(args[0])
    n_hidden = int(args[1])
    n_output = int(args[2])
    
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
        hidden.append(neuron('hidden', activation_func))
    hidden.append(neuron('bias'))
    hidden = np.array(hidden)
    
    # Initializing weights for the hidden layer
    for u in range(hidden.size):
        hidden[u].initialize_weights(inputs.size, w_distribution, w_range)
    
    
    
    # Creating array of output units
    outputs = []
    for n in range(n_output):
        outputs.append(neuron('output'))
    outputs = np.array(outputs) 
    
    # Initializing weights for links to the output layer
    for u in range(outputs.size):
        outputs[u].initialize_weights(hidden.size, w_distribution, w_range)
        
    # Print weights   
    for u in range(hidden.size - 1):
        print("Weights coming in to hidden unit {} are {}".format(u, hidden[u].weights))        
        print("\n")
             
    
    for u in range(outputs.size):
        print("Weights coming in to output unit {} are {}".format(u, outputs[u].weights))

            
    output = network(inputs, hidden, outputs, 0, 0)
    return(output)

# -------------------------------------------------------------------------


def train_nn(data, target, params, max_epochs, validation_data = None, validation_target = None, 
             save_to_file = None, given_eta = None, count = None,
             w_distribution = 'uniform', w_range = 0.3, activation_func = 'sigmoid', prev_init = None):

    """
    Trains the neural network
        - Data = training data. Can be string linking to .txt file or numpy array
        - Target = training target. Can be string linking to .txt file or numpy array
        - Params = .txt file specifying number of input, hidden, and output units as well as constants for
                    momemtun, eta, and population error
        - Max_epochs = the maximum number of epochs the network will train for if it doesn't reach goal population error
        - Validation_data = test data, numpy array
        - Validation_target = test target, numpy array
        - save_to_file = default is don't save, can specify a file to write results to:
                                        (Train/test accuracy, n epochs, population error)
        - given_eta = by default uses eta specified in params, but can specify other (used to test different values)
        - count = used to count iteration number (used to test different values of eta and keep track of iterations)
        - w_distribution = can be 'uniform' or 'gaussian'. Gaussian is mean 0, sd 1.
        - w_range = if w_distribution == 'uniform' can specifiy the range to draw values from (default -0.3-0.3)
        - network = if weights have previously been initialized, use those
    """


    # Reading param file and saving to variables
    with open(params, 'r') as f:
        args = f.read().split("\n")
         
    
    n_input = int(args[0])
    n_hidden = int(args[1])
    n_output = int(args[2])
    if given_eta is None:
        eta = float(args[3])
    else:
        eta = given_eta
    momentum = float(args[4])
    error_criterion = float(args[5])
    
    # Reading input data
    
    # If string, read the txt file
    if type(data) == str:
        d = read_input(data)
    # If supplied with an array, use the array
    if type(data) == np.ndarray:
        d = data
    
    if type(target) == str:
        t = read_teacher(target)
    if type(target) == np.ndarray:
        t = target
       
    
    
    # ----------------------------------------------------------------------------------------
    
    
    # Setting up network architecture
    if prev_init == None:
        # Creating array of input units for as many units as specified in param.txt + a bias unit
        inputs = []
        for n in range(n_input):
            inputs.append(neuron('input'))
        inputs.append(neuron('bias'))
        inputs = np.array(inputs)
        
        
        # Making an array of the hidden layer; repeating for as many units specified + a bias unit
        hidden = []
        for n in range(n_hidden):
            hidden.append(neuron('hidden', activation_func))
        hidden.append(neuron('bias'))
        hidden = np.array(hidden)
        
        # Initializing weights for the hidden layer
        for u in range(hidden.size):
            hidden[u].initialize_weights(inputs.size, w_distribution, w_range)
        
        
        
        # Creating array of output units
        outputs = []
        for n in range(n_output):
            outputs.append(neuron('output'))
        outputs = np.array(outputs) 
        
        # Initializing weights for links to the output layer
        for u in range(outputs.size):
            outputs[u].initialize_weights(hidden.size, w_distribution, w_range)
    
    else:   
        inputs = prev_init.input
        hidden = prev_init.hidden
        outputs = prev_init.outputs
    
    
    # Initializing population error with an arbitrarily high value
    population_error = 10000.0
    n_epochs = 0
    
    # Looping for each training input until error is below threshold
    while population_error > error_criterion:
    
        # Setting up variable to store pattern errors 
        pat_error = np.zeros(len(d))        
        
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
                     
            
            p_squared_error = []
            # For all neurons in output layer, calculate activation, and calculate delta.
            for u in range(outputs.size):
                outputs[u].calculate_activation(hidden_activations)
                outputs[u].calculate_delta(t[index[p],u])
                
                p_squared_error.append(outputs[u].squared_error)
            
            p_sum_squared_error = sum(p_squared_error)
            pat_error[p] = p_sum_squared_error
    
      
            # Update weights coming to the output(s)
            for u in range(outputs.size):
                outputs[u].update_weight(hidden_activations, eta, momentum)
               
            
            # Calculate delta for hidden unit(s) and update incoming weights
            for u in range(hidden.size):
                hidden[u].calculate_delta(outputs, u)
                
                #FOR ONLINE LEARNING
                hidden[u].update_weight(input_activations, eta, momentum)
                
      
        # Calculate population error
        population_error = sum(pat_error) / (n_output * len(d))

        n_epochs += 1
        if n_epochs % 100 == 0: 
            print("Number of epochs: {}".format(n_epochs))
            print("Pop error is {}".format(population_error))
            
            # Calculating test/train accuracy (if testing set is given)
            if validation_data is None:
                pass
            else:
                model = network(inputs, hidden, outputs, pat_error, population_error)
                train_acc = test_multi(model, data, target)
                test_acc = test_multi(model, validation_data, validation_target)
                #print("Training accuracy is {} and test accuracy is {}".format(train_acc, test_acc))
                
                #Saving to file
                if save_to_file is None:
                    pass
                else:
                    output = "{},{},{},{},{},{},{},{}\n".format(n_epochs, train_acc, test_acc, population_error, 
                                                            eta, w_distribution, w_range, count)
                    with open(save_to_file, 'a') as f:
                        f.write(output)
        
        # Stopping if maximum number of epochs is reached
        if n_epochs == max_epochs:
            print('\nMax number of epochs reached.\nNumber of epochs: {}, population error: {}'.format(n_epochs, population_error))
            output = network(inputs, hidden, outputs, pat_error, population_error)
            return(output)
            break
    
    print('\nTraining done.\nNumber of epochs: {}, population error: {}'.format(n_epochs, population_error))
    output = network(inputs, hidden, outputs, pat_error, population_error)
    return(output)

# -------------------------------------------------------------------        
    
# User interface
print("Hello Anthony, and welcome to my neural network builder!\nTo begin, choose an activation function for hidden units.")
while True:
    
    # Getting activation function to draw weights from   
   while True:    
        act_func = input("Type 'sigmoid' for sigmoid or 'relu' for ReLU\n")    
        if act_func not in ['sigmoid', 'relu']:
            print("Invalid input. Try again.")
        else:
            break

   print("Alright, let's initialize the weights.\n")
   
   while True:
        print("If you want to draw weights from a uniform distribution press 1")
        print("If you want to draw weights from a Gaussian distribution press 2")
        dist = input()

        if dist not in ["1","2"]:
            print("Invalid input, try again")
        else:
            if dist == '1':
                w_distribution = 'uniform'
            if dist == '2':
                w_distribution = 'gaussian'
                
            print("Sweet, {} distribution it is.".format(w_distribution))
            break

    # Getting range to draw weights from
   if w_distribution == 'uniform':
       while True:
           try:
               w_range = float(input("From which range around zero should the weights be drawn?\n"))
               break
           except ValueError:
               print("Not numeric. Try again.")
 
   if w_distribution == 'gaussian':
       while True:
           try:
               w_range = float(input("Mean is 0, what should the standard deviation be?\n"))
               break
           except ValueError:
               print("Not numeric. Try again.") 
                
 
    
    
   print("All set! Initialzing weights.\n \n")
    
   init_network = init_weights('param.txt', w_distribution, w_range, act_func)
    
   while True:
       proceed = input("""If you are happy with these weights, press 1.\n
       If you want to redraw weights, press 2.\n""")
       if proceed not in ["1", "2"]:
           print("Invalid input, try again")
        
       if proceed == '1':
           break
       if proceed == '2':
           init_network = init_weights('param.txt', w_distribution, w_range, act_func)


   # n epochs
   while True:
       try:
           max_epochs = int(input("""The network will train until it reaches the population error specified in param.txt. 
           If it does not reach it, after how many epochs should the program terminate?\n""")) 
           break
       except ValueError:
           print("Not an integer. Try again.")
        
   print("Ready to go! The network will now begin training")
   
   
   net = train_nn(data = 'in.txt', target = 'target.txt', params =  'param.txt', max_epochs = max_epochs, 
                  w_distribution = w_distribution, w_range = w_range, activation_func = act_func)
                  
   print("\nAll done!\nPress 1 if you want to exit\nPress 2 if you want to test performance")

   while True:
       try:   
           what_now = int(input())
           if what_now in [1, 2]:
               break
       except ValueError:
           print("Invalid input. Try again.")
   
   if what_now == 1:
       break
   
   # Testing performance
   if what_now == 2:
       data = read_input('in.txt')
       target = read_teacher('target.txt')
       train_acc = test_multi(net, data, target)
       
       print("The network's accuracy across the whole dataset is {}%".format(train_acc))
       print("To test how well the network fares on a specific row in your data, type the row number you want to test\n")
       print("Else, type 'q' to quit")

       while True:
           row_n = input()
           
           if row_n == 'q':
               break
           
           else:
               try:
                   row_n = int(row_n)
               except ValueError:
                   print("Invalid input. Try again.")
               
               while True:
                   perf = test_performance(net, data[row_n], target[row_n], verbose = True)
                   print("\n \nPress 'q' to quit, otherwise type another row number to test")
                   
                   test_more = input()
                   if test_more == 'q':
                       break
                   else:
                       row_n = int(test_more)
           break
       
 
   break
    
