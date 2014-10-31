# Restricted Boltzmann Machines demo adapted from 
# "Introduction to Restricted Boltzmann Machines by Edwin Chen on Mon 18 July 2011"
# http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
#
# Jonathan Scholz
# Pfunk Lab Meeting
# 10/30/2014

from __future__ import print_function
import numpy as np

class RBM:
  
  def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.learning_rate = learning_rate

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a Gaussian distribution with mean 0 and standard deviation 0.1.
    self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)

    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      ## Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      #
      # ********** FILL IN HERE ********** 
      #

      ## Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      #
      # ********** FILL IN HERE ********** 
      #

      ## Update weights.
      #
      # ********** FILL IN HERE ********** 

      ## Compute final error (between data and visible unit probabilities)
      #
      # ********** FILL IN HERE ********** 

      print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    return self.gibbs_step(data, self.weights)
    
  def run_hidden(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of hidden units, to get a sample of the visible units.

    Parameters
    ----------
    data: A matrix where each row consists of the states of the hidden units.

    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """
    return self.gibbs_step(data, self.weights.T)
  
  def gibbs_step(self, data, weights):
    """
    Performs a single block-gibbs step of the RBM given the provided data.  
    This function can be used to resample either the hidden or visible 
    units, depending on the data and weights passed in.  

    Parameters
    ----------
    To resample the hidden units conditional on the visible units [e.g. H ~ P(H|V)]
    you should pass in:
      1) A matrix where each row consists of the states of the hidden units.
      2) The weight matrix of the RBM

    To resample the visible units conditional on the hidden units [e.g. V ~ P(V|H)]
    you should pass in:
      1) A matrix where each row consists of the states of the visible units.
      2) The TRANSPOSE of the weight matrix of the RBM

    Returns
    -------
    output_states: A matrix where each row consists of the appropraite units
    activated from the other layer of the RBM.
    """

    num_examples = data.shape[0]

    # obtain output dimension from weights
    num_output_nodes = weights.shape[1]

    ## Create a matrix, where each row is to be the output units (plus a bias unit)
    # sampled from a training example.
    output_states = np.ones((num_examples, num_output_nodes))

    ## Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    ## Calculate the activations of the output units.
    #
    # ********** FILL IN HERE ********** 

    ## Calculate the probabilities of turning the output units on.
    #
    # ********** FILL IN HERE ********** 
    
    ## Turn the output units on with their specified probabilities.
    #
    # ********** FILL IN HERE ********** 

    ## Always fix the bias unit to 1.
    # output_states[:,0] = 1

    # Ignore the bias units.
    output_states = output_states[:,1:]
    return output_states
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

  def print_query(self, query, names):
    query = np.asarray(query)
    names = np.asarray(names)
    print("latent activation for query \"%s\": %s)" % (names[query[0]==1],
     self.run_visible(query)))

if __name__ == '__main__':
  # example problem: movie categories
  # Latent => something like "oscar winners" and "scifi"
  #   oscar winners: 'LOTR', 'Gladiator', 'Titanic' ([0,0,1,1,1,0])
  #   scifi/fantasy: 'Harry Potter', 'Avatar', 'LOTR' ([1,1,1,0,0,0])
  key = ['Harry Potter', 'Avatar', 'LOTR', 'Gladiator', 'Titanic', 'Glitter']
  
  r = RBM(num_visible=6, num_hidden=2)
  training_data = np.array([
    [1,1,1,0,0,0],
    [1,0,1,0,0,0],
    [1,1,1,0,0,0],
    [0,0,1,1,1,0], 
    [0,0,1,1,0,0],
    [0,0,1,1,1,0]])

  r.train(training_data, max_epochs = 5000)
  print("Done (key=%s)" % (key))
  print("network weights", r.weights)

  r.print_query([[0,0,0,1,1,0]], key) # should be "oscar winners"
  r.print_query([[0,0,1,1,1,0]], key) # should be "oscar winners"
  r.print_query([[0,0,1,0,0,0]], key) # LOTR is both, so we'll see
  r.print_query([[1,1,1,0,0,0]], key) # should be SF/fantasy
  r.print_query([[1,1,0,0,0,0]], key) # should be strictly SF/fantasy
  r.print_query([[0,0,0,0,0,1]], key) # glitter is neither, so we'll see

  print('run hidden:\n', r.run_hidden(np.array([[0,0], [0,1], [1,0], [1,1]])))
  print('run visible:\n', r.run_visible(training_data))
  import ipdb;ipdb.set_trace()
