'''
-------------------------------------------------------------------------------------------------
CLASS IMPORTS
-------------------------------------------------------------------------------------------------
'''

# External libraries
from matplotlib import pyplot as plt
from numpy.random import seed as np_seed
from numpy.random import random as np_random
from numpy.random import randint as np_randint
from numpy import cumsum as np_cumsum
import pandas as pd

# Internal libraries
# None


'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: CandyMachine
-------------------------------------------------------------------------------------------------
'''

class CandyMachine():
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, machine_number, cost = 1, return_price_if_win = 1):
        
        # Assert and initialize machine_number
        self.assert_machine_number(machine_number)
        
        # Initialize machine_type
        self.machine_type = None
        
        # Assert and initialize machine cost
        self.assert_machine_cost(cost)
        
        # Assert and initialize return_price_if_win
        self.assert_return_price_if_win(return_price_if_win)
        
    
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    # None
    -------------------------------------------------------------------------------------------------
    '''
    
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
    
    def assert_machine_number(self, machine_number):
        
        # Assert and initialize machine_number
        err_str1 = "Variable machine_number should be a positive integer "
        err_str2 = "(you passed machine_number = {}).".format(machine_number)
        err_str = err_str1 + err_str2
        assert isinstance(machine_number, int) and machine_number > 0, err_str
        self.machine_number = machine_number
        
        
    def assert_machine_cost(self, cost):
        
        # Assert and initialize machine cost
        err_str1 = "Variable cost should be a strictly positive integer "
        err_str2 = "(you passed cost = {}).".format(cost)
        err_str = err_str1 + err_str2
        assert isinstance(cost, int) and cost > 0, err_str
        self.cost = cost
        
    
    def assert_return_price_if_win(self, return_price_if_win):
        
        # Assert and initialize return_price_if_win
        err_str1 = "Variable return_price_if_win should be a strictly positive integer "
        err_str2 = "(you passed return_price_if_win = {}).".format(return_price_if_win)
        err_str = err_str1 + err_str2
        assert isinstance(return_price_if_win, int) and return_price_if_win > 0, err_str
        self.return_price_if_win = return_price_if_win
        
    
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    -------------------------------------------------------------------------------------------------
    '''
    
    def describe(self):
        
        # Describe object method for machines (to be updated later)
        print(self.__dict__)
        
        
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: DeterministicCandyMachine
-------------------------------------------------------------------------------------------------
'''

class DeterministicCandyMachine(CandyMachine):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, machine_number, cost = 1, return_price_if_win = 1):
        
        # Reuse super class __init__
        super().__init__(machine_number, cost = cost, return_price_if_win = return_price_if_win)
        
        # Initialize parameters, specific to DeterministicCandyMachines
        self.init_deterministic_machine_parameters()
        
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def put_one_coin(self):
        return self.return_price_if_win
    
    
    def put_n_coins(self, coins_number):
        return sum([self.put_one_coin() for i in range(coins_number)])
    
    
    def expected_reward(self, coins_number = 1):
        expectation = coins_number*self.return_price_if_win/self.cost
        return expectation
    
    
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
    
    def init_deterministic_machine_parameters(self):
        
        # Initialize parameters, specific to DeterministicCandyMachines
        self.machine_type = 'deterministic'
        self.return_price_if_loss = 1
        self.probability_win = 1
        
        
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing the one from Agent class.
    -------------------------------------------------------------------------------------------------
    '''
    
    
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: RandomCandyMachine
-------------------------------------------------------------------------------------------------
'''

class RandomCandyMachine(CandyMachine):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, machine_number, cost = 1, return_price_if_win = 2, \
                 return_price_if_loss = 0, probability_win = None):
        
        # Reuse super class __init__
        super().__init__(machine_number, cost = cost, return_price_if_win = return_price_if_win)
        
        # Initialize parameters, specific to RandomCandyMachines
        self.init_random_machine_parameters()
        
        # Assert and initialize return_price_if_loss
        self.assert_return_price_if_loss(return_price_if_loss)
        
        # Assert and initialize probability_win
        self.assert_probability_win(probability_win)
            
            
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def put_one_coin(self):
        random_val = np_random()
        price_value = self.return_price_if_win if self.probability_win >= random_val else self.return_price_if_loss
        return price_value
    
    
    def put_n_coins(self, coins_number):
        return sum([self.put_one_coin() for i in range(coins_number)])
    
    
    def expected_reward(self, coins_number = 1):
        expectation = coins_number/self.cost*(self.return_price_if_win*self.probability_win \
                        + (1 - self.probability_win)*self.return_price_if_loss)
        return expectation
    
    
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
    
    def init_random_machine_parameters(self):
        
        # Initialize parameters, specific to RandomCandyMachines
        self.machine_type = 'random'
        
    
    def assert_return_price_if_loss(self, return_price_if_loss):
        
        # Assert and initialize return_price_if_loss
        err_str1 = "Variable return_price_if_loss should be a positive integer "
        err_str2 = "(you passed return_price_if_loss = {}).".format(return_price_if_loss)
        err_str = err_str1 + err_str2
        assert isinstance(return_price_if_loss, int) and return_price_if_loss >= 0, err_str
        self.return_price_if_loss = return_price_if_loss
        
        
    def assert_probability_win(self, probability_win):
        
        # Assert and initialize probability_win
        if probability_win == None:
            # If no value is passed, decide on a random threshold
            self.probability_win = np_random()
        else:
            # Otherwise, use passed value, if valid
            err_str1 = "Variable probability_win should be an integer or float value in [0, 1] "
            err_str2 = "(you passed probability_win = {}).".format(probability_win)
            err_str = err_str1 + err_str2
            assert isinstance(probability_win, (int, float)) and probability_win >= 0 and probability_win <= 1, err_str
            self.probability_win = probability_win
            
            
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing the one from Agent class.
    -------------------------------------------------------------------------------------------------
    '''
    