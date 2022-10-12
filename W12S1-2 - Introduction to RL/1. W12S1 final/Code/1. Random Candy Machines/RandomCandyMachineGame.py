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
from tqdm import tqdm

# Internal libraries
from CandyMachines import *
from CandyAgents import *


'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: CandyMachineGame
-------------------------------------------------------------------------------------------------
'''

class CandyMachineGame():
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, machines_cost = 1, \
                 playable_money = 10, \
                 number_of_deterministic_machines = 1, \
                 number_of_random_machines = 4, \
                 return_prices_for_deterministic_machines = 1, \
                 return_prices_for_random_machines_win = 2, \
                 return_prices_for_random_machines_loss = 0, \
                 random_parameters = None, \
                 agent_type = 'user', \
                 agent_parameters = None, \
                 random_seed = None, \
                 agent_random_seed = None, \
                 display_bool = True):
        
        
        # Assert and initialize random_seeds
        self.agent_random_seed = agent_random_seed
        self.assert_random_seed(random_seed)
        
        # Assert and initialize playable_money
        self.assert_playable_money(machines_cost, playable_money)
        
        # Assert and initialize machines_cost
        self.assert_machine_cost(machines_cost)
        
        # Assert and initialize number_of_deterministic_machines
        self.assert_deterministic_machines(number_of_deterministic_machines)
        
        # Assert and initialize number_of_random_machines
        self.assert_random_machines(number_of_random_machines)
        
        # Define total_number_of_machines
        self.total_number_of_machines = self.number_of_random_machines + self.number_of_deterministic_machines
        
        # Assert and initialize random_parameters
        self.assert_random_parameters(number_of_random_machines, random_parameters)
        
        # Return prices for deterministic and random machines (win/loss)
        self.assert_return_prices(return_prices_for_deterministic_machines, \
                                  return_prices_for_random_machines_win, \
                                  return_prices_for_random_machines_loss)
        
        # Initialize list_of_machines
        self.initialize_list_of_machines()
        
        # Assert and initialize display_bool
        self.assert_display_bool(display_bool)
        
        # Initialize and assert agent
        self.define_agent_game(agent_type, agent_parameters)
    
    
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
           
    def get_machine_with_machine_number(self, machine_number):
        
        # Check machine for machine_number
        for machine in self.list_of_machines:
            if machine.machine_number == machine_number:
                return machine
            
        # If we did not manage to retrieve a machine, display error.
        assert False, "Machine with number {} does not exist.".format(machine_number)
    
    
    def display_machines_list(self):
        
        # If already machines_info_df already exists, just print it.
        try:
            print(self.machines_info_df)
            
            # Otherwise, recreate it and store it in an attribute for later reuse.
        except:
            # Initialize machines_info_df as a pandas DataFrame
            self.machines_info_df = pd.DataFrame(columns = ['Machine_number', 'Machine_type', 'Cost', \
                                                      'Return_win', 'Return_loss', 'Win_probability', 'Expected_return'])
            
            # Retrieve infos for each machine and update machines_info_df
            for machine_number in range(self.total_number_of_machines):
                machine = self.get_machine_with_machine_number(machine_number + 1)
                machine_info_element = {'Machine_number': machine.machine_number, \
                                        'Machine_type': machine.machine_type, \
                                        'Cost': machine.cost, \
                                        'Return_win': machine.return_price_if_win, \
                                        'Return_loss': machine.return_price_if_loss, \
                                        'Win_probability': machine.probability_win, \
                                        'Expected_return': machine.expected_reward()}
                self.machines_info_df = self.machines_info_df.append(machine_info_element, ignore_index=True)

            # Display machines_info_df pandas
            print(self.machines_info_df)
        
    
    def play_on_machine(self, machine_number, exploration_index = None):
        
        # Retrieve machine
        machine = self.get_machine_with_machine_number(machine_number)
        
        # Check and update playable coins
        err_str1 = "You no longer have enough coins to play on this machine. "
        err_str2 = "(playable_money = {} and machine.cost = {})".format(self.playable_money, machine.cost)
        assert self.playable_money >= machine.cost, err_str
        self.playable_money = self.playable_money - machine.cost
        
        # Play on machine and return outcome
        outcome = machine.put_one_coin()
        
        # Update playable money
        self.agent.update_playable_money(machine.cost)
        
        # Update history
        self.agent.update_history(machine_number, exploration_index, machine.cost, outcome)
        
        # Update estimates
        self.agent.update_estimates(machine_number, machine.cost, outcome)
        
        # Update decay (if agent is epsilon_greedydecay)
        if self.agent_type in ['epsilon_greedydecay_naive', 'epsilon_greedydecay_linear', 'epsilon_greedydecay_softmax']:
            self.agent.update_decay_rate()
        
        
    def play(self, return_bool = False):
        
        # Iinitialize waitbar
        if self.display_bool:
            desc_str = "Playing {} coins with agent {}".format(self.playable_money, self.agent.agent_type)
            t = tqdm(desc = desc_str, total = self.playable_money)
        
        # Play on machines until
        while self.playable_money > 0:
            
            # Get machine_number via agent strategy
            machine_number, exploration_index = self.agent.get_machine_to_play()
            
            # Put a coin in machine
            self.play_on_machine(machine_number, exploration_index)
            
            # Update waitbar
            if self.display_bool:
                t.update(self.get_machine_with_machine_number(machine_number).cost)
        
        # Close waitbar
        if self.display_bool:
            t.close()
        
        # Return
        if return_bool:
            avg_performance =  self.agent.compute_agent_outcome()/self.initial_money
            avg_regret =  self.agent.compute_agent_regret()/self.initial_money
            agent_outcome_history = self.agent.history['Outcome']
            agent_regret_history = self.agent.history['Regret']
            return avg_performance, avg_regret, agent_outcome_history, agent_regret_history
    
    
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
    
    
    def assert_playable_money(self, machines_cost, playable_money):
        
        # Assert and initialize playable_money
        err_str1 = "Variable playable_money should be a strictly positive integer, and a multiple of machines_cost "
        err_str2 = "(you passed machines_cost = {} and playable_money = {}).".format(machines_cost, playable_money)
        err_str = err_str1 + err_str2
        assert isinstance(playable_money, (int)) and playable_money % machines_cost == 0, err_str
        self.playable_money = playable_money
        self.initial_money = playable_money
        
    
    def assert_machine_cost(self, machines_cost):
        
        # Assert and initialize machines_cost
        err_str1 = "Variable machines_cost should be a strictly positive integer "
        err_str2 = "(you passed machines_cost = {}).".format(machines_cost)
        err_str = err_str1 + err_str2
        assert isinstance(machines_cost, int) and machines_cost > 0, err_str
        self.machines_cost = machines_cost
        
        
    def assert_deterministic_machines(self, number_of_deterministic_machines):
        
        # Assert and initialize number_of_deterministic_machines
        err_str1 = "Variable number_of_deterministic_machines should be a strictly positive integer "
        err_str2 = "(you passed number_of_deterministic_machines = {}).".format(number_of_deterministic_machines)
        assert isinstance(number_of_deterministic_machines, int) and number_of_deterministic_machines > 0, err_str
        self.number_of_deterministic_machines = number_of_deterministic_machines
        
        
    def assert_random_machines(self, number_of_random_machines):
        
        # Assert and initialize number_of_random_machines
        err_str1 = "Variable number_of_random_machines should be a strictly positive integer "
        err_str2 = "(you passed number_of_random_machines = {}).".format(number_of_random_machines)
        assert isinstance(number_of_random_machines, int) and number_of_random_machines > 0, err_str
        self.number_of_random_machines = number_of_random_machines
        
    
    def assert_random_parameters(self, number_of_random_machines, random_parameters):
        
        # Initialize and assert random_parameters
        if random_parameters == None:
            self.random_parameters = [np_random() for i in range(number_of_random_machines)]
        else:
            '''
            Note: To be updated later, to take into account custom random parameters for random machines.
            '''
            assert False, "Variable random_parameters should be set to None for now."
            
            
    def assert_return_prices(self, return_prices_for_deterministic_machines, \
                             return_prices_for_random_machines_win, return_prices_for_random_machines_loss):
        
        # Return prices for deterministic and random machines (win/loss)
        '''
        Note: To be updated later to take into account custom values for return prices.
        '''
        self.return_prices_for_deterministic_machines = return_prices_for_deterministic_machines
        self.return_prices_for_random_machines_win = return_prices_for_random_machines_win
        self.return_prices_for_random_machines_loss = return_prices_for_random_machines_loss
        
    def initialize_list_of_machines(self):
        
        # Initialize DeterministicCandyMachines
        list_of_deterministic_machines = [DeterministicCandyMachine(machine_number = i + 1, \
                                            cost = self.machines_cost, \
                                            return_price_if_win = self.return_prices_for_deterministic_machines)
                                          for i in range(0, self.number_of_deterministic_machines)]
        
        # Initialize RandomCandyMachines
        list_of_random_machines = [RandomCandyMachine(machine_number = i + 1 + self.number_of_deterministic_machines, \
                                      cost = self.machines_cost, \
                                      return_price_if_win = self.return_prices_for_random_machines_win, \
                                      return_price_if_loss = self.return_prices_for_random_machines_loss, \
                                      probability_win = self.random_parameters[i])
                                    for i in range(0, self.number_of_random_machines)]
        
        # Initialize list_of_machines
        self.list_of_machines = list_of_deterministic_machines + list_of_random_machines
        
        
    def assert_display_bool(self, display_bool):
        
        # Assert and initialize display_bool
        self.display_bool = display_bool
        
        
    def assert_random_seed(self, random_seed = None):
        
        # Assert and initialize random_seed
        if isinstance(random_seed, int):
            self.random_seed = random_seed
            np_seed(random_seed)
        else:
            print("Warning: Invalid value passed for random_seed parameters, using None instead.")
            self.random_seed = None
            
    
    def define_agent_game(self, agent_type, agent_parameters = None):
            
        # Initialize agent_type and parameters
        self.agent_type = agent_type
        self.agent_parameters = agent_parameters
        
        # Initialize agent
        if self.agent_type == 'user':
            # Initialize user agent
            self.agent = UserAgent(list_of_machines = self.list_of_machines, \
                                   playable_money = self.playable_money, \
                                   random_seed = self.agent_random_seed)
            # Set display bool to False is user agent
            self.display_bool = False
            
        elif self.agent_type == 'deterministic':
            # Initialize deterministic agent
            self.agent = DeterministicAgent(list_of_machines = self.list_of_machines, \
                                            playable_money = self.playable_money, \
                                            random_seed = self.agent_random_seed)
            
        elif self.agent_type == 'prior_knowledge':
            # Initialize prior knowledge agent
            self.agent = PriorKnowledgeAgent(list_of_machines = self.list_of_machines, \
                                             playable_money = self.playable_money, \
                                             random_seed = self.agent_random_seed)

        elif self.agent_type == 'random_naive':
            # Initialize random agent (naive)
            self.agent = RandomAgent(list_of_machines = self.list_of_machines, \
                                     playable_money = self.playable_money, \
                                     random_type = 'naive', \
                                     random_seed = self.agent_random_seed)

        elif self.agent_type == 'random_weighted_linear':
            # Initialize random agent (weigthed linear)
            self.agent = RandomAgent(list_of_machines = self.list_of_machines, \
                                     playable_money = self.playable_money, \
                                     random_type = 'linear', \
                                     random_seed = self.agent_random_seed)

        elif self.agent_type == 'random_weighted_softmax':
            # Initialize random agent (weighted softmax)
            self.agent = RandomAgent(list_of_machines = self.list_of_machines, \
                                     playable_money = self.playable_money, \
                                     random_type = 'softmax', \
                                     random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_first_naive':
            # Initialize epsilon first random agent (naive)
            self.agent = EpsilonFirstAgent(list_of_machines = self.list_of_machines, \
                                           playable_money = self.playable_money, \
                                           random_type = 'naive', \
                                           epsilon_value = self.agent_parameters, \
                                           random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_first_linear':
            # Initialize epsilon first random agent (linear)
            self.agent = EpsilonFirstAgent(list_of_machines = self.list_of_machines, \
                                           playable_money = self.playable_money, \
                                           random_type = 'linear', \
                                           epsilon_value = self.agent_parameters, \
                                           random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_first_softmax':
            # Initialize epsilon first random agent (softmax)
            self.agent = EpsilonFirstAgent(list_of_machines = self.list_of_machines, \
                                           playable_money = self.playable_money, \
                                           random_type = 'softmax', \
                                           epsilon_value = self.agent_parameters, \
                                           random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_greedy_naive':
            # Initialize epsilon greedy random agent (naive)
            self.agent = EpsilonGreedyAgent(list_of_machines = self.list_of_machines, \
                                            playable_money = self.playable_money, \
                                            random_type = 'naive', \
                                            epsilon_value = self.agent_parameters, \
                                            random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_greedy_linear':
            # Initialize epsilon greedy random agent (linear)
            self.agent = EpsilonGreedyAgent(list_of_machines = self.list_of_machines, \
                                            playable_money = self.playable_money, \
                                            random_type = 'linear', \
                                            epsilon_value = self.agent_parameters, \
                                            random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_greedy_softmax':
            # Initialize epsilon greedy random agent (softmax)
            self.agent = EpsilonGreedyAgent(list_of_machines = self.list_of_machines, \
                                            playable_money = self.playable_money, \
                                            random_type = 'softmax', \
                                            epsilon_value = self.agent_parameters, \
                                            random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_firstgreedy_naive':
            # Initialize epsilon first-greedy random agent (naive)
            self.agent = EpsilonFirstGreedyAgent(list_of_machines = self.list_of_machines, \
                                                 playable_money = self.playable_money, \
                                                 random_type = 'naive', \
                                                 epsilon_values = self.agent_parameters, \
                                                 random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_firstgreedy_linear':
            # Initialize epsilon first-greedy random agent (linear)
            self.agent = EpsilonFirstGreedyAgent(list_of_machines = self.list_of_machines, \
                                                 playable_money = self.playable_money, \
                                                 random_type = 'linear', \
                                                 epsilon_values = self.agent_parameters, \
                                                 random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_firstgreedy_softmax':
            # Initialize epsilon first-greedy random agent (softmax)
            self.agent = EpsilonFirstGreedyAgent(list_of_machines = self.list_of_machines, \
                                                 playable_money = self.playable_money, \
                                                 random_type = 'softmax', \
                                                 epsilon_values = self.agent_parameters, \
                                                 random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_greedydecay_naive':
            # Initialize epsilon greedy-decay random agent (naive)
            '''
            Note: To be updated later to take into account custom values for agent_parameters.
            '''
            self.agent = EpsilonGreedyDecayAgent(list_of_machines = self.list_of_machines, \
                                                 playable_money = self.playable_money, \
                                                 random_type = 'naive', \
                                                 epsilon_value = 1, \
                                                 decay_ratio = 0.5, \
                                                 decay_step = 0.05, \
                                                 random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_greedydecay_linear':
            # Initialize epsilon greedy-decay random agent (linear)
            '''
            Note: To be updated later to take into account custom values for agent_parameters.
            '''
            self.agent = EpsilonGreedyDecayAgent(list_of_machines = self.list_of_machines, \
                                                 playable_money = self.playable_money, \
                                                 random_type = 'linear', 
                                                 epsilon_value = 1, \
                                                 decay_ratio = 0.5, \
                                                 decay_step = 0.05, \
                                                 random_seed = self.agent_random_seed)

        elif self.agent_type == 'epsilon_greedydecay_softmax':
            # Initialize epsilon greedy-decay random agent (softmax)
            '''
            Note: To be updated later to take into account custom values for agent_parameters.
            '''
            self.agent = EpsilonGreedyDecayAgent(list_of_machines = self.list_of_machines, \
                                                 playable_money = self.playable_money, \
                                                 random_type = 'softmax', 
                                                 epsilon_value = 1, \
                                                 decay_ratio = 0.5, \
                                                 decay_step = 0.05, \
                                                 random_seed = self.agent_random_seed)
            
    
    def compute_agent_outcome(self):
        # Compute number of candies obtained so far
        return self.agent.compute_agent_outcome()
    
    
    def compute_agent_regret(self):
        # Compute number of candies missed so far
        return self.agent.compute_agent_regret()
        
        
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    -------------------------------------------------------------------------------------------------
    '''
    
    def describe(self):
        
        # Describe object method for game (to be updated later)
        print(self.__dict__)
        