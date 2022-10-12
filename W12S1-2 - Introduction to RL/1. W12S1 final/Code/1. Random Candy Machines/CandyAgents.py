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
from numpy import exp as np_exp
from numpy import mean as np_mean
from numpy import array as np_array
import pandas as pd

# Internal libraries
from CandyMachines import *
        
        
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: Agent
-------------------------------------------------------------------------------------------------
'''

class Agent():
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_seed = None):
        
        # Initialize agent_type
        self.agent_type = None
        
        # Initialize playable money
        self.initial_money = playable_money
        self.playable_money = playable_money
        
        # Initialize total_number_of_machines
        self.total_number_of_machines = len(list_of_machines)
        
        # Upper bound perfomance for all machines
        self.display_parameters = {'upper_bound': max([machine.expected_reward() for machine in list_of_machines]), \
                                   'min_val': 0.95*min([machine.return_price_if_loss/machine.cost for machine in list_of_machines]), \
                                   'max_val': 1.05*max([machine.return_price_if_win/machine.cost for machine in list_of_machines])}
        best_machine_index = [machine.expected_reward() for machine in list_of_machines].index(self.display_parameters['upper_bound'])
        self.display_parameters['best_machine'] = list_of_machines[best_machine_index].machine_number
                                                   
        
        # Initialize history
        self.history = {'Machine_number': [], 'Exploration_play': [], 'Cost': [], 'Outcome': [], 'Regret': []}
        
        # Assert and initialize random_seed
        self.assert_random_seed(random_seed)
        
        # Initialize machines_estimates
        initial_value = 1/2
        self.machine_estimates = {machine.machine_number: (initial_value*machine.return_price_if_win \
                                  + (1 - initial_value)*machine.return_price_if_loss)/machine.cost for machine in list_of_machines}
        self.machine_times_played = {machine.machine_number: 0 for machine in list_of_machines}
        
    
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def update_playable_money(self, machine_cost):
        
        # Assert machine_cost is not larger than the playable money you have
        err_str1 = "Machine cost is higher than the currently available money "
        err_str2 = "(playable money = {}, machine cost = {})".format(self.playable_money, machine_cost)
        assert machine_cost <= self.playable_money, err_str1 + err_str2
        
        # Remove machine cost from playable money
        self.playable_money -= machine_cost
    
    
    def update_history(self, machine_number, exploration_index, machine_cost, outcome):
        
        # Update history with given roll information
        self.history['Machine_number'].append(machine_number)
        '''
        Add exploration index later.
        '''
        self.history['Exploration_play'].append(exploration_index)
        self.history['Cost'].append(machine_cost)
        self.history['Outcome'].append(outcome)
        self.history['Regret'].append(self.display_parameters['upper_bound'] - outcome)
        
        
    def display_history(self):
        
        # Convert dicitonnary into pandas and display
        history_df = pd.DataFrame(data = self.history)
        print(history_df)
        
    
    def update_estimates(self, machine_number, machine_cost, outcome):
        
        # Increase number of times machine has been played by one
        self.machine_times_played[machine_number] += 1
        
        # Update estimated reward for machine based on previous rolls and new outcome
        self.machine_estimates[machine_number] = (self.machine_estimates[machine_number]*self.machine_times_played[machine_number] \
            + (outcome/machine_cost)) / (self.machine_times_played[machine_number] + 1)
        
        
    def compute_agent_outcome(self):
        # Sum all values in 'Outcome' column of history
        return sum(self.history['Outcome'])
    
    
    def compute_agent_regret(self):
        # Sum all values in 'Regret' column of history
        return sum(self.history['Regret'])
    
    
    def display_machines_time_played(self, scaling_factor = 100):
        
        # Display warning if not all coins have been used
        if self.playable_money > 0:
            warning_str1 = "Warning: you still have playable money left ({} coins remaining).".format(self.playable_money)
            warning_str2 = "\nDisplaying graphs anyway."
            print(warning_str1 + warning_str2)
        
        # Display number of times each machine has been played
        plt.figure(figsize = (10, 7))
        plt.bar(list(self.machine_times_played.keys()), list(self.machine_times_played.values()))
        title_str1 = "Number of times each machine has been played by agent."
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Machine number')
        plt.ylabel('Number of times played')
        plt.show()
        
        # Display which machine was played on each turn
        window_size = max(1, int(self.initial_money/scaling_factor))
        self.assert_window_size(window_size)
        machine_numbers = self.history['Machine_number']
        machine_numbers_avg = [max(set(machine_numbers[i:min(i + window_size, len(machine_numbers))]), \
                               key = machine_numbers[i:min(i + window_size, len(machine_numbers))].count) \
                               for i in range(0, len(machine_numbers), window_size)]
        plt.figure(figsize = (10, 7))
        plt.plot([i*window_size for i in range(len(machine_numbers_avg))], machine_numbers_avg)
        title_str1 = "Machine played on each turn by agent."
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Machine number')
        plt.show()
        
        
    def display_performance_over_time(self, scaling_factor = 100):
        
        # Display average performance over time for agent
        window_size = max(1, int(self.initial_money/scaling_factor))
        self.assert_window_size(window_size)
        machine_outcome = self.history['Outcome']
        machine_outcome_avg = [np_mean(machine_outcome[i:min(i + window_size, len(machine_outcome))]) \
                               for i in range(0, len(machine_outcome), window_size)]
        
        # Display average performance over time for agent (normalized axis)
        plt.figure(figsize = (10, 7))
        label_str1 = 'Avg. performance of agent'
        label_str2 = 'Best performance (th. avg.)'
        plt.plot([i*window_size for i in range(len(machine_outcome_avg))], machine_outcome_avg, label = label_str1)
        plt.plot([i*window_size for i in range(len(machine_outcome_avg))], \
                 [self.display_parameters['upper_bound'] for i in range(len(machine_outcome_avg))], \
                 color = 'r', linestyle = '--', label = label_str2)
        title_str1 = "Windowed average outcome (avg over {} turns) wrt. turns played".format(window_size)
        title_str2 = "\n(Best machine = {}, best avg outcome for machine = {})".format(self.display_parameters['best_machine'], \
                                                                                       self.display_parameters['upper_bound'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Average outcome')
        plt.ylim(self.display_parameters['min_val'], self.display_parameters['max_val'])
        plt.legend(loc = 'best')
        plt.show()
        
        # Display average performance over time for agent (non-normalized axis)
        plt.figure(figsize = (10, 7))
        label_str1 = 'Avg. performance of agent'
        label_str2 = 'Best performance (th. avg.)'
        plt.plot([i*window_size for i in range(len(machine_outcome_avg))], machine_outcome_avg, label = label_str1)
        plt.plot([i*window_size for i in range(len(machine_outcome_avg))], \
                 [self.display_parameters['upper_bound'] for i in range(len(machine_outcome_avg))], \
                 color = 'r', linestyle = '--', label = label_str2)
        title_str1 = "Windowed average outcome (avg over {} turns) wrt. turns played".format(window_size)
        title_str2 = "\n(Best machine = {}, best avg outcome for machine = {})".format(self.display_parameters['best_machine'], \
                                                                                       self.display_parameters['upper_bound'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Average outcome')
        #plt.ylim(self.display_parameters['min_val'], self.display_parameters['max_val'])
        plt.legend(loc = 'best')
        plt.show()
        
        
    def display_regret_over_time(self, scaling_factor = 100):
        
        # Display regret performance over time for agent
        window_size = max(1, int(self.initial_money/scaling_factor))
        self.assert_window_size(window_size)
        machine_regret = self.history['Regret']
        machine_regret_avg = [np_mean(machine_regret[i:min(i + window_size, len(machine_regret))]) \
                               for i in range(0, len(machine_regret), window_size)]
        
        # Display regret performance over time for agent (normalized axis)
        plt.figure(figsize = (10, 7))
        plt.plot([i*window_size for i in range(len(machine_regret_avg))], machine_regret_avg)
        title_str = "Windowed average regret (avg over {} turns) wrt. turns played".format(window_size)
        plt.title(title_str)
        plt.xlabel('Turn number')
        plt.ylabel('Average regret')
        plt.show()
        
    
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
    
    def assert_random_seed(self, random_seed = None):
        
        '''
        Note: To be updated later to take into account custom values for random_seeds.
        '''
        if isinstance(random_seed, int):
            self.random_seed = random_seed
            np_seed(random_seed)
        else:
            #print("Warning: Invalid value passed for random_seed parameters, using None instead.")
            self.random_seed = None
            
    
    def assert_window_size(self, window_size = 1):
        
        if not (isinstance(window_size, int) and window_size >= 1):
            err_str1 = "Window size should be a strictly positive integer "
            err_str2 = "(you passed window_size = {}).".format(window_size)
            assert False, err_str1 + err_str2
            
        
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    -------------------------------------------------------------------------------------------------
    '''
    
    def describe(self):
        
        # Describe object method for agent (to be updated later)
        print(self.__dict__)
        
    
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: UserAgent
-------------------------------------------------------------------------------------------------
'''

class UserAgent(Agent):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_seed = None):
        
        # Reuse super class __init__
        super().__init__(list_of_machines, playable_money, random_seed)
        
        # Initialize agent_type
        self.agent_type = 'user'
        
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def get_machine_to_play(self):
        
        # Display separator
        print("-----")
        
        while True:
            # Asking for user input
            machine_number_input = input("Select a machine number to play: ")
            
            # Check if input is valid (can be interpreted as integer)
            try:
                machine_number = int(machine_number_input)
                # Check that machine_number is in [1, self.total_number_of_machines]
                if machine_number >= 1 and machine_number <= self.total_number_of_machines:
                    break
                else:
                    err_str1 = "Warning: Invalid user input "
                    err_str2 = "(you typed {}, which is not in ".format(machine_number_input)
                    err_str3 = "[1, number_of_machines] = [1, {}]).".format(self.total_number_of_machines)
                    print(err_str1 + err_str2 + err_str3)
            except:
                err_str1 = "Warning: Invalid user input (you typed "
                err_str2 = "{}, which could not be interpreted as an integer).".format(machine_number_input)
                print(err_str1 + err_str2)
            
        # Return machine selected by user if it passes all the tests
        return machine_number, None
    
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    # None
    -------------------------------------------------------------------------------------------------
    '''
    
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing Agent class descriptor 
    -------------------------------------------------------------------------------------------------
    '''
    
    
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: DeterministicAgent
-------------------------------------------------------------------------------------------------
'''

class DeterministicAgent(Agent):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_seed = None):
        
        # Reuse super class __init__
        super().__init__(list_of_machines, playable_money, random_seed)
        
        # Initialize agent_type
        self.agent_type = 'deterministic'
        
        # List of deterministic machines
        list_determ_machines = []
        for machine in list_of_machines:
            if machine.machine_type == 'deterministic':
                list_determ_machines.append(machine)
                
        # Find best machine
        assert len(list_determ_machines) > 0, "No deterministic machines to play on."
        self.best_machine = list_determ_machines[0].machine_number
        machine_val = list_of_machines[0].expected_reward()
        for machine in list_determ_machines:
            reward = machine.expected_reward()
            if machine_val < reward:
                self.best_machine = machine.machine_number
                machine_val = reward
                
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def get_machine_to_play(self):
        # Return the best deterministic machine everytime (i.e. the deterministic machine with the highest expected reward)
        return self.best_machine, None
    
    
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    # None
    -------------------------------------------------------------------------------------------------
    '''
    
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing Agent class descriptor 
    -------------------------------------------------------------------------------------------------
    '''


'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: PriorKnowledgeAgent
-------------------------------------------------------------------------------------------------
'''

class PriorKnowledgeAgent(Agent):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_seed = None):
        
        # Reuse super class __init__
        super().__init__(list_of_machines, playable_money, random_seed)
        
        # Initialize agent_type
        self.agent_type = 'prior_knowledge'
                
        # Find best machine
        self.best_machine = list_of_machines[0].machine_number
        machine_val = list_of_machines[0].expected_reward()
        for machine in list_of_machines:
            reward = machine.expected_reward()
            if machine_val < reward:
                self.best_machine = machine.machine_number
                machine_val = reward
                
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def get_machine_to_play(self):
        # Return the best machine everytime (i.e. the machine with the highest expected reward)
        return self.best_machine, None
    
    
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    # None
    -------------------------------------------------------------------------------------------------
    '''
    
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing Agent class descriptor 
    -------------------------------------------------------------------------------------------------
    '''


'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: RandomAgent
-------------------------------------------------------------------------------------------------
'''

class RandomAgent(Agent):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_type = 'naive', random_seed = None):
        
        # Reuse super class __init__
        super().__init__(list_of_machines, playable_money, random_seed)
        
        # Initialize agent_type
        self.agent_type = 'random'
        self.random_type = random_type
        
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def get_machine_to_play(self):
        
        # Choose random machine (naive)
        if self.random_type == 'naive':
            return np_randint(1, self.total_number_of_machines + 1), None
        
        # Choose random machine (weighted linear)
        if self.random_type == 'linear':
            # Compute thresholds for each machine
            total_value = sum(self.machine_estimates.values())
            machine_thr = {val: self.machine_estimates[val]/total_value for val in self.machine_estimates.keys()}
            # Roll
            roll_value = np_random()
            # Retrieve machine
            machine_index = 0
            while roll_value > 0:
                machine_number = list(machine_thr.keys())[machine_index]
                roll_value -= machine_thr[machine_number]
                machine_index += 1
            return machine_number, None
        
        # Choose random machine (weighted softmax)
        if self.random_type == 'softmax':
            # Compute thresholds for each machine
            total_value = sum(np_exp(list(self.machine_estimates.values())))
            machine_thr = {val: np_exp(self.machine_estimates[val])/total_value for val in self.machine_estimates.keys()}
            # Roll
            roll_value = np_random()
            # Retrieve machine
            machine_index = 0
            while roll_value > 0:
                machine_number = list(machine_thr.keys())[machine_index]
                roll_value -= machine_thr[machine_number]
                machine_index += 1 
            return machine_number, None
        
        
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    # None
    -------------------------------------------------------------------------------------------------
    '''
    
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing Agent class descriptor 
    -------------------------------------------------------------------------------------------------
    '''
    
    
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: EpsilonFirstAgent
-------------------------------------------------------------------------------------------------
'''

class EpsilonFirstAgent(Agent):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_type = 'naive', epsilon_value = 0.1, \
                 random_seed = None, softmax_weights = 3):
        
        # Reuse super class __init__
        super().__init__(list_of_machines, playable_money, random_seed)
        
        # Assert and initialize epsilon value
        self.assert_epsilon_value(epsilon_value)
        
        # Assert softmax_weights
        if random_type == 'softmax':
            self.assert_softmax_weights(softmax_weights)
        
        # Initialize agent_type
        self.agent_type = 'epsilon_first'
        self.random_type = random_type
        
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
        
    def get_machine_to_play(self):
        
        # Check if it is an exploration or exploitation play
        explore = (self.initial_money - self.playable_money + 1) < (self.initial_money*self.epsilon_value)
        
        ## 1. Exploration play
        if explore:
            
            # Choose random machine (naive)
            if self.random_type == 'naive':
                return np_randint(1, self.total_number_of_machines + 1), True

            # Choose random machine (weighted linear)
            if self.random_type == 'linear':
                # Compute thresholds for each machine
                total_value = sum(self.machine_estimates.values())
                machine_thr = {val: self.machine_estimates[val]/total_value for val in self.machine_estimates.keys()}
                # Roll
                roll_value = np_random()
                # Retrieve machine
                machine_index = 0
                while roll_value > 0:
                    machine_number = list(machine_thr.keys())[machine_index]
                    roll_value -= machine_thr[machine_number]
                    machine_index += 1 
                return machine_number, True

            # Choose random machine (weighted softmax)
            if self.random_type == 'softmax':
                # Compute thresholds for each machine
                total_value = sum(np_exp(list(i*self.softmax_weights for i in self.machine_estimates.values())))
                machine_thr = {val: np_exp(self.softmax_weights*self.machine_estimates[val])/total_value \
                               for val in self.machine_estimates.keys()}
                # Roll
                roll_value = np_random()
                # Retrieve machine
                machine_index = 0
                while roll_value > 0:
                    machine_number = list(machine_thr.keys())[machine_index]
                    roll_value -= machine_thr[machine_number]
                    machine_index += 1 
                return machine_number, True
            
        ## 2. Exploitation play
        else:
            # Use the machine with the maximal estimated expectation
            machine_number = max(self.machine_estimates, key = self.machine_estimates.get) 
            return machine_number, False
        
        
    def display_machines_time_played(self, scaling_factor = 100):
        
        # Display warning if not all coins have been used
        if self.playable_money > 0:
            warning_str1 = "Warning: you still have playable money left ({} coins remaining).".format(self.playable_money)
            warning_str2 = "\nDisplaying graphs anyway."
            print(warning_str1 + warning_str2)
        
        # Compute number of coins used in exploration phase
        expl_coin_number = int(round(self.epsilon_value*self.initial_money))
        
        ## Display number of times each machine has been played (during exploration phase en in total)
        # Retrieve values from history
        plt.figure(figsize = (10, 7))
        machines_times_played_total = self.machine_times_played
        machine_numbers_expl = self.history['Machine_number'][0:expl_coin_number]
        machines_times_played_expl = [machine_numbers_expl.count(machine_index) \
                                      for machine_index in list(machines_times_played_total.keys())]
        shift = 0.2
        # Plot bars
        plt.bar([i - shift for i in machines_times_played_total.keys()], \
                list(machines_times_played_total.values()), width = 2*shift, color = 'b', label = 'Total times played')
        plt.bar([i + shift for i in machines_times_played_total.keys()], \
                machines_times_played_expl, width = 2*shift, color = 'r', label = 'Times played during exploration')
        # Add labels above bars
        v_shift = max(3, max(list(machines_times_played_total.values()))/50)
        for i, v in enumerate(list(machines_times_played_total.values())):
            plt.text(i + 1 - 3/2*shift, v + v_shift, str(v), color='blue', fontweight='bold')
        for i, v in enumerate(machines_times_played_expl):
            plt.text(i + 1 + 3/4*shift, v + v_shift, str(v), color='red', fontweight='bold')
        # Description
        title_str1 = "Number of times each machine has been played by agent."
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Machine number')
        plt.ylabel('Number of times played')
        plt.legend(loc = 'best')
        plt.show()
        
        # Display which machine was played on each turn
        plt.figure(figsize = (10, 7))
        plt.plot([i for i in range(len(machine_numbers_expl))], machine_numbers_expl)
        title_str1 = "Machine played on each turn by agent during exploration phase."
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Machine number')
        plt.show()
        
        
        # Display which machine was played on each turn
        window_size = max(1, int(expl_coin_number/scaling_factor))
        self.assert_window_size(window_size)
        machine_numbers_avg = [max(set(machine_numbers_expl[i:min(i + window_size, len(machine_numbers_expl))]), \
                               key = machine_numbers_expl[i:min(i + window_size, len(machine_numbers_expl))].count) \
                               for i in range(0, len(machine_numbers_expl), window_size)]
        plt.figure(figsize = (10, 7))
        plt.plot([i*window_size for i in range(len(machine_numbers_avg))], machine_numbers_avg)
        title_str1 = "Machine played on each turn by agent (machine most played over window of {} coins).".format(window_size)
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Machine number')
        plt.show()
        
        
        
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
    
    def assert_epsilon_value(self, epsilon_value):
        
        # Initialize and assert random_parameters
        if isinstance(epsilon_value, (int, float)) and epsilon_value >= 0 and epsilon_value <=1:
            self.epsilon_value = epsilon_value
        else:
            err_str1 = "Variable epsilon_value should be a number in [0,1],"
            err_str2 = " you passed epsilon_value = {}.".format(epsilon_value)
            assert False, err_str1 + err_str2
            
    
    def assert_softmax_weights(self, softmax_weights):
        if isinstance(softmax_weights, (int, float)) and softmax_weights > 0:
            self.softmax_weights = softmax_weights
        else:
            err_str1 = "Variable softmax_weights should be a strictly positive number,"
            err_str2 = " you passed softmax_weights = {}.".format(softmax_weights)
            assert False, err_str1 + err_str2
    
    
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing Agent class descriptor 
    -------------------------------------------------------------------------------------------------
    '''
        
        
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: EpsilonGreedyAgent
-------------------------------------------------------------------------------------------------
'''

class EpsilonGreedyAgent(Agent):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_type = 'naive', epsilon_value = 0.1, random_seed = None):
        
        # Reuse super class __init__
        super().__init__(list_of_machines, playable_money, random_seed)
        
        # Assert and initialize epsilon value
        self.assert_epsilon_value(epsilon_value)
        
        # Initialize agent_type
        self.agent_type = 'epsilon_greedy'
        self.random_type = random_type
       
            
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def get_machine_to_play(self):
        
        # Check if it is an exploration or exploitation play
        explore = np_random() < self.epsilon_value
        
        ## 1. Exploration play
        if explore:
            
            # Choose random machine (naive)
            if self.random_type == 'naive':
                return np_randint(1, self.total_number_of_machines + 1), True

            # Choose random machine (weighted linear)
            if self.random_type == 'linear':
                # Compute thresholds for each machine
                total_value = sum(self.machine_estimates.values())
                machine_thr = {val: self.machine_estimates[val]/total_value for val in self.machine_estimates.keys()}
                # Roll
                roll_value = np_random()
                # Retrieve machine
                machine_index = 0
                while roll_value > 0:
                    machine_number = list(machine_thr.keys())[machine_index]
                    roll_value -= machine_thr[machine_number]
                    machine_index += 1
                return machine_number, True

            # Choose random machine (weighted softmax)
            if self.random_type == 'softmax':
                # Compute thresholds for each machine
                total_value = sum(np_exp(list(self.machine_estimates.values())))
                machine_thr = {val: np_exp(self.machine_estimates[val])/total_value for val in self.machine_estimates.keys()}
                # Roll
                roll_value = np_random()
                # Retrieve machine
                machine_index = 0
                while roll_value > 0:
                    machine_number = list(machine_thr.keys())[machine_index]
                    roll_value -= machine_thr[machine_number]
                    machine_index += 1 
                return machine_number, True
            
        ## 2. Exploitation play
        else:
            # Use the machine with the maximal estimated expectation
            machine_number = max(self.machine_estimates, key = self.machine_estimates.get) 
            return machine_number, False
        
    
    def display_machines_time_played(self, scaling_factor = 100):
        
        # Display warning if not all coins have been used
        if self.playable_money > 0:
            warning_str1 = "Warning: you still have playable money left ({} coins remaining).".format(self.playable_money)
            warning_str2 = "\nDisplaying graphs anyway."
            print(warning_str1 + warning_str2)
        
        # Compute number of coins used in exploration phase
        expl_coin_number = int(round(self.epsilon_value*self.initial_money))
        
        ## Display number of times each machine has been played (during exploration phase en in total)
        # Retrieve values from history
        plt.figure(figsize = (10, 7))
        machines_times_played_total = self.machine_times_played
        machine_numbers_expl = self.history['Machine_number'][0:expl_coin_number]
        machines_times_played_expl = [machine_numbers_expl.count(machine_index) \
                                      for machine_index in list(machines_times_played_total.keys())]
        shift = 0.2
        # Plot bars
        plt.bar([i - shift for i in machines_times_played_total.keys()], \
                list(machines_times_played_total.values()), width = 2*shift, color = 'b', label = 'Total times played')
        plt.bar([i + shift for i in machines_times_played_total.keys()], \
                machines_times_played_expl, width = 2*shift, color = 'r', label = 'Times played during exploration')
        # Add labels above bars
        v_shift = max(3, max(list(machines_times_played_total.values()))/50)
        for i, v in enumerate(list(machines_times_played_total.values())):
            plt.text(i + 1 - 3/2*shift, v + v_shift, str(v), color='blue', fontweight='bold')
        for i, v in enumerate(machines_times_played_expl):
            plt.text(i + 1 + 3/4*shift, v + v_shift, str(v), color='red', fontweight='bold')
        # Description
        title_str1 = "Number of times each machine has been played by agent."
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Machine number')
        plt.ylabel('Number of times played')
        plt.legend(loc = 'best')
        plt.show()
        
        # Display which machine was played on each turn
        plt.figure(figsize = (10, 7))
        plt.plot([i for i in range(len(machine_numbers_expl))], machine_numbers_expl)
        title_str1 = "Machine played on each turn by agent during exploration phase."
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Machine number')
        plt.show()
        
        
        # Display which machine was played on each turn
        window_size = max(1, int(expl_coin_number/scaling_factor))
        self.assert_window_size(window_size)
        machine_numbers_avg = [max(set(machine_numbers_expl[i:min(i + window_size, len(machine_numbers_expl))]), \
                               key = machine_numbers_expl[i:min(i + window_size, len(machine_numbers_expl))].count) \
                               for i in range(0, len(machine_numbers_expl), window_size)]
        plt.figure(figsize = (10, 7))
        plt.plot([i*window_size for i in range(len(machine_numbers_avg))], machine_numbers_avg)
        title_str1 = "Machine played on each turn by agent (machine most played over window of {} coins).".format(window_size)
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Machine number')
        plt.show()
        
        
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
        
    def assert_epsilon_value(self, epsilon_value):
        
        # Initialize and assert random_parameters
        if isinstance(epsilon_value, (int, float)) and epsilon_value >= 0 and epsilon_value <=1:
            self.epsilon_value = epsilon_value
        else:
            assert False, "Variable epsilon_value should be a number in [0,1], you passed epsilon_value = {}.".format(epsilon_value)
            
            
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing Agent class descriptor 
    -------------------------------------------------------------------------------------------------
    '''
        
        
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: EpsilonFirstGreedyAgent
-------------------------------------------------------------------------------------------------
'''

class EpsilonFirstGreedyAgent(Agent):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_type = 'naive', \
                 epsilon_values = [0.1, 0.1], random_seed = None):
        
        # Reuse super class __init__
        super().__init__(list_of_machines, playable_money, random_seed)
        
        # Assert and initialize epsilon values
        self.assert_epsilon_values(epsilon_values)
        
        # Initialize agent_type
        self.agent_type = 'epsilon_first_greedy'
        self.random_type = random_type
        
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def get_machine_to_play(self):
        
        # Check if it is an exploration or exploitation play
        explore = (self.initial_money - self.playable_money + 1) < (self.initial_money*self.epsilon_value1)
        if not explore:
            explore = np_random() < self.epsilon_value2
        
        ## 1. Exploration play
        if explore:
            
            # Choose random machine (naive)
            if self.random_type == 'naive':
                return np_randint(1, self.total_number_of_machines + 1), True

            # Choose random machine (weighted linear)
            if self.random_type == 'linear':
                # Compute thresholds for each machine
                total_value = sum(self.machine_estimates.values())
                machine_thr = {val: self.machine_estimates[val]/total_value for val in self.machine_estimates.keys()}
                # Roll
                roll_value = np_random()
                # Retrieve machine
                machine_index = 0
                while roll_value > 0:
                    machine_number = list(machine_thr.keys())[machine_index]
                    roll_value -= machine_thr[machine_number]
                    machine_index += 1 
                return machine_number, True

            # Choose random machine (weighted softmax)
            if self.random_type == 'softmax':
                # Compute thresholds for each machine
                total_value = sum(np_exp(list(self.machine_estimates.values())))
                machine_thr = {val: np_exp(self.machine_estimates[val])/total_value for val in self.machine_estimates.keys()}
                # Roll
                roll_value = np_random()
                # Retrieve machine
                machine_index = 0
                while roll_value > 0:
                    machine_number = list(machine_thr.keys())[machine_index]
                    roll_value -= machine_thr[machine_number]
                    machine_index += 1 
                return machine_number, True
            
        ## 2. Exploitation play
        else:
            # Use the machine with the maximal estimated expectation
            machine_number = max(self.machine_estimates, key = self.machine_estimates.get) 
            return machine_number, False
        
    
    def display_machines_time_played(self, scaling_factor = 100):
        
        # Display warning if not all coins have been used
        if self.playable_money > 0:
            warning_str1 = "Warning: you still have playable money left ({} coins remaining).".format(self.playable_money)
            warning_str2 = "\nDisplaying graphs anyway."
            print(warning_str1 + warning_str2)
        
        # Compute number of coins used in exploration phase
        val1 = int(round(self.epsilon_value1*self.initial_money))
        val2 = int(round(self.epsilon_value1*(self.initial_money - val1)))
        expl_coin_number =  val1+ val2
        
        ## Display number of times each machine has been played (during exploration phase en in total)
        # Retrieve values from history
        plt.figure(figsize = (10, 7))
        machines_times_played_total = self.machine_times_played
        machine_numbers_expl = self.history['Machine_number'][0:expl_coin_number]
        machines_times_played_expl = [machine_numbers_expl.count(machine_index) \
                                      for machine_index in list(machines_times_played_total.keys())]
        shift = 0.2
        # Plot bars
        plt.bar([i - shift for i in machines_times_played_total.keys()], \
                list(machines_times_played_total.values()), width = 2*shift, color = 'b', label = 'Total times played')
        plt.bar([i + shift for i in machines_times_played_total.keys()], \
                machines_times_played_expl, width = 2*shift, color = 'r', label = 'Times played during exploration')
        # Add labels above bars
        v_shift = max(3, max(list(machines_times_played_total.values()))/50)
        for i, v in enumerate(list(machines_times_played_total.values())):
            plt.text(i + 1 - 3/2*shift, v + v_shift, str(v), color='blue', fontweight='bold')
        for i, v in enumerate(machines_times_played_expl):
            plt.text(i + 1 + 3/4*shift, v + v_shift, str(v), color='red', fontweight='bold')
        # Description
        title_str1 = "Number of times each machine has been played by agent."
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Machine number')
        plt.ylabel('Number of times played')
        plt.legend(loc = 'best')
        plt.show()
        
        # Display which machine was played on each turn
        plt.figure(figsize = (10, 7))
        plt.plot([i for i in range(len(machine_numbers_expl))], machine_numbers_expl)
        title_str1 = "Machine played on each turn by agent during exploration phase."
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Machine number')
        plt.show()
        
        
        # Display which machine was played on each turn
        window_size = max(1, int(expl_coin_number/scaling_factor))
        self.assert_window_size(window_size)
        machine_numbers_avg = [max(set(machine_numbers_expl[i:min(i + window_size, len(machine_numbers_expl))]), \
                               key = machine_numbers_expl[i:min(i + window_size, len(machine_numbers_expl))].count) \
                               for i in range(0, len(machine_numbers_expl), window_size)]
        plt.figure(figsize = (10, 7))
        plt.plot([i*window_size for i in range(len(machine_numbers_avg))], machine_numbers_avg)
        title_str1 = "Machine played on each turn by agent (machine most played over window of {} coins).".format(window_size)
        title_str2 = "\n(Best machine to play: {})".format(self.display_parameters['best_machine'])
        plt.title(title_str1 + title_str2)
        plt.xlabel('Turn number')
        plt.ylabel('Machine number')
        plt.show()
        
        
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
    
    def assert_epsilon_values(self, epsilon_values):
        
        # Initialize and assert random_parameters
        if not isinstance(epsilon_values, list):
            err_str1 = "Variable epsilon_values should be a list of two numbers in [0,1], "
            err_str2 = "you passed epsilon_values = {}.".format(epsilon_values)
        elif not len(epsilon_values) == 2:
            err_str1 = "Variable epsilon_values should be a list of two numbers in [0,1], "
            err_str2 = "you passed epsilon_values = {}.".format(epsilon_values)
        else:
            bool1 = isinstance(epsilon_values[0], (int, float)) and epsilon_values[0] >= 0 and epsilon_values[0] <=1
            bool2 = isinstance(epsilon_values[1], (int, float)) and epsilon_values[1] >= 0 and epsilon_values[1] <=1
        if bool1 and bool2:
            self.epsilon_value1 = epsilon_values[0]
            self.epsilon_value2 = epsilon_values[1]
        else:
            err_str1 = "Variable epsilon_values should be a list of two numbers in [0,1], "
            err_str2 = "you passed epsilon_values = {}.".format(epsilon_values)
            assert False, err_str1 + err_str2
            
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing Agent class descriptor 
    -------------------------------------------------------------------------------------------------
    '''
            
        
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: EpsilonGreedyDecayAgent
-------------------------------------------------------------------------------------------------
'''

class EpsilonGreedyDecayAgent(Agent):
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, list_of_machines, playable_money, random_type = 'naive', epsilon_value = 1, \
                 decay_ratio = 0.9, decay_step = 0.05, random_seed = None):
        
        # Reuse super class __init__
        super().__init__(list_of_machines, playable_money, random_seed)
        
        # Initialize epsilon values
        self.epsilon_value = epsilon_value
        self.initial_epsilon_value = epsilon_value
        
        # Initialize decay step size and ratio
        self.decay_step = decay_step
        self.decay_step_size = int(round(self.decay_step*self.initial_money))
        self.decay_ratio = decay_ratio
        
        # Initialize decay values dictionary
        self.decay_values = {0: self.initial_epsilon_value}
        
        # Initialize agent_type
        self.agent_type = 'epsilon_greedy_decay'
        self.random_type = random_type
        
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def update_decay_rate(self):
        
        # Check not first coin
        bool1 = (self.playable_money != self.initial_money) 
        # Check for decay step
        bool2 = ((self.initial_money - self.playable_money) % self.decay_step_size == 0)
        # If needed, decay value of epsilon
        if bool1 and bool2:
            self.epsilon_value = self.epsilon_value*self.decay_ratio
            self.decay_values[self.initial_money - self.playable_money] = self.epsilon_value
            
            
    def display_decay_values(self):
        
        # Display decay rate
        plt.figure()
        plt.plot(list(self.decay_values.keys()), list(self.decay_values.values()))
        plt.title('Epsilon value decay over turns')
        plt.xlabel('Number of coins played')
        plt.ylabel('Epsilon value')
        plt.show()
            
            
    def get_machine_to_play(self):
        
        # Check if it is an exploration or exploitation play
        explore = np_random() < self.epsilon_value
        
        ## 1. Exploration play
        if explore:
            
            # Choose random machine (naive)
            if self.random_type == 'naive':
                return np_randint(1, self.total_number_of_machines + 1), True

            # Choose random machine (weighted linear)
            if self.random_type == 'linear':
                # Compute thresholds for each machine
                total_value = sum(self.machine_estimates.values())
                machine_thr = {val: self.machine_estimates[val]/total_value for val in self.machine_estimates.keys()}
                # Roll
                roll_value = np_random()
                # Retrieve machine
                machine_index = 0
                while roll_value > 0:
                    machine_number = list(machine_thr.keys())[machine_index]
                    roll_value -= machine_thr[machine_number]
                    machine_index += 1 
                return machine_number, True

            # Choose random machine (weighted softmax)
            if self.random_type == 'softmax':
                # Compute thresholds for each machine
                total_value = sum(np_exp(list(self.machine_estimates.values())))
                machine_thr = {val: np_exp(self.machine_estimates[val])/total_value for val in self.machine_estimates.keys()}
                # Roll
                roll_value = np_random()
                # Retrieve machine
                machine_index = 0
                while roll_value > 0:
                    machine_number = list(machine_thr.keys())[machine_index]
                    roll_value -= machine_thr[machine_number]
                    machine_index += 1 
                return machine_number, True
            
        ## 2. Exploitation play
        else:
            # Use the machine with the maximal estimated expectation
            machine_number = max(self.machine_estimates, key = self.machine_estimates.get) 
            
            return machine_number, False
        
        
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    # None
    -------------------------------------------------------------------------------------------------
    '''
    
    def assert_epsilon_values(self, epsilon_values):
        
        # Initialize and assert random_parameters
        if not isinstance(epsilon_values, list):
            err_str1 = "Variable epsilon_values should be a list of two numbers in [0,1], "
            err_str2 = "you passed epsilon_values = {}.".format(epsilon_values)
        elif not len(epsilon_values) == 2:
            err_str1 = "Variable epsilon_values should be a list of two numbers in [0,1], "
            err_str2 = "you passed epsilon_values = {}.".format(epsilon_values)
        else:
            bool1 = isinstance(epsilon_values[0], (int, float)) and epsilon_values[0] >= 0 and epsilon_values[0] <=1
            bool2 = isinstance(epsilon_values[1], (int, float)) and epsilon_values[1] >= 0 and epsilon_values[1] <=1
        if bool1 and bool2:
            self.epsilon_value1 = epsilon_values[0]
            self.epsilon_value2 = epsilon_values[1]
        else:
            err_str1 = "Variable epsilon_values should be a list of two numbers in [0,1], "
            err_str2 = "you passed epsilon_values = {}.".format(epsilon_values)
            assert False, err_str1 + err_str2
            
            
    '''
    -------------------------------------------------------------------------------------------------
    4. Descriptor
    # None, reusing Agent class descriptor 
    -------------------------------------------------------------------------------------------------
    '''
        