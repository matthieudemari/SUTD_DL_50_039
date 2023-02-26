'''
-------------------------------------------------------------------------------------------------
CLASS IMPORTS
-------------------------------------------------------------------------------------------------
'''

# External libraries
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.random import seed as np_seed
from numpy.random import random as np_random
from numpy.random import randint as np_randint
from numpy import cumsum as np_cumsum
from numpy import exp as np_exp
from numpy import mean as np_mean
from numpy import histogram as np_hist
from numpy import arange as np_arange
from numpy import array as np_array
from numpy import sum as np_sum
from numpy import log10 as np_log10
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import max as np_max
from numpy import argmax as np_amax
from numpy import min as np_min
from numpy import argmin as np_amin
from numpy import unravel_index as np_urvl
from numpy import meshgrid as np_mesh
import pandas as pd
from tqdm import tqdm

# Internal libraries
from RandomCandyMachineGame import *
        
        
'''
-------------------------------------------------------------------------------------------------
CLASS DEFINITION: MonteCarloRCM
-------------------------------------------------------------------------------------------------
'''

class MonteCarloRCM():
    
    '''
    -------------------------------------------------------------------------------------------------
    1. Constructor
    -------------------------------------------------------------------------------------------------
    '''
    
    def __init__(self, machines_cost = 1, \
                 playable_money = 1000, \
                 number_of_deterministic_machines = 1, \
                 return_prices_for_deterministic_machines = 1, \
                 return_prices_for_random_machines_win = 2, \
                 return_prices_for_random_machines_loss = 0, \
                 number_of_random_machines = 4, \
                 random_parameters = None, \
                 n_iter = 1000, \
                 agent_list = None, \
                 display_bool = True):
        
        # Machine parameters
        self.machines_cost = machines_cost
        self.playable_money = playable_money
        self.number_of_deterministic_machines = number_of_deterministic_machines
        self.number_of_random_machines = number_of_random_machines
        self.return_prices_for_deterministic_machines = return_prices_for_deterministic_machines
        self.return_prices_for_random_machines_win = return_prices_for_random_machines_win
        self.return_prices_for_random_machines_loss = return_prices_for_random_machines_loss
        self.random_parameters = random_parameters
        
        # MonteCarlo iterations number
        self.n_iter = n_iter
        
        # Agent types list
        if agent_list == None:
            self.agent_list = ['deterministic', 'prior_knowledge', \
                               'random_naive', 'random_weighted_linear', 'random_weighted_softmax', \
                               'epsilon_first_naive', 'epsilon_first_linear', 'epsilon_first_softmax', \
                               'epsilon_greedy_naive', 'epsilon_greedy_linear', 'epsilon_greedy_softmax', \
                               'epsilon_firstgreedy_naive', 'epsilon_firstgreedy_linear', 'epsilon_firstgreedy_softmax', \
                               'epsilon_greedydecay_naive', 'epsilon_greedydecay_linear', 'epsilon_greedydecay_softmax']
        else:
            self.assert_agent_list(agent_list)
        
        # Define displaystyles
        self.define_linestyles_for_display()
        
        # Display boolean
        self.display_bool = display_bool
        
        # Dictionaries for storing results
        self.outcome_avg = {agent: [0 for i in range(self.n_iter)] for agent in self.agent_list}
        self.regret_avg = {agent: [0 for i in range(self.n_iter)] for agent in self.agent_list}
        self.scaling_factor = 100
        self.window_size = max(1, int(self.playable_money/self.scaling_factor))
        self.performance_over_time = {agent: np_array([0 for j in range(self.scaling_factor)]) for agent in self.agent_list}
        self.regret_over_time = {agent: np_array([0 for j in range(self.scaling_factor)]) for agent in self.agent_list}
        
        
    '''
    -------------------------------------------------------------------------------------------------
    2. Methods
    -------------------------------------------------------------------------------------------------
    '''
    
    def define_linestyles_for_display(self):
        
        # Initialize colors, linestyles and markers lists
        self.agent_colors = []
        self.agent_linestyles = []
        self.agent_markers = []
        
        for agent_type in self.agent_list:
            if 'deterministic' in agent_type:
                color = 'b'
                linestyle = '-'
                marker = ','
            elif 'prior_knowledge' in agent_type:
                color = 'g'
                linestyle = '-'
                marker = ','
            elif 'random' in agent_type:
                color = 'r'
                if 'naive' in agent_type:
                    linestyle = '--'
                    marker = 'x'
                elif 'linear' in agent_type:
                    linestyle = '-.'
                    marker = '+'
                else:
                    linestyle = '-'
                    marker = ','
            elif 'epsilon_first_' in agent_type:
                color = 'c'
                if 'naive' in agent_type:
                    linestyle = '--'
                    marker = 'x'
                elif 'linear' in agent_type:
                    linestyle = '-.'
                    marker = '+'
                else:
                    linestyle = '-'
                    marker = ','
            elif 'epsilon_greedy_' in agent_type:
                color = 'm'
                if 'naive' in agent_type:
                    linestyle = '--'
                    marker = 'x'
                elif 'linear' in agent_type:
                    linestyle = '-.'
                    marker = '+'
                else:
                    linestyle = '-'
                    marker = ','
            elif 'epsilon_firstgreedy_' in agent_type:
                color = 'y'
                if 'naive' in agent_type:
                    linestyle = '--'
                    marker = 'x'
                elif 'linear' in agent_type:
                    linestyle = '-.'
                    marker = '+'
                else:
                    linestyle = '-'
                    marker = ','
            elif 'epsilon_greedydecay_' in agent_type:
                color = 'k'
                if 'naive' in agent_type:
                    linestyle = '--'
                    marker = 'x'
                elif 'linear' in agent_type:
                    linestyle = '-.'
                    marker = '+'
                else:
                    linestyle = '-'
                    marker = ','
            
            # Add to lists
            self.agent_colors.append(color)
            self.agent_linestyles.append(linestyle)
            self.agent_markers.append(marker)
    
    
    def run_all_iterations(self):
        
        # Iinitialize waitbar
        if self.display_bool:
            desc_str = "Monte Carlo over {} iterations".format(self.n_iter)
            t = tqdm(desc = desc_str, total = self.n_iter)
        
        for random_seed in range(1, self.n_iter + 1):
            for agent_type in self.agent_list:
                
                # Initialize the CandyMachineGame
                cmg = CandyMachineGame(machines_cost = 1, \
                                       playable_money = self.playable_money, \
                                       number_of_deterministic_machines = 1, \
                                       number_of_random_machines = self.number_of_random_machines, \
                                       return_prices_for_deterministic_machines = self.return_prices_for_deterministic_machines, \
                                       return_prices_for_random_machines_win = self.return_prices_for_random_machines_win, \
                                       return_prices_for_random_machines_loss = self.return_prices_for_random_machines_loss, \
                                       random_parameters = self.random_parameters, \
                                       agent_type = agent_type, \
                                       random_seed = random_seed, \
                                       agent_random_seed = random_seed, \
                                       display_bool = False)
                
                # Play for agent
                avg_performance, avg_regret, agent_outcome_history, agent_regret_history = cmg.play(return_bool = True)
                
                # Store into array
                self.outcome_avg[agent_type][random_seed - 1] = avg_performance
                self.regret_avg[agent_type][random_seed - 1] = avg_regret
                processed_outcome_history = np_array([np_mean(agent_outcome_history[i:min(i + self.window_size, \
                                    len(agent_outcome_history))]) for i in range(0, len(agent_outcome_history), self.window_size)])
                self.performance_over_time[agent_type] = ((random_seed - 1)*self.performance_over_time[agent_type] + \
                                                          processed_outcome_history)/random_seed
                processed_regret_history = np_array([np_mean(agent_regret_history[i:min(i + self.window_size, \
                                   len(agent_regret_history))]) for i in range(0, len(agent_regret_history), self.window_size)])
                self.regret_over_time[agent_type] = ((random_seed - 1)*self.regret_over_time[agent_type] + \
                                                     processed_regret_history)/random_seed
                
            # Update waitbar
            if self.display_bool:
                t.update(1)
                
        # Close waitbar
        if self.display_bool:
            t.close()
            
            
    def display_montecarlo_histograms(self, n_bins = 25):
        
        # Avg performance histogram
        plt.figure(figsize = (10, 7))
        for agent_index in range(len(self.agent_list)):
            agent = self.agent_list[agent_index]
            color = self.agent_colors[agent_index]
            linestyle = self.agent_linestyles[agent_index]
            marker = self.agent_markers[agent_index]
            outcome_vals = self.outcome_avg[agent]
            str_label = "Agent type: {}\n(avg: {})".format(agent, np_mean(outcome_vals))
            n, bins = np_hist(outcome_vals, n_bins)
            plt.plot(bins[:-1], n, color = color, linestyle = linestyle, marker = marker, label = str_label)
        title_str = "Average outcome performance for each strategy played by agent."
        plt.title(title_str)
        plt.xlabel('Average number of candies per coin used')
        plt.ylabel('Density')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
        
        # Avg regret histogram
        plt.figure(figsize = (10, 7))
        for agent_index in range(len(self.agent_list)):
            agent = self.agent_list[agent_index]
            color = self.agent_colors[agent_index]
            linestyle = self.agent_linestyles[agent_index]
            marker = self.agent_markers[agent_index]
            regret_vals = self.regret_avg[agent]
            str_label = "Agent type: {}\n(avg: {})".format(agent, np_mean(regret_vals))
            n, bins = np_hist(regret_vals, n_bins)
            plt.plot(bins[:-1], n, color = color, linestyle = linestyle, marker = marker, label = str_label)
        title_str = "Average regret performance for each strategy played by agent."
        plt.title(title_str)
        plt.xlabel('Average missed number of candies per coin used (regret)')
        plt.ylabel('Density')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
        
        
    def display_average_performance(self):
        
        # Avg performance bar graph
        fig = plt.figure(figsize = (10, 5))
        ax = fig.add_subplot(111)
        mean_perf = []
        for agent_index in range(len(self.agent_list)):
            agent = self.agent_list[agent_index]
            outcome_vals = self.outcome_avg[agent]
            mean_perf.append(np_mean(outcome_vals))
        plt.bar(self.agent_list, mean_perf)
        title_str = "Average outcome performance for each strategy played by agent."
        plt.title(title_str)
        plt.xlabel('Agent strategy')
        plt.ylabel('Average number of candies per coin used')
        ax.set_xticks(np_arange(0, len(self.agent_list), 1))
        #ax.set_xticks(np_arange(-1/2, -1/2 + len(self.agent_list), 1))
        ax.set_xticklabels(self.agent_list, rotation = 45)
        plt.show()
        
        
    def display_average_regret(self):
        
        # Avg regret bar graph
        fig = plt.figure(figsize = (10, 5))
        ax = fig.add_subplot(111)
        mean_perf = []
        for agent_index in range(len(self.agent_list)):
            agent = self.agent_list[agent_index]
            regret_vals = self.regret_avg[agent]
            mean_perf.append(np_mean(regret_vals))
        plt.bar(self.agent_list, mean_perf)
        title_str = "Average regret performance for each strategy played by agent."
        plt.title(title_str)
        plt.xlabel('Agent strategy')
        plt.ylabel('Average number of candies missed per coin used')
        ax.set_xticks(np_arange(0, len(self.agent_list), 1))
        #ax.set_xticks(np_arange(-1/2, -1/2 + len(self.agent_list), 1))
        ax.set_xticklabels(self.agent_list, rotation = 45)
        plt.show()
        
        
    def display_performance_over_time(self):
        
        # Outcome over time graph
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111)
        for agent_index in range(len(self.agent_list)):
            agent = self.agent_list[agent_index]
            color = self.agent_colors[agent_index]
            linestyle = self.agent_linestyles[agent_index]
            marker = self.agent_markers[agent_index]
            performance_over_time = self.performance_over_time[agent]
            str_label = "Agent type: {}\n(avg: {})".format(agent, np_mean(performance_over_time))
            plt.plot(performance_over_time, color = color, linestyle = linestyle, marker = marker, label = str_label)
        title_str = "Window avg outcome over time for all agents (window size = {})".format(self.window_size)
        plt.title(title_str)
        plt.ylabel('Avg outcome')
        plt.xlabel('Turn index')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
    
    
    def display_regret_over_time(self):
        
        # Regret over time graph
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111)
        for agent_index in range(len(self.agent_list)):
            agent = self.agent_list[agent_index]
            color = self.agent_colors[agent_index]
            linestyle = self.agent_linestyles[agent_index]
            marker = self.agent_markers[agent_index]
            regret_over_time = self.regret_over_time[agent]
            str_label = "Agent type: {}\n(avg: {})".format(agent, np_mean(regret_over_time))
            plt.plot(regret_over_time, color = color, linestyle = linestyle, marker = marker, label = str_label)
        title_str = "Window avg regret over time for all agents (window size = {})".format(self.window_size)
        plt.title(title_str)
        plt.ylabel('Avg regret')
        plt.xlabel('Turn index')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
        
    
    def display_cumulated_performance_over_time(self):
        
        # Outcome over time graph
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111)
        for agent_index in range(len(self.agent_list)):
            agent = self.agent_list[agent_index]
            color = self.agent_colors[agent_index]
            linestyle = self.agent_linestyles[agent_index]
            marker = self.agent_markers[agent_index]
            performance_over_time = self.window_size*self.performance_over_time[agent]
            cumulated_performance_over_time = np_cumsum(performance_over_time)
            str_label = "Agent type: {}\n(avg: {})".format(agent, np_mean(performance_over_time))
            plt.plot(cumulated_performance_over_time, color = color, linestyle = linestyle, marker = marker, label = str_label)
        title_str = "Cumulated outcome over time for all agents (window size = {})".format(self.window_size)
        plt.title(title_str)
        plt.ylabel('Cumulated outcome')
        plt.xlabel('Turn index')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
    
    
    def display_cumulated_regret_over_time(self):
        
        # Regret over time graph
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111)
        for agent_index in range(len(self.agent_list)):
            agent = self.agent_list[agent_index]
            color = self.agent_colors[agent_index]
            linestyle = self.agent_linestyles[agent_index]
            marker = self.agent_markers[agent_index]
            regret_over_time = self.window_size*self.regret_over_time[agent]
            cumulated_regret_over_time = np_cumsum(regret_over_time)
            str_label = "Agent type: {}\n(avg: {})".format(agent, np_mean(regret_over_time))
            plt.plot(cumulated_regret_over_time, color = color, linestyle = linestyle, marker = marker, label = str_label)
        title_str = "Cumulated regret over time for all agents (window size = {})".format(self.window_size)
        plt.title(title_str)
        plt.ylabel('Cumulated regret')
        plt.xlabel('Turn index')
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.show()
        
        
    '''
    -------------------------------------------------------------------------------------------------
    3. Setters, getters, asserts on attributes
    -------------------------------------------------------------------------------------------------
    '''
    
    def assert_agent_list(self, agent_list):
        
        # List of possible agent strategies
        agent_list_possible = ['deterministic', 'prior_knowledge', \
                               'random_naive', 'random_weighted_linear', 'random_weighted_softmax', \
                               'epsilon_first_naive', 'epsilon_first_linear', 'epsilon_first_softmax', \
                               'epsilon_greedy_naive', 'epsilon_greedy_linear', 'epsilon_greedy_softmax', \
                               'epsilon_firstgreedy_naive', 'epsilon_firstgreedy_linear', 'epsilon_firstgreedy_softmax', \
                               'epsilon_greedydecay_naive', 'epsilon_greedydecay_linear', 'epsilon_greedydecay_softmax']
        
        # Check if all agent listed in agent_list are in the list of possible agent strategies
        boolean = all([agent in agent_list_possible for agent in agent_list])
        
        # Initialize agent_list attribute accordingly
        if boolean:
            self.agent_list = agent_list
        else:
            assert False, "The agent list contains elements that are not acceptable as agent types."
            
            
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
FUNCTION DEFINITION: MonteCarlo Analysis for impact of epsilon in epsilon-first strategies
-------------------------------------------------------------------------------------------------
'''

def display_impact_epsilon_first(n_seeds = 100, \
                                 epsilon_values_list = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, \
                                                        2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, \
                                                        5e-2, 0.1, 0.2, 0.5, 0.75, 1], \
                                 playable_money = 1000, \
                                 number_of_deterministic_machines = 1, \
                                 number_of_random_machines = 5, \
                                 machines_cost = 1, \
                                 return_prices_for_deterministic_machines = 1, \
                                 return_prices_for_random_machines_win = 2, \
                                 return_prices_for_random_machines_loss = 0):

    # List of seeds
    seeds_list = [i + 1 for i in range(n_seeds)]

    # List of epsilon_values (log10)
    epsilon_values_list_log = np_log10(epsilon_values_list)
    
    # List of agent types
    agent_types_list = ['random_naive', 'prior_knowledge', 'epsilon_first_naive', \
                        'epsilon_first_linear', 'epsilon_first_softmax']

    # Initialize strategies dictionaries
    strategies_performance_outcomes = {agent: [] for agent in agent_types_list}
    strategies_performance_regrets = {agent: [] for agent in agent_types_list}

    # Initialize waitbar
    desc_str = "Testing impact of epsilon value on epsilon-first strategies"
    n_iter = len(epsilon_values_list)*(len(agent_types_list) - 2)*n_seeds
    t = tqdm(desc = desc_str, total = n_iter)

    # Play for all strategies
    for agent_type in strategies_performance_outcomes.keys():
        if agent_type == 'random_naive' or agent_type == 'prior_knowledge':
            outcomes_list = []
            regret_list = []
            for random_seed in seeds_list:
                # Play game with said strategy and epsilon_value
                cmg = CandyMachineGame(playable_money = playable_money, \
                                       number_of_deterministic_machines = number_of_deterministic_machines, \
                                       number_of_random_machines = number_of_random_machines, \
                                       machines_cost = machines_cost, \
                                       return_prices_for_deterministic_machines = return_prices_for_deterministic_machines, \
                                       return_prices_for_random_machines_win = return_prices_for_random_machines_win, \
                                       return_prices_for_random_machines_loss = return_prices_for_random_machines_loss, \
                                       agent_type = agent_type, \
                                       random_seed = random_seed, \
                                       display_bool = False)
                cmg.play()
                avg_outcome = cmg.agent.compute_agent_outcome()/playable_money
                avg_regret = cmg.agent.compute_agent_regret()/playable_money
                outcomes_list.append(avg_outcome)
                regret_list.append(avg_regret)
            
            # Store average performance and regret
            n_epsilon = len(epsilon_values_list)
            strategies_performance_outcomes[agent_type] = [np_mean(outcomes_list) for i in range(n_epsilon)]
            strategies_performance_regrets[agent_type] = [np_mean(regret_list) for i in range(n_epsilon)]
            
        else:
            for epsilon_value in epsilon_values_list:
                outcomes_list = []
                regret_list = []
                for random_seed in seeds_list:
                    # Play game with said strategy and epsilon_value
                    cmg = CandyMachineGame(playable_money = playable_money, \
                                           number_of_deterministic_machines = number_of_deterministic_machines, \
                                           number_of_random_machines = number_of_random_machines, \
                                           machines_cost = machines_cost, \
                                           return_prices_for_deterministic_machines = return_prices_for_deterministic_machines,\
                                           return_prices_for_random_machines_win = return_prices_for_random_machines_win, \
                                           return_prices_for_random_machines_loss = return_prices_for_random_machines_loss, \
                                           agent_type = agent_type, \
                                           agent_parameters = epsilon_value, \
                                           random_seed = random_seed, \
                                           display_bool = False)
                    cmg.play()
                    avg_outcome = cmg.agent.compute_agent_outcome()/playable_money
                    avg_regret = cmg.agent.compute_agent_regret()/playable_money
                    outcomes_list.append(avg_outcome)
                    regret_list.append(avg_regret)

                    # Update waitbar
                    t.update(1)

                # Store average performance and regret
                strategies_performance_outcomes[agent_type].append(np_mean(outcomes_list))
                strategies_performance_regrets[agent_type].append(np_mean(regret_list))
            
    # Close waitbar
    t.close()

    # Display (epsilon_value vs. outcome)
    x_values = epsilon_values_list
    labels_outcome = []
    labels_outcome_log = []
    for agent_type in agent_types_list:
        outcome_vals = strategies_performance_outcomes[agent_type]
        max_val = max(outcome_vals)
        max_eps = epsilon_values_list[outcome_vals.index(max_val)]
        if agent_type == 'random_naive' or agent_type == 'prior_knowledge':
            str_add1 = str(agent_type) + "\n(Max at with value = {})".format(max_val)
            str_add2 = str(agent_type) + "\n(Max at with value = {})".format(max_val)
        else:
            str_add1 = str(agent_type) + "\n(Max at epsilon = {}, with value = {})".format(max_eps, max_val)
            str_add2 = str(agent_type) + "\n(Max at epsilon = {}, with value = {})".format(np_log10(max_eps), max_val)
        labels_outcome.append(str_add1)
        labels_outcome_log.append(str_add2)
        
    plt.figure(figsize = (10, 7))
    for y_values, agent_type in zip(strategies_performance_outcomes.values(), agent_types_list):
        plt.plot(x_values, y_values, label = labels_outcome[agent_types_list.index(agent_type)])
    title_str = "Average outcome performance for each strategy played by agent vs. epsilon value."
    plt.title(title_str)
    plt.xlabel('epsilon value')
    plt.ylabel('Outcome')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()

    # Display (epsilon_value vs. regret)
    labels_regret = []
    labels_regret_log = []
    for agent_type in agent_types_list:
        regret_vals = strategies_performance_regrets[agent_type]
        min_val = min(regret_vals)
        min_eps = epsilon_values_list[outcome_vals.index(max_val)]
        str_add = str(agent_type) + "\n(Min at epsilon = {}, with value = {})".format(min_eps, min_val)
        labels_regret.append(str_add)
        str_add = str(agent_type) + "\n(Min at epsilon = {}, with value = {})".format(np_log10(min_eps), min_val)
        labels_regret_log.append(str_add)
    plt.figure(figsize = (10, 7))
    for y_values, agent_type in zip(strategies_performance_regrets.values(), agent_types_list):
        plt.plot(x_values, y_values, label = labels_regret[agent_types_list.index(agent_type)])
    title_str = "Average regret performance for each strategy played by agent vs. epsilon value."
    plt.title(title_str)
    plt.xlabel('epsilon value')
    plt.ylabel('Regret')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()

    # Display (epsilon_value vs. outcome)
    x_values = epsilon_values_list_log
    plt.figure(figsize = (10, 7))
    for y_values, agent_type in zip(strategies_performance_outcomes.values(), agent_types_list):
        plt.plot(x_values, y_values, label = labels_outcome_log[agent_types_list.index(agent_type)])
    title_str = "Average outcome performance for each strategy played by agent vs. epsilon value (log10 applied)."
    plt.title(title_str)
    plt.xlabel('Log10 of epsilon value')
    plt.ylabel('Outcome')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()

    # Display (epsilon_value vs. regret)
    plt.figure(figsize = (10, 7))
    for y_values, agent_type in zip(strategies_performance_regrets.values(), agent_types_list):
        plt.plot(x_values, y_values, label = labels_regret_log[agent_types_list.index(agent_type)])
    title_str = "Average regret performance for each strategy played by agent vs. epsilon value (log10 applied)."
    plt.title(title_str)
    plt.xlabel('Log10 of epsilon value')
    plt.ylabel('Regret')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()
    
    
'''
-------------------------------------------------------------------------------------------------
FUNCTION DEFINITION: MonteCarlo Analysis for impact of epsilon in epsilon-greedy strategies
-------------------------------------------------------------------------------------------------
'''

def display_impact_epsilon_greedy(n_seeds = 100, \
                                  epsilon_values_list = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, \
                                                  5e-2, 0.1, 0.2, 0.5, 0.75, 1], \
                                  playable_money = 1000, \
                                  number_of_deterministic_machines = 1, \
                                  number_of_random_machines = 5, \
                                  machines_cost = 1, \
                                  return_prices_for_deterministic_machines = 1, \
                                  return_prices_for_random_machines_win = 2, \
                                  return_prices_for_random_machines_loss = 0):

    # List of seeds
    seeds_list = [i + 1 for i in range(n_seeds)]

    # List of epsilon_values (log10)
    epsilon_values_list_log = np_log10(epsilon_values_list)
    
    # List of agent types
    agent_types_list = ['random_naive', 'prior_knowledge', 'epsilon_greedy_naive', \
                        'epsilon_greedy_linear', 'epsilon_greedy_softmax']

    # Initialize strategies dictionaries
    strategies_performance_outcomes = {agent: [] for agent in agent_types_list}
    strategies_performance_regrets = {agent: [] for agent in agent_types_list}

    # Initialize waitbar
    desc_str = "Testing impact of epsilon value on epsilon-greedy strategies"
    n_iter = len(epsilon_values_list)*(len(agent_types_list) - 2)*n_seeds
    t = tqdm(desc = desc_str, total = n_iter)

    # Play for all strategies
    for agent_type in strategies_performance_outcomes.keys():
        if agent_type == 'random_naive' or agent_type == 'prior_knowledge':
            outcomes_list = []
            regret_list = []
            for random_seed in seeds_list:
                # Play game with said strategy and epsilon_value
                cmg = CandyMachineGame(playable_money = playable_money, \
                                       number_of_deterministic_machines = number_of_deterministic_machines, \
                                       number_of_random_machines = number_of_random_machines, \
                                       machines_cost = machines_cost, \
                                       return_prices_for_deterministic_machines = return_prices_for_deterministic_machines, \
                                       return_prices_for_random_machines_win = return_prices_for_random_machines_win, \
                                       return_prices_for_random_machines_loss = return_prices_for_random_machines_loss, \
                                       agent_type = agent_type, \
                                       random_seed = random_seed, \
                                       display_bool = False)
                cmg.play()
                avg_outcome = cmg.agent.compute_agent_outcome()/playable_money
                avg_regret = cmg.agent.compute_agent_regret()/playable_money
                outcomes_list.append(avg_outcome)
                regret_list.append(avg_regret)
            
            # Store average performance and regret
            n_epsilon = len(epsilon_values_list)
            strategies_performance_outcomes[agent_type] = [np_mean(outcomes_list) for i in range(n_epsilon)]
            strategies_performance_regrets[agent_type] = [np_mean(regret_list) for i in range(n_epsilon)]
            
        else:
            for epsilon_value in epsilon_values_list:
                outcomes_list = []
                regret_list = []
                for random_seed in seeds_list:
                    # Play game with said strategy and epsilon_value
                    cmg = CandyMachineGame(playable_money = playable_money, \
                                           number_of_deterministic_machines = number_of_deterministic_machines, \
                                           number_of_random_machines = number_of_random_machines, \
                                           machines_cost = machines_cost, \
                                           return_prices_for_deterministic_machines = return_prices_for_deterministic_machines,\
                                           return_prices_for_random_machines_win = return_prices_for_random_machines_win, \
                                           return_prices_for_random_machines_loss = return_prices_for_random_machines_loss, \
                                           agent_type = agent_type, \
                                           agent_parameters = epsilon_value, \
                                           random_seed = random_seed, \
                                           display_bool = False)
                    cmg.play()
                    avg_outcome = cmg.agent.compute_agent_outcome()/playable_money
                    avg_regret = cmg.agent.compute_agent_regret()/playable_money
                    outcomes_list.append(avg_outcome)
                    regret_list.append(avg_regret)

                    # Update waitbar
                    t.update(1)

                # Store average performance and regret
                strategies_performance_outcomes[agent_type].append(np_mean(outcomes_list))
                strategies_performance_regrets[agent_type].append(np_mean(regret_list))
            
    # Close waitbar
    t.close()

    # Display (epsilon_value vs. outcome)
    x_values = epsilon_values_list
    labels_outcome = []
    labels_outcome_log = []
    for agent_type in agent_types_list:
        outcome_vals = strategies_performance_outcomes[agent_type]
        max_val = max(outcome_vals)
        max_eps = epsilon_values_list[outcome_vals.index(max_val)]
        if agent_type == 'random_naive' or agent_type == 'prior_knowledge':
            str_add1 = str(agent_type) + "\n(Max at with value = {})".format(max_val)
            str_add2 = str(agent_type) + "\n(Max at with value = {})".format(max_val)
        else:
            str_add1 = str(agent_type) + "\n(Max at epsilon = {}, with value = {})".format(max_eps, max_val)
            str_add2 = str(agent_type) + "\n(Max at epsilon = {}, with value = {})".format(np_log10(max_eps), max_val)
        labels_outcome.append(str_add1)
        labels_outcome_log.append(str_add2)
        
    plt.figure(figsize = (10, 7))
    for y_values, agent_type in zip(strategies_performance_outcomes.values(), agent_types_list):
        plt.plot(x_values, y_values, label = labels_outcome[agent_types_list.index(agent_type)])
    title_str = "Average outcome performance for each strategy played by agent vs. epsilon value."
    plt.title(title_str)
    plt.xlabel('epsilon value')
    plt.ylabel('Outcome')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()

    # Display (epsilon_value vs. regret)
    labels_regret = []
    labels_regret_log = []
    for agent_type in agent_types_list:
        regret_vals = strategies_performance_regrets[agent_type]
        min_val = min(regret_vals)
        min_eps = epsilon_values_list[outcome_vals.index(max_val)]
        str_add = str(agent_type) + "\n(Min at epsilon = {}, with value = {})".format(min_eps, min_val)
        labels_regret.append(str_add)
        str_add = str(agent_type) + "\n(Min at epsilon = {}, with value = {})".format(np_log10(min_eps), min_val)
        labels_regret_log.append(str_add)
    plt.figure(figsize = (10, 7))
    for y_values, agent_type in zip(strategies_performance_regrets.values(), agent_types_list):
        plt.plot(x_values, y_values, label = labels_regret[agent_types_list.index(agent_type)])
    title_str = "Average regret performance for each strategy played by agent vs. epsilon value."
    plt.title(title_str)
    plt.xlabel('epsilon value')
    plt.ylabel('Regret')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()

    # Display (epsilon_value vs. outcome)
    x_values = epsilon_values_list_log
    plt.figure(figsize = (10, 7))
    for y_values, agent_type in zip(strategies_performance_outcomes.values(), agent_types_list):
        plt.plot(x_values, y_values, label = labels_outcome_log[agent_types_list.index(agent_type)])
    title_str = "Average outcome performance for each strategy played by agent vs. epsilon value (log10 applied)."
    plt.title(title_str)
    plt.xlabel('Log10 of epsilon value')
    plt.ylabel('Outcome')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()

    # Display (epsilon_value vs. regret)
    plt.figure(figsize = (10, 7))
    for y_values, agent_type in zip(strategies_performance_regrets.values(), agent_types_list):
        plt.plot(x_values, y_values, label = labels_regret_log[agent_types_list.index(agent_type)])
    title_str = "Average regret performance for each strategy played by agent vs. epsilon value (log10 applied)."
    plt.title(title_str)
    plt.xlabel('Log10 of epsilon value')
    plt.ylabel('Regret')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.show()
    
    
'''
-------------------------------------------------------------------------------------------------
FUNCTION DEFINITION: MonteCarlo Analysis for impact of epsilon in epsilon-greedy strategies
-------------------------------------------------------------------------------------------------
'''

def display_impact_epsilon_firstgreedy(n_seeds = 100, \
                                       epsilon_values_list = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, \
                                                              5e-2, 0.1, 0.2, 0.5, 0.75, 1], \
                                       playable_money = 1000, \
                                       number_of_deterministic_machines = 1, \
                                       number_of_random_machines = 5, \
                                       machines_cost = 1, \
                                       return_prices_for_deterministic_machines = 1, \
                                       return_prices_for_random_machines_win = 2, \
                                       return_prices_for_random_machines_loss = 0):

    # List of seeds
    seeds_list = [i + 1 for i in range(n_seeds)]

    # List of epsilon_values (log10)
    N = len(epsilon_values_list)
    epsilon_values_list1 = epsilon_values_list
    epsilon_values_list2 = epsilon_values_list
    epsilon_values_list1_log = np_log10(epsilon_values_list1)
    epsilon_values_list2_log = np_log10(epsilon_values_list2)
    
    # List of agent types
    agent_types_list = ['random_naive', 'prior_knowledge', 'epsilon_firstgreedy_naive', \
                        'epsilon_firstgreedy_linear', 'epsilon_firstgreedy_softmax']

    # Initialize strategies dictionaries
    strategies_performance_outcomes = {agent: np_zeros((N, N)) for agent in agent_types_list}
    strategies_performance_regrets = {agent: np_zeros((N, N)) for agent in agent_types_list}

    # Initialize waitbar
    desc_str = "Testing impact of epsilon value on epsilon-firstgreedy strategies"
    n_iter = N*N*(len(agent_types_list) - 2)*n_seeds
    t = tqdm(desc = desc_str, total = n_iter)

    # Play for all strategies
    for agent_type in strategies_performance_outcomes.keys():
        
        if agent_type == 'random_naive' or agent_type == 'prior_knowledge':
            outcomes_list = []
            regret_list = []
            for random_seed in seeds_list:
                # Play game with said strategy and epsilon_value
                cmg = CandyMachineGame(playable_money = playable_money, \
                                       number_of_deterministic_machines = number_of_deterministic_machines, \
                                       number_of_random_machines = number_of_random_machines, \
                                       machines_cost = machines_cost, \
                                       return_prices_for_deterministic_machines = return_prices_for_deterministic_machines, \
                                       return_prices_for_random_machines_win = return_prices_for_random_machines_win, \
                                       return_prices_for_random_machines_loss = return_prices_for_random_machines_loss, \
                                       agent_type = agent_type, \
                                       random_seed = random_seed, \
                                       display_bool = False)
                cmg.play()
                avg_outcome = cmg.agent.compute_agent_outcome()/playable_money
                avg_regret = cmg.agent.compute_agent_regret()/playable_money
                outcomes_list.append(avg_outcome)
                regret_list.append(avg_regret)
            
            # Store average performance and regret
            strategies_performance_outcomes[agent_type] = np_mean(outcomes_list)*np_ones((N, N))
            strategies_performance_regrets[agent_type] = np_mean(regret_list)*np_ones((N, N))
        
        else:
            for epsilon_index1, epsilon_value1 in enumerate(epsilon_values_list1):
                for epsilon_index2, epsilon_value2 in enumerate(epsilon_values_list2):
                    outcomes_list = []
                    regret_list = []
                    for random_seed in seeds_list:
                        # Define epsilon_values
                        epsilon_values = [epsilon_value1, epsilon_value2]
                        # Play game with said strategy and epsilon_value
                        val = return_prices_for_deterministic_machines
                        cmg = CandyMachineGame(playable_money = playable_money, \
                                               number_of_deterministic_machines = number_of_deterministic_machines, \
                                               number_of_random_machines = number_of_random_machines, \
                                               machines_cost = machines_cost, \
                                               return_prices_for_deterministic_machines = val, \
                                               return_prices_for_random_machines_win = return_prices_for_random_machines_win, \
                                               return_prices_for_random_machines_loss =return_prices_for_random_machines_loss,\
                                               agent_type = agent_type, \
                                               agent_parameters = epsilon_values, \
                                               random_seed = random_seed, \
                                               display_bool = False)
                        cmg.play()
                        avg_outcome = cmg.agent.compute_agent_outcome()/playable_money
                        avg_regret = cmg.agent.compute_agent_regret()/playable_money
                        outcomes_list.append(avg_outcome)
                        regret_list.append(avg_regret)

                        # Update waitbar
                        t.update(1)

                    # Store average performance and regret
                    strategies_performance_outcomes[agent_type][epsilon_index1, epsilon_index2] = (np_mean(outcomes_list))
                    strategies_performance_regrets[agent_type][epsilon_index1, epsilon_index2] = (np_mean(regret_list))
            
    # Close waitbar
    t.close()
    
    # Surfplot: epsilon_firstgreedy_naive
    agent_name = 'epsilon_firstgreedy_naive'
    fig = plt.figure(figsize = (10, 7))
    ax = fig.gca(projection = '3d')
    ax.view_init()
    X, Y = np_mesh(epsilon_values_list, epsilon_values_list)
    Z = strategies_performance_outcomes[agent_name]
    surf = ax.plot_surface(X, Y, Z, \
                           cmap = cm.coolwarm,
                           linewidth = 0, \
                           antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    ax.set_xlabel('$\epsilon_1$', fontsize = 20)
    ax.set_ylabel('$\epsilon_2$', fontsize = 20)
    max_val = np_max(Z)
    indexes = np_urvl(np_amax(Z), Z.shape)
    eps1 = epsilon_values_list[indexes[0]]
    eps2 = epsilon_values_list[indexes[1]]
    title_str1 = "Average outcome performance vs. epsilons values, for {} agent".format(agent_name)
    title_str2 = "\nMax at (epsilon_1 = {}, epsilon_2 = {}) with value {}".format(eps1, eps2, max_val)
    title_str3 = "\nUpper bound performance = {}".format(np_max(strategies_performance_outcomes['prior_knowledge']))
    ax.set_title(title_str1 + title_str2 + title_str3)
    plt.show()
    
    # Surfplot: epsilon_firstgreedy_linear
    agent_name = 'epsilon_firstgreedy_linear'
    fig = plt.figure(figsize = (10, 7))
    ax = fig.gca(projection = '3d')
    ax.view_init()
    X, Y = np_mesh(epsilon_values_list, epsilon_values_list)
    Z = strategies_performance_outcomes[agent_name]
    surf = ax.plot_surface(X, Y, Z, \
                           cmap = cm.coolwarm,
                           linewidth = 0, \
                           antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    ax.set_xlabel('$\epsilon_1$', fontsize = 20)
    ax.set_ylabel('$\epsilon_2$', fontsize = 20)
    max_val = np_max(Z)
    indexes = np_urvl(np_amax(Z), Z.shape)
    eps1 = epsilon_values_list[indexes[0]]
    eps2 = epsilon_values_list[indexes[1]]
    title_str1 = "Average outcome performance vs. epsilons values, for {} agent".format(agent_name)
    title_str2 = "\nMax at (epsilon_1 = {}, epsilon_2 = {}) with value {}".format(eps1, eps2, max_val)
    title_str3 = "\nUpper bound performance = {}".format(np_max(strategies_performance_outcomes['prior_knowledge']))
    ax.set_title(title_str1 + title_str2 + title_str3)
    plt.show()
    
    # Surfplot: epsilon_firstgreedy_softmax
    agent_name = 'epsilon_firstgreedy_softmax'
    fig = plt.figure(figsize = (10, 7))
    ax = fig.gca(projection = '3d')
    ax.view_init()
    X, Y = np_mesh(epsilon_values_list, epsilon_values_list)
    Z = strategies_performance_outcomes[agent_name]
    surf = ax.plot_surface(X, Y, Z, \
                           cmap = cm.coolwarm,
                           linewidth = 0, \
                           antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    ax.set_xlabel('$\epsilon_1$', fontsize = 20)
    ax.set_ylabel('$\epsilon_2$', fontsize = 20)
    max_val = np_max(Z)
    indexes = np_urvl(np_amax(Z), Z.shape)
    eps1 = epsilon_values_list[indexes[0]]
    eps2 = epsilon_values_list[indexes[1]]
    title_str1 = "Average outcome performance vs. epsilons values, for {} agent".format(agent_name)
    title_str2 = "\nMax at (epsilon_1 = {}, epsilon_2 = {}) with value {}".format(eps1, eps2, max_val)
    title_str3 = "\nUpper bound performance = {}".format(np_max(strategies_performance_outcomes['prior_knowledge']))
    ax.set_title(title_str1 + title_str2 + title_str3)
    plt.show()
    
    # Surfplot: epsilon_firstgreedy_naive
    agent_name = 'epsilon_firstgreedy_naive'
    fig = plt.figure(figsize = (10, 7))
    ax = fig.gca(projection = '3d')
    ax.view_init()
    X, Y = np_mesh(epsilon_values_list, epsilon_values_list)
    Z = strategies_performance_regrets[agent_name]
    surf = ax.plot_surface(X, Y, Z, \
                           cmap = cm.coolwarm,
                           linewidth = 0, \
                           antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    ax.set_xlabel('$\epsilon_1$', fontsize = 20)
    ax.set_ylabel('$\epsilon_2$', fontsize = 20)
    max_val = np_min(Z)
    indexes = np_urvl(np_amin(Z), Z.shape)
    eps1 = epsilon_values_list[indexes[0]]
    eps2 = epsilon_values_list[indexes[1]]
    title_str1 = "Average regret performance vs. epsilons values, for {} agent".format(agent_name)
    title_str2 = "\nMax at (epsilon_1 = {}, epsilon_2 = {}) with value {}".format(eps1, eps2, max_val)
    title_str3 = "\nUpper bound performance (regret) = {}".format(np_max(strategies_performance_regrets['prior_knowledge']))
    ax.set_title(title_str1 + title_str2 + title_str3)
    plt.show()
    
    # Surfplot: epsilon_firstgreedy_linear
    agent_name = 'epsilon_firstgreedy_linear'
    fig = plt.figure(figsize = (10, 7))
    ax = fig.gca(projection = '3d')
    ax.view_init()
    X, Y = np_mesh(epsilon_values_list, epsilon_values_list)
    Z = strategies_performance_regrets[agent_name]
    surf = ax.plot_surface(X, Y, Z, \
                           cmap = cm.coolwarm,
                           linewidth = 0, \
                           antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    ax.set_xlabel('$\epsilon_1$', fontsize = 20)
    ax.set_ylabel('$\epsilon_2$', fontsize = 20)
    max_val = np_min(Z)
    indexes = np_urvl(np_amin(Z), Z.shape)
    eps1 = epsilon_values_list[indexes[0]]
    eps2 = epsilon_values_list[indexes[1]]
    title_str1 = "Average regret performance vs. epsilons values, for {} agent".format(agent_name)
    title_str2 = "\nMax at (epsilon_1 = {}, epsilon_2 = {}) with value {}".format(eps1, eps2, max_val)
    title_str3 = "\nUpper bound performance (regret) = {}".format(np_max(strategies_performance_regrets['prior_knowledge']))
    ax.set_title(title_str1 + title_str2 + title_str3)
    plt.show()
    
    # Surfplot: epsilon_firstgreedy_softmax
    agent_name = 'epsilon_firstgreedy_softmax'
    fig = plt.figure(figsize = (10, 7))
    ax = fig.gca(projection = '3d')
    ax.view_init()
    X, Y = np_mesh(epsilon_values_list, epsilon_values_list)
    Z = strategies_performance_regrets[agent_name]
    surf = ax.plot_surface(X, Y, Z, \
                           cmap = cm.coolwarm,
                           linewidth = 0, \
                           antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    ax.set_xlabel('$\epsilon_1$', fontsize = 20)
    ax.set_ylabel('$\epsilon_2$', fontsize = 20)
    max_val = np_min(Z)
    indexes = np_urvl(np_amin(Z), Z.shape)
    eps1 = epsilon_values_list[indexes[0]]
    eps2 = epsilon_values_list[indexes[1]]
    title_str1 = "Average regret performance vs. epsilons values, for {} agent".format(agent_name)
    title_str2 = "\nMax at (epsilon_1 = {}, epsilon_2 = {}) with value {}".format(eps1, eps2, max_val)
    title_str3 = "\nUpper bound performance (regret) = {}".format(np_max(strategies_performance_regrets['prior_knowledge']))
    ax.set_title(title_str1 + title_str2 + title_str3)
    plt.show()
    