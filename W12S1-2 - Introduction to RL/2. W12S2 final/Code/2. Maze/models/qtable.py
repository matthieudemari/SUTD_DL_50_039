import random
import numpy as np
from datetime import datetime
from environment.maze import Status
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """
    Abstract base class for prediction models.
    """
    
    def __init__(self, maze, **kwargs):
        self.environment = maze
        self.name = kwargs.get("name", "model")

    def load(self, filename):
        """ Load model from file. """
        pass

    def save(self, filename):
        """ Save model to file. """
        pass

    def train(self, stop_at_convergence=False, **kwargs):
        """ Train model. """
        pass

    @abstractmethod
    def q(self, state):
        """ Return q values for state. """
        pass

    @abstractmethod
    def predict(self, state):
        """ Predict value based on state. """
        pass

    

class QTableModel(AbstractModel):
    """
    Tabular Q-learning prediction model.
    For every state (here: the agents current location ) the value for each of the actions is stored in a table.
    The key for this table is (state + action). Initially all values are 0. When playing training games
    after every move the value in the table is updated based on the reward gained after making the move. Training
    ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).
    """
    
    default_check_convergence_every = 5  # by default check for convergence every # episodes

    def __init__(self, game, **kwargs):
        """ Create a new prediction model for 'game'.
        - param class Maze game: Maze game object
        - param kwargs: model dependent init parameters
        """
        super().__init__(game, **kwargs)
        self.Q = dict()  # table with value for (state, action) combination

    def train(self, stop_at_convergence = False, **kwargs):
        """
        Train the model.
        - param stop_at_convergence: stop training as soon as convergence is reached

        Hyperparameters:
        - discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
        - exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
        - exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
        - learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
        - episodes: number of training games to play
        Returns number of training episodes, total time spent
        """
        
        # Extract parameters from kwargs
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = max(kwargs.get("episodes", 1000), 1)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        # Variables for history
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []
        start_list = list()
        start_time = datetime.now()

        # Training
        for episode in range(1, episodes + 1):
            # Display
            print("Episode: {}".format(episode))
                  
            # Optimization: make sure to start from all possible cells
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)
            state = self.environment.reset(start_cell)
            # Change np.ndarray to tuple so it can be used as dictionary key
            state = tuple(state.flatten())  

            while True:
                # Choose action following epsilon greedy
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = self.predict(state)
                
                # Produce next state, reward and cumulative reward
                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())
                cumulative_reward += reward
                
                # Check for entry in Q table, if does not exist, add it.
                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0
                    
                # Update rule for Q learning
                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])
                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                # If have reached final cell of max number of iterations, stop.
                if status in (Status.WIN, Status.LOSE):
                    break
                state = next_state
                self.environment.render_q(self)
                
            # Append to reward history
            cumulative_reward_history.append(cumulative_reward)
            
            # Check for convergence on Q table
            if episode % check_convergence_every == 0:
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    print("Won from all start cells. Stopped learning")
                    break
                    
            # Decay on exploration rate for epsilon greedy
            exploration_rate *= exploration_decay

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    
    def q(self, state):
        """
        Getter for q values for all actions for a certain state.
        """
        
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    
    def predict(self, state):
        """
        Policy: choose the action with the highest value from the Q-table.
        Random choice if multiple actions have the same (max) value.
        Parameters:
        - state: game state
        Returns selected action
        """
        
        q = self.q(state)
        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)
