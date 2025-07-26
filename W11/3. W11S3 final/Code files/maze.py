from enum import Enum, IntEnum
import numpy as np
import matplotlib.pyplot as plt

class Cell(IntEnum):
    # Using an Enum for simplicity
    # 0 = Empty cell where the agent can move
    # 1 = Cell containing a wall, not accessible
    EMPTY = 0
    OCCUPIED = 1

class Action(IntEnum):
    # Using an Enum for simplicity
    # Self explanatory
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3

class Status(Enum):
    # Using an Enum for simplicity
    # Self explanatory
    WIN = 0
    LOSE = 1
    PLAYING = 2

class Maze:
    """
    A Maze environment where an agent must navigate from a start cell to an exit cell.
    """
    def __init__(self, maze, start_cell, exit_cell):
        # Initialize maze
        self.create_maze(maze, start_cell, exit_cell)
        # Define the RL parameters for agent (rewards, etc.)
        self.define_rl_params()
        # Reset
        self.reset(start_cell)

    def create_maze(self, maze, start_cell, exit_cell):
        # Maze definition
        self.maze = maze
        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        self.exit_cell = exit_cell
        self.empty.remove(self.exit_cell)
        self.start_cell = start_cell
        
    def define_rl_params(self):
        # List of all 4 possible actions
        self.actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]
        # Reward for reaching the exit
        self.reward_exit = 10.0
        # Penalty applied everytime a move is taken (encourages the agent to find the exit ASAP)
        self.penalty_move = -0.05
        # Penalty for revisiting a cell (encourages the agent not to revisit cells it has already gone through)
        self.penalty_visited = -0.25
        # Penalty for invalid moves (encourages agent not to bump into walls)
        self.penalty_impossible_move = -0.75
        # Threshold to force game end, if cumulated reward falls below this value
        self.minimum_reward = -0.5*self.maze.size
        
    def reset(self, start_cell):
        """
        Reset the maze to its initial state.
        """
        self.previous_cell = self.current_cell = start_cell
        self.total_reward = 0.0
        self.visited = set()
        return self.observe()

    def step(self, action):
        """
        Perform an action in the maze.
        """
        reward = self.execute(action)
        self.total_reward += reward
        status = self.status()
        state = self.observe()
        return state, reward, status

    def execute(self, action):
        """
        Execute an action and return the reward of this action.
        """
        # Compute possible actions in the current location
        possible_actions = self.possible_actions(self.current_cell)
        if not possible_actions:
            # No valid moves, force game over
            return self.minimum_reward - 1
        # Tried going into a wall, apply penalty and do nothing in terms of movement
        if action not in possible_actions:
            return self.penalty_impossible_move
        # Otherwise, move the agent based on the selected action
        col, row = self.current_cell
        if action == Action.MOVE_LEFT:
            col -= 1
        elif action == Action.MOVE_UP:
            row -= 1
        elif action == Action.MOVE_RIGHT:
            col += 1
        elif action == Action.MOVE_DOWN:
            row += 1
        # Update cell position
        self.previous_cell = self.current_cell
        self.current_cell = (col, row)
        # Check for penalties or rewards based on the new cell
        if self.current_cell == self.exit_cell:
            return self.reward_exit
        elif self.current_cell in self.visited:
            # Apply penalty for revisiting
            return self.penalty_visited
        else:
            # Mark as visited
            self.visited.add(self.current_cell)
            return self.penalty_move

    def possible_actions(self, cell = None):
        """
        Determine all valid actions from the current cell.
        """
        if cell is None:
            col, row = self.current_cell
        else:
            col, row = cell
        possible_actions = self.actions.copy()
        nrows, ncols = self.maze.shape
        # Update list of moves
        # For each direction, remove from list of actions if it leads into a wall or out of the maze
        if row == 0 or self.maze[row - 1, col] == Cell.OCCUPIED:
            possible_actions.remove(Action.MOVE_UP)
        if row == nrows - 1 or self.maze[row + 1, col] == Cell.OCCUPIED:
            possible_actions.remove(Action.MOVE_DOWN)
        if col == 0 or self.maze[row, col - 1] == Cell.OCCUPIED:
            possible_actions.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or self.maze[row, col + 1] == Cell.OCCUPIED:
            possible_actions.remove(Action.MOVE_RIGHT)
        return possible_actions

    def status(self):
        """
        Determine the current status of the game.
        """
        if self.current_cell == self.exit_cell:
            # Reached the exit
            return Status.WIN
        elif self.total_reward < self.minimum_reward:
            # Cumulative reward going below minimum_reward means
            # too many moves taken without finding the exit
            return Status.LOSE
        else:
            # Otherwise, still playing
            return Status.PLAYING

    def observe(self):
        """
        Helper function that returns the current state of the agent.
        """
        return np.array([[*self.current_cell]])

    def play(self, model, start_cell = (0, 0)):
        """
        Play a game using the provided model to predict actions.
        """
        self.reset(start_cell)
        state = self.observe()
        while True:
            # Decide on action to use in state
            action = model.predict(state)
            # Update environment by applying action
            state, reward, status = self.step(action)
            # When game ends (exit reach or too many moves), return outcome (win or lose)
            if status in (Status.WIN, Status.LOSE):
                return status

    def check_win_all(self, model):
        """
        Check if the model wins from all starting cells.
        """
        win = 0
        lose = 0
        for cell in self.empty:
            if self.play(model, cell) == Status.WIN:
                win += 1
            else:
                lose += 1
        return lose == 0, win/(win + lose)
    
    def draw_full_maze(self):
        """
        Draw the entire maze with walls, valid cells, start, and exit positions.
        Ensures (0,0) is at the top-left and (nrows-1, ncols-1) at the bottom-right.
        """
        nrows, ncols = self.maze.shape
        fig, ax = plt.subplots(figsize = (6, 6), tight_layout = True)
        # Display the maze
        ax.imshow(self.maze, cmap = "Greys", origin = "upper")
        # Set major ticks to match correct grid positions
        ax.set_xticks(np.arange(ncols))
        ax.set_yticks(np.arange(nrows))
        # Label axes correctly with 0 at top-left
        ax.set_xticklabels(np.arange(ncols))
        ax.set_yticklabels(np.arange(nrows))
        # Draw grid lines
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor = True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor = True)
        ax.grid(which = "minor", color = "black", linestyle = "-", linewidth = 0.5)
        # Mark the starting position
        start_col, start_row = self.start_cell
        ax.plot(start_col, start_row, "rs", markersize = 20)
        ax.text(start_col, start_row, "Start", ha = "center", va = "center", color = "white")
        # Mark the exit position
        exit_col, exit_row = self.exit_cell
        ax.plot(exit_col, exit_row, "gs", markersize = 20)
        ax.text(exit_col, exit_row, "Exit", ha = "center", va = "center", color = "white")
        plt.show()

    def visualize_best_moves(self, model):
        """
        Plot the best move (policy) for each non-wall cell in the maze after training.
        """
        # Initialize plot
        nrows, ncols = self.maze.shape
        fig, ax = plt.subplots(figsize = (6, 6))
        ax.imshow(self.maze, cmap = "Greys", origin = "upper")
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor = True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor = True)
        ax.grid(which = "minor", color = "black", linestyle = "-", linewidth = 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("Best Move in Each Cell")
        
        # Action mapping to arrows
        action_arrows = {0: "←", 1: "→", 2: "↑",  3: "↓"}
    
        # Plot best moves according to model in each walkable cell
        for row in range(nrows):
            for col in range(ncols):
                # Skip walls and exit cell
                if self.maze[row, col] == 0 and not (row, col) == self.exit_cell:
                    state = np.array([[col, row]])
                    best_action = model.predict(state)
                    ax.text(col, row, action_arrows[best_action], ha = 'center', va = 'center', fontsize = 12, color = 'red')
                    
        # Mark the exit and display
        ax.text(*self.exit_cell, "Exit", ha = "center", va = "center", color = "green")
        plt.show()