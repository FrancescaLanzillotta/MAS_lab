import random
import time
from enum import Enum
from pprint import pprint

import matplotlib.pyplot
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import animation


class Cell(Enum):
    AGENT = 0.001,
    ENEMY = -1,
    FREE = -0.04,
    OBSTACLE = 0,
    GOAL = 1,


class Direction(Enum):
    UP = (-1, 0),
    DOWN = (1, 0),
    LEFT = (0, -1),
    RIGHT = (0, 1),
    STAY = (0, 0),


def get_opposite(direction):
    if direction == Direction.UP:
        return direction.DOWN
    if direction == Direction.DOWN:
        return direction.UP
    if direction == Direction.LEFT:
        return Direction.RIGHT
    if direction == Direction.RIGHT:
        return Direction.LEFT


def get_next_coord(direction, x, y):
    """
    Returns the coordinates of the adjacent cell in specified direction.

    :param direction: direction of the next cell
    :param x: row coordinate of starting cell
    :param y: column coordinate of starting cell
    :return: coordinates of the adjacent cell
    """

    off_x, off_y = direction.value[0]
    return x + off_x, y + off_y


def plot_static_grid(rows, columns, fig=None, ax=None):
    """
    Helper method used to plot a static grid of specified dimensions.

    :param rows: number of rows in the desired grid
    :param columns: number of columns in the desired grid
    :param fig: Figure matplotlib object. If none passed, new one is created
    :param ax: Axes matplotlib object. If none passed, new one is created
    :return: Figure and Axes matplotlib objects for the plot
    """

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    extent = (0, columns, rows - 1, -1)
    static_grid = np.zeros((rows, columns))

    ax.matshow(static_grid, alpha=0.1, cmap='gray', extent=extent)

    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    ax.grid(linewidth=1, color='black')

    return fig, ax


def annotate_grid_plot(artist, type, x, y, c=None):
    """
    Helper function to add consistent textual annotations to grid. If type is not recognized, no annotation is made.

    :param artist: artist used in matplotlib for visible elements
    :param type: type of annotations to make
    :param x: row coordinate
    :param y: column coordinate
    :return: updated artist after annotations
    """

    if type == 'COORDS':
        artist = artist.text(y, x, f'({x}, {y})')
    else:
        s = ''
        col = None
        size = 0
        x_offset = 0
        if type == 'OBSTACLE':
            s = '✘'
            col = '#3c3c3c'
            size = 32
            x_offset = 0.4
        elif type == 'AGENT':
            s = '●'
            col = 'forestgreen'
            size = 24
            x_offset = 0.5
        elif type == 'ENEMY':
            s = '\u2B23'
            col = 'crimson'
            size = 24
            x_offset = 0.45
        elif type == 'GOAL':
            s = '\u2691'
            col = 'navy'
            size = 28
            x_offset = 0.40
        elif type == 'UP':
            s = '\u2191'
            col = 'black'
            size = 18
            x_offset = 0.45
        elif type == 'DOWN':
            s = '\u2193'
            col = 'black'
            size = 18
            x_offset = 0.45
        elif type == 'RIGHT':
            s = '\u2192'
            col = 'black'
            size = 18
            x_offset = 0.45
        elif type == 'LEFT':
            s = '\u2190'
            col = 'black'
            size = 18
            x_offset = 0.45
        elif isinstance(type, float):
            s = type
            col = 'black' if c is None else c
            size = 9
            x_offset = 0.5

        # rows are on the vertical axis, columns on the horizontal axis
        artist = artist.text(y + 0.5, x - x_offset, s, ha='center', va='center', color=col, fontsize=size)

    return artist


def transition_probabilities(direction, possible_moves, main_prob=0.8):
    """
    Compute transition probabilities as follows: assign the value main_prob to the specified direction and split the
    remainder (1 - main_prob) between the other possible moves, excluding the direction opposite to the main one.

    :param direction: main direction
    :param possible_moves: list of all possible moves, including main direction
    :param main_prob: probability to assign to main direction.
    :return: dictionary with directions as keys and transition probabilities as values
    """

    probs = {}
    opp = get_opposite(direction)
    poss = possible_moves.copy()
    if opp in poss:
        poss.remove(opp)

    for d in poss:
        if d == direction:
            probs[d] = main_prob
        else:
            if len(poss) > 1:
                probs[d] = round((1 - main_prob) / (len(poss) - 1), 3)
            else:
                probs[d] = round(1 - main_prob, 3)

    return probs


class Grid:
    """
    Class representing grid world were the objects reside. The grid can contain obstacles, a single goal cell,
    an arbitrary number of enemies and an arbitrary number of agents. The objects that can move within the grid are
    tracked through class attributes.
    """

    def __init__(self, rows, columns):
        """
        Constructor for Grid class. Builds a grid of specified dimension containing only FREE cell type. Moreover, it
        initializes the goal attribute to None and two empty lists, one for the ENEMY cells and one for the AGENT cells.

        :param rows: number of rows in the desired grid
        :param columns: number of columns in the desired grid
        """

        self.rows = rows
        self.columns = columns
        self.goal = None
        self.agents = []
        self.enemies = []
        self.grid = np.full((rows, columns), Cell.FREE)

    def are_valid(self, x, y):
        """
        Whether the specified coordinates are valid for the current grid.

        :param x: row coordinate
        :param y: column coordinate
        :return: True if the coordinates are valid, False otherwise
        """

        return (-1 < x < self.rows) and (-1 < y < self.columns)

    def is_reachable(self, x, y):
        """
        Whether the specified cell can be moved to i.e. the coordinates are valid and there is no obstacle.
        Notice that this implies that the presence of the goal and the enemies do not hinder the possibility to move to
        that cell.

        :param x: row coordinate
        :param y: column coordinate
        :return: True if cell is reachable, False otherwise
        """

        return self.are_valid(x, y) and self.get_cell(x, y) != Cell.OBSTACLE

    def get_cell(self, x, y):
        """
        Returns the cell at desired position. If the coordinates aren't valid, raises an error.

        :param x: row coordinate
        :param y: column coordinate
        :return: cell at the specified coordinates
        """

        if self.are_valid(x, y):
            return self.grid[x][y]
        else:
            raise TypeError('Invalid coordinates')

    def set_cell(self, x, y, cell):
        """
        Sets cell at specified coordinates. Raises error if the coordinates aren't valid or the cell parameter is not a
        Cell.

        :param x: row coordinate
        :param y: column coordinate
        :param cell: Cell to set at coordinates
        :return:
        """

        valid = self.are_valid(x, y)
        if valid and isinstance(cell, Cell):
            self.grid[x][y] = cell
        else:
            if valid:
                raise TypeError('Not a valid cell object')
            else:
                raise TypeError('Invalid coordinates')

    def add_obj(self, obj):
        """
        Adds MovingObject to the grid in the object's coordinates if the cell is free. The object is added to the specific
        tracking attribute of the Grid class depending on its type.

        :param obj: MovingObject to add to grid
        :return:
        """

        if isinstance(obj, MovingObject) and self.get_cell(obj.x, obj.y) == Cell.FREE:

            if obj.type == Cell.GOAL:
                self.goal = obj
            elif obj.type == Cell.AGENT:
                self.agents.append(obj)
            elif obj.type == Cell.ENEMY:
                self.enemies.append(obj)

            self.set_cell(obj.x, obj.y, obj.type)

        else:
            raise TypeError('Something went wrong adding object to the grid')

    def valid_directions(self, x, y):
        """
        Returns all valid directions from specified cell.

        :param x: row coordinate
        :param y: column coordinate
        :return: list of valid directions
        """

        return [d for d in Direction if d != Direction.STAY and
                self.is_reachable(*get_next_coord(d, x, y))]

    def show_grid(self, interval=500, repeat=True):
        """
        Shows an animated plot of the grid. If the objects in the grid move, the animation will show the complete
        movements of all the objects. This function relies on the fact that all objects have paths of the same length,
        make sure to pad the objects' paths accordingly if this isn't true.

        :param interval: delay between frames in milliseconds
        :param repeat: whether the animation repeats when the sequence of frames is completed
        :return: Animation variable
        """

        fig, ax = plot_static_grid(self.rows, self.columns)

        enemies_arts = []
        agents_arts = []

        for i in range(self.rows):
            for j in range(self.columns):
                # annotate_grid_plot(ax, 'COORDS', i, j)
                if self.grid[i][j] == Cell.OBSTACLE:
                    annotate_grid_plot(ax, Cell.OBSTACLE.name, i, j)

        for en in self.enemies:
            x, y = en.path[0]
            art = annotate_grid_plot(ax, 'ENEMY', x, y)
            enemies_arts.append(art)

        for ag in self.agents:
            x, y = ag.path[0]
            art = annotate_grid_plot(ax, 'AGENT', x, y)
            agents_arts.append(art)

        goal_path = self.goal.path
        x, y = goal_path[0]
        goal_art = annotate_grid_plot(ax, 'GOAL', x, y)

        def update(frame):
            x_goal, y_goal = goal_path[frame]

            for en_idx in range(len(self.enemies)):
                x, y = self.enemies[en_idx].path[frame]
                enemies_arts[en_idx].set_x(y + 0.5)
                enemies_arts[en_idx].set_y(x - 0.5)

            on_goal = False
            for ag_idx in range(len(self.agents)):
                x, y = self.agents[ag_idx].path[frame]
                agents_arts[ag_idx].set_x(y + 0.5)
                agents_arts[ag_idx].set_y(x - 0.5)
                if x == x_goal and y == y_goal:
                    on_goal = True

            if not on_goal:
                goal_art.set_x(y_goal + 0.5)
                goal_art.set_y(x_goal - 0.4)
            else:
                goal_art.set_x(-5)
                goal_art.set_y(-5)

            return goal_art, *enemies_arts, *agents_arts

        global anim     # needs to be global to avoid garbage collection
        anim = animation.FuncAnimation(fig=fig, func=update, frames=len(goal_path), interval=interval,
                                       blit=True, repeat=repeat)
        plt.tight_layout()
        plt.show()
        return anim

    def get_reward_grid(self):
        """
        Builds and returns a numpy array containing the rewards associated with each cell.

        :return: np.array of floats with shape (self.rows, self.columns)
        """

        rew_grid = np.full_like(self.grid, 0, dtype=float)
        for t in ((i, j) for i in range(self.rows) for j in range(self.columns)):
            c = self.get_cell(*t)
            if c == Cell.AGENT:
                rew_grid[*t] = -0.04
            else:
                rew_grid[*t] = c.value[0]
        return rew_grid

    def get_obj_grid(self):
        """
        Builds and returns a numpy array containing the names of each cell.

        :return: np.array of strings with shape (self.rows, self.columns)
        """

        obj_grid = np.empty((self.rows, self.columns), object)
        for t in ((i, j) for i in range(self.rows) for j in range(self.columns)):
            obj_grid[*t] = self.get_cell(*t).name

        return obj_grid

    def value_iteration_step(self, old_v, v, policy, x, y):
        """
        Computes a step of the value iteration algorithm on a single state. Used in the complete implementations of both
        the synchronous and asynchronous version of the algorithm.

        :param old_v: value iteration function matrix of the previous step
        :param v: value iteration function matrix of the current step
        :param policy: policy matrix of the current step
        :param x: row coordinate of the state
        :param y: column coordinate of the state
        :return: updated version of the matrices v and policy after performing an iteration of the value iteration
        algorithm
        """

        c = self.get_cell(x, y)
        reward = -0.04 if c == Cell.AGENT else c.value[0]

        if c == Cell.AGENT or c == Cell.FREE:
            poss_dir = self.valid_directions(x, y)
            utility = {}
            for d in poss_dir:
                probs = transition_probabilities(d, poss_dir, 0.8)
                utility[d.name] = sum([prob * old_v[get_next_coord(poss_d, x, y)] for poss_d, prob in probs.items()])

            best_dir = max(utility, key=utility.get)
            val = utility[best_dir]

        else:
            best_dir = Direction.STAY
            val = 0

        v[x][y] = reward + val
        policy[x][y] = best_dir

        return v, policy

    def sync_value_iteration(self, eps=1e-9):
        """
        Computes the synchronous version of the value iteration algorithm on each of the possible states. In this
        version of the algorithm, each iteration updates the values of all the states thus converging with a minimal
        number of iterations. The algorithm stops when |old_v - v| < eps.

        :param eps: value used for the stopping condition
        :return: v matrix containing the converged values of the algorithm, policy matrix containing the best policy for
        each state, number of iterations
        """

        policy = np.full_like(self.grid, Direction.STAY.name)
        v = self.get_reward_grid()

        satisfied = False
        it = 1      # one iteration already performed
        while not satisfied:
            old_v = v.copy()
            for state in ((i, j) for i in range(self.rows) for j in range(self.columns)):
                v, policy = self.value_iteration_step(old_v, v, policy, *state)

            it += 1
            if np.all(abs(old_v - v) < eps):
                satisfied = True

        return v, policy, it

    def async_value_iteration(self, eps=1e-9):
        """
        Computes the asynchronous version of the value iteration algorithm: in this version, at each iteration one state
        is randomly selected and its value is updated, leaving all the other states' values unchanged. Convergence is
        guaranteed provided that all the states are selected infinitely often. The algorithm stops when
        |old_v[state] - v[state]| < eps and all states have been visited.

        :param eps: value used for the stopping condition
        :return: v matrix containing the converged values of the algorithm, policy matrix containing the best policy for
        each state, number of iterations
        """

        policy = np.full_like(self.grid, Direction.STAY.name)
        v = self.get_reward_grid()

        satisfied = False
        it = 1
        visited = set()
        while not satisfied:
            old_v = v.copy()
            state = (random.randint(0, self.rows - 1), random.randint(0, self.columns - 1))
            if state not in visited:
                visited.add(state)
            v, policy = self.value_iteration_step(old_v, v, policy, *state)
            it += 1

            diff = abs(v[*state] - old_v[*state])
            all_visited = len(visited) == self.rows * self.columns
            if diff < eps and all_visited:
                satisfied = True
            elif all_visited:
                visited.clear()

        return v, policy, it


class MovingObject:
    """
    Class representing objects that can move through the grid, such goals, enemies and agents.
    """

    def __init__(self, pos_x, pos_y, type, grid):
        """
        Constructor for MovingObject class. The object can be of type Cell.GOAL, Cell.AGENT or Cell.ENEMY. Each object
        is created with specific coordinates and then it's added to the grid. Each object is responsible for tracking
        its path within the grid.

        :param pos_x: row coordinate
        :param pos_y: column coordinate
        :param type: type of object, among Cell.GOAL, Cell.AGENT or Cell.ENEMY
        :param grid: Grid object where the MovingObject resides.
        """

        if type != Cell.OBSTACLE and type != Cell.FREE:
            if grid.are_valid(pos_x, pos_y):
                self.x = pos_x
                self.y = pos_y
                self.type = type
                self.path = [(pos_x, pos_y)]
                self.grid = grid
                self.grid.add_obj(self)
            else:
                raise TypeError('Invalid coordinates')
        else:
            raise TypeError("Wrong type of cell specified")

    def is_valid(self, direction):
        """
        Whether the cell in the specified direction is reachable.

        :param direction: direction of the adjacent cell
        :return: True if adjacent cell is reachable, False otherwise
        """

        val = False
        if isinstance(direction, Direction):
            val = self.grid.is_reachable(get_next_coord(direction, self.x, self.y))

        return val

    def _move(self, new_x, new_y):
        """
        Move object to new coordinates on the grid.

        :param new_x:new row coordinate
        :param new_y: new column coordinate
        :return:
        """

        self.grid.set_cell(self.x, self.y, Cell.FREE)
        self.x = new_x
        self.y = new_y
        self.grid.set_cell(self.x, self.y, self.type)
        self.path.append((new_x, new_y))

    def move(self, direction):
        """
        Move object in the adjacent cell in the specified direction.

        :param direction: direction of the adjacent cell
        :return:
        """

        next_cell = get_next_coord(direction, self.x, self.y)
        if self.is_valid(direction):
            if self.type == Cell.AGENT:
                self._move(*next_cell)
            elif self.grid.get_cell(*next_cell) == Cell.FREE:   # only the agent can move on goal or enemy
                self._move(*next_cell)
        else:
            self._move(self.x, self.y)

    def random_move(self):
        """
        Make the object move in one of the reachable adjacent cells.

        :return:
        """

        moves = [d for d in Direction]
        valid = False
        while not valid:
            idx = random.randint(0, len(moves) - 2)
            if self.is_valid(moves[idx]):
                valid = True

        self.move(moves[idx])


if __name__ == '__main__':
    rows = 6
    columns = 6
    sync = False
    eps = 1e-9
    random.seed(42)

    g = Grid(rows, columns)
    goal = MovingObject(0, 0, Cell.GOAL, g)
    agent = MovingObject(3, 5, Cell.AGENT, g)
    en1 = MovingObject(4, 3, Cell.ENEMY, g)
    en2 = MovingObject(1, 4, Cell.ENEMY, g)

    g.set_cell(2, 2, Cell.OBSTACLE)

    if sync:
        v, policy, it = g.sync_value_iteration(eps)
    else:
        v, policy, it = g.async_value_iteration(eps)

    print(v)
    print(policy)
    print(f'Iterations: {it}')

    anim = g.show_grid(100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    if sync:
        t_sync = 'Synchronous'
    else:
        t_sync = 'Asynchronous'

    fig.suptitle(f'{t_sync} value iteration - {it} iterations', fontsize=18, x=0.5, y=0.9)

    #                                            ##### POLICY GRID #####

    fig, ax1 = plot_static_grid(rows, columns, fig, ax1)

    for c in ((i, j) for i in range(rows) for j in range(columns)):
        x, y = c
        pol = policy[*c]
        if g.get_cell(*c) != Cell.FREE:
            annotate_grid_plot(ax1, g.get_cell(x, y).name, x, y)
        else:
            annotate_grid_plot(ax1, pol, x, y)

    #                                            ##### VALUES GRID #####

    extent = (0, columns, rows - 1, -1)
    ax2.matshow(v, alpha=0.6, cmap='RdYlGn', extent=extent)
    ax2.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='x', colors='white')
    plt.grid(linewidth=1, color='black')

    for c in ((i, j) for i in range(rows) for j in range(columns)):
        val = round(v[*c], 3)
        if val != 0:
            annotate_grid_plot(ax2, val, *c)
            if abs(val) == 1:
                arts = ax2.findobj(
                    lambda artist: isinstance(artist, Text) and artist.get_text() == f'{val}')
                [art.set_fontweight('bold') for art in arts]
        # else:
        #     annotate_grid_plot(ax2, 'OBSTACLE', *c)

    #                                            ##### CONVERGENCE PLOT #####

    # fig3, ax3 = plt.subplots()
    # keys = list(v_plot.keys())
    # for key in v_plot.keys():
    #     if g.get_cell(*key) != Cell.ENEMY:
    #
    #         # if (key[0] % 2 == 0 and key[1] % 2 == 1) or (key[0] % 2 == 1 and key[1] % 2 == 0):
    #
    #         ax3.plot(v_plot[key], label=f'({key[0]}, {key[1]})')
    #
    #         last_el = v_plot[key][-1]
    #         ax3.text(len(v_plot[key]) -1, last_el, f'({key[0]}, {key[1]})')
    #
    # plt.tight_layout()
    #
    # plt.savefig('img/', dpi=800)
    plt.show()



