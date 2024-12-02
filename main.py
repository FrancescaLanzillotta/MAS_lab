import random
import time
from enum import Enum
from collections import deque
import matplotlib.artist
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class Cell(Enum):
    AGENT = 0.001,
    FAIL = -1,
    FREE = -0.04,
    OBSTACLE = 0,
    GOAL = 1,


class Direction(Enum):
    UP = (-1, 0),
    DOWN = (1, 0),
    LEFT = (0, -1),
    RIGHT = (0, 1),
    STAY = (0, 0),


def to_arrow(string):
    s = ''
    if string == 'UP':
        s = '\u2191'
    elif string == 'DOWN':
        s = '\u2193'
    elif string == 'RIGHT':
        s = '\u2192'
    elif string == 'LEFT':
        s = '\u2190'

    return s


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


def plot_value_iteration(grid, v, policy, it, sync, display=True):
    """
    Helper method to plot the results found using the value iteration algorithm. Plots both the policy and the
    values

    :param grid: Grid object
    :param v: matrix containing the values found by VI
    :param policy: matrix containing the optimal policy for each cell
    :param it: number of iterations
    :param sync: True if using the synchronous version of the algorithm, False if using the asynchronous version
    :param display: whether to display the plots or not
    :return: plot's figure and axes
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    if sync:
        t_sync = 'Synchronous'
    else:
        t_sync = 'Asynchronous'

    fig.suptitle(f'{t_sync} value iteration - {it} iterations', fontsize=18, x=0.5, y=0.9)

    #                                            ##### POLICY GRID #####

    fig, ax1 = plot_static_grid(grid.rows, grid.columns, fig, ax1)

    for c in ((i, j) for i in range(grid.rows) for j in range(grid.columns)):
        x, y = c
        pol = policy[*c]
        if grid.get_cell(*c) != Cell.FREE and grid.get_cell(*c) != Cell.AGENT:

            annotate_grid_plot(ax1, grid.get_cell(x, y).name, x, y)
            # annotate_grid_plot(ax1, pol, x, y)
        else:
            annotate_grid_plot(ax1, pol, x, y)

    #                                            ##### VALUES GRID #####

    extent = (0, grid.columns, grid.rows - 1, -1)

    ax2.matshow(v, alpha=0.6, cmap='RdYlGn', extent=extent)
    ax2.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='x', colors='white')
    plt.grid(linewidth=1, color='black')

    for c in ((i, j) for i in range(grid.rows) for j in range(grid.columns)):
        if grid.get_cell(*c) == Cell.FAIL:
            val = -1.0
        elif grid.get_cell(*c) == Cell.GOAL:
            val = 1.0
        else:
            val = round(v[*c], 3)

        if val != 0:
            annotate_grid_plot(ax2, val, *c)
            if abs(val) == 1:
                arts = ax2.findobj(
                    lambda artist: isinstance(artist, Text) and artist.get_text() == f'{val}')
                [art.set_fontweight('bold') for art in arts]
        else:
            annotate_grid_plot(ax2, 'OBSTACLE', *c)

    if display:
        plt.show()

    return fig, (ax1, ax2)


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
        col = 'black'
        size = 18
        x_offset = 0.45
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
        elif type == 'FAIL':
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
        if len(poss) > 1:
            if d == direction:
                probs[d] = main_prob
            else:
                probs[d] = round((1 - main_prob) / (len(poss) - 1), 3)
        else:
            probs[d] = 1

    return probs


def random_transitions_probabilities(possible_moves):
    probs = {}
    prob = round(1 / len(possible_moves), 3)
    for d in possible_moves:
        probs[d] = prob

    return probs


class Grid:
    """
    Class representing grid world were the objects reside. The grid can contain obstacles, a goal cell,
    a fail cell and an agent. The objects that can move within the grid are tracked through class attributes.
    """

    def __init__(self, rows, columns):
        """
        Constructor for Grid class. Builds a grid of specified dimension containing only FREE cell type.
        :param rows: number of rows in the desired grid
        :param columns: number of columns in the desired grid
        """

        self.rows = rows
        self.columns = columns
        self.goal = None
        self.agent = None
        self.fail = None
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
        Adds MovingObject to the grid in the object's coordinates if the cell is free. The object is added to the
        specific tracking attribute of the Grid class depending on its type.

        :param obj: MovingObject to add to grid
        :return:
        """

        if isinstance(obj, MovingObject) and self.get_cell(obj.x, obj.y) == Cell.FREE:

            if obj.type == Cell.GOAL and self.goal is None:
                self.goal = obj
            elif obj.type == Cell.AGENT and self.agent is None:
                self.agent = obj
            elif obj.type == Cell.FAIL and self.fail is None:
                self.fail = obj

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

    def show_grid(self, policy, interval=500, repeat=True):
        """
        Shows an animated plot of the grid. If the objects in the grid move, the animation will show the movements of
        all the objects and the best policies for each cell. This function relies on the fact that all objects have
        paths of the same length: the objects' paths are padded according to the length of the longest path among the
        objects on the grid.

        :param policy: policy found using VI algorithm.
        :param interval: delay between frames in milliseconds
        :param repeat: whether the animation repeats when the sequence of frames is completed
        :return: Animation variable
        """

        fig, ax = plot_static_grid(self.rows, self.columns)

        m = max(len(self.agent.path), len(self.goal.path), len(self.fail.path))
        if m > 0:
            if len(self.agent.path) < m:
                self.agent.path += [self.agent.path[-1]] * (m - len(self.agent.path))
            if len(self.goal.path) < m:
                self.goal.path += [self.goal.path[-1]] * (m - len(self.goal.path))
            if len(self.fail.path) < m:
                self.fail.path += [self.fail.path[-1]] * (m - len(self.fail.path))

        enemy_art = annotate_grid_plot(ax, 'FAIL', *self.fail.path[0])

        agent_art = annotate_grid_plot(ax, 'AGENT', *self.agent.path[0])

        goal_art = annotate_grid_plot(ax, 'GOAL', *self.goal.path[0])

        pol_arts = np.empty_like(self.grid, dtype=matplotlib.artist.Artist)
        for c in ((i, j) for i in range(self.rows) for j in range(self.columns)):
            pol = policy[*self.goal.path[0], *self.fail.path[0], *c]
            if self.get_cell(*c) == Cell.FREE:
                pol_arts[*c] = annotate_grid_plot(ax, pol, *c)
            else:
                pol_arts[*c] = annotate_grid_plot(ax, '', *c)

        def update(frame):

            x_goal, y_goal = self.goal.path[frame]

            x_fail, y_fail = self.fail.path[frame]
            enemy_art.set_x(y_fail + 0.5)
            enemy_art.set_y(x_fail - 0.5)

            on_goal = False
            x_agent, y_agent = self.agent.path[frame]
            agent_art.set_x(y_agent + 0.5)
            agent_art.set_y(x_agent - 0.5)
            if x_agent == x_goal and y_agent == y_goal:
                on_goal = True

            if not on_goal:
                goal_art.set_x(y_goal + 0.5)
                goal_art.set_y(x_goal - 0.4)
            else:
                goal_art.set_x(-5)
                goal_art.set_y(-5)

            for c in ((i, j) for i in range(self.rows) for j in range(self.columns)):
                pol = policy[x_goal, y_goal, x_fail, y_fail, *c]
                if c != (x_goal, y_goal) and c != (x_agent, y_agent) and c != (x_fail, y_fail):
                    pol_arts[*c].set_text(to_arrow(pol))
                else:
                    pol_arts[*c].set_text('')

            return goal_art, enemy_art, agent_art, *pol_arts

        global anim     # needs to be global to avoid garbage collection
        anim = animation.FuncAnimation(fig=fig, func=update, frames=len(self.goal.path), interval=interval,
                                       blit=False, repeat=repeat)
        plt.tight_layout()

        plt.show()
        return anim

    def value_iteration_step(self, old_v, x_goal, y_goal, x_fail, y_fail, x, y):
        """
        Computes a step of the value iteration algorithm on a single state (x_goal, y_goal, x_fail, y_fail, x, y).
        Used in the complete implementations of both the synchronous and asynchronous version of the algorithm.

        :param old_v: value iteration function matrix of the previous step
        :param x_goal: row coordinate of the goal state
        :param y_goal: column coordinate of the goal state
        :param x_fail: row coordinate of the fail state
        :param y_fail: column coordinate of the fail state
        :param x: row coordinate of the cell
        :param y: column coordinate of the cell

        :return: updated value for the state (x_goal, y_goal, x_fail, y_fail, x, y)
        """

        on_goal = x == x_goal and y == y_goal
        on_fail = x == x_fail and y == y_fail

        if on_goal:
            reward = 1
        elif on_fail:
            reward = -1
        else:
            reward = -0.04

        if not on_goal and not on_fail:
            poss_dir = self.valid_directions(x, y)
            g_poss_dir = self.valid_directions(x_goal, y_goal) + [Direction.STAY]
            en_poss_dir = self.valid_directions(x_fail, y_fail) + [Direction.STAY]
            g_probs = random_transitions_probabilities(g_poss_dir)
            en_probs = random_transitions_probabilities(en_poss_dir)
            utility = {}
            for d in poss_dir:
                c_probs = transition_probabilities(d, poss_dir, 0.8)
                utility[d.name] = sum([c_prob * g_prob * en_prob * old_v[x_goal, y_goal, x_fail, y_fail, *get_next_coord(c_poss_d, x, y)]
                                       for c_poss_d, c_prob in c_probs.items()
                                       for g_prob in g_probs.values()
                                       for en_prob in en_probs.values()
                                       ])

            best_dir = max(utility, key=utility.get)
            val = utility[best_dir]

        else:
            best_dir = Direction.STAY.name
            val = 0

        return reward + val, best_dir

    def sync_value_iteration(self, eps=1e-9):
        """
        Computes the synchronous version of the value iteration algorithm on each of the possible states. In this
        version of the algorithm, each iteration updates the values of all the states thus converging with a minimal
        number of iterations. The algorithm stops when |old_v - v| < eps.

        :param eps: value used for the stopping condition
        :return: v matrix containing the converged values of the algorithm for each state, policy matrix containing the
        best policy for each state, number of iterations performed
        """

        v = np.zeros((self.rows, self.columns, self.rows, self.columns, self.rows, self.columns))
        policy = np.full_like(v, Direction.STAY.name, dtype=object)
        satisfied = False
        it = 0
        while not satisfied:
            old_v = v.copy()
            for state in ((i, j) for i in range(self.rows) for j in range(self.columns)):
                for g_state in ((i, j) for i in range(self.rows) for j in range(self.columns)):
                    for f_state in ((i, j) for i in range(self.rows) for j in range(self.columns)):
                        v[*g_state, *f_state, *state], policy[*g_state, *f_state, *state] = self.value_iteration_step(
                            old_v, *g_state, *f_state, *state)

            it += 1

            if np.all(abs(old_v - v) < eps) or it > 300000:
                satisfied = True

        return v, policy, it

    def async_value_iteration(self, eps=1e-9):
        """
        Computes the asynchronous version of the value iteration algorithm: in this version, at each iteration one state
        is randomly selected and its value is updated, leaving all the other states' values unchanged. Convergence is
        guaranteed provided that all the states are selected infinitely often. The algorithm stops when
        |v_{k} - v_{k + 50}| < eps.

        :param eps: value used for the stopping condition
        :return: v matrix containing the converged values of the algorithm for each state, policy matrix containing the
        best policy for each state, number of iterations performed
        """

        v = np.zeros((self.rows, self.columns, self.rows, self.columns, self.rows, self.columns))
        policy = np.full_like(v, Direction.STAY.name, dtype=object)

        satisfied = False
        it = 0
        history = deque([v.copy()])
        while not satisfied:
            old_v = v.copy()
            state = (random.randint(0, self.rows - 1), random.randint(0, self.columns - 1))
            g_state = (random.randint(0, self.rows - 1), random.randint(0, self.columns - 1))
            f_state = (random.randint(0, self.rows - 1), random.randint(0, self.columns - 1))
            while f_state == g_state:
                f_state = (random.randint(0, self.rows - 1), random.randint(0, self.columns - 1))

            v[*g_state, *f_state, *state], policy[*g_state, *f_state, *state] = self.value_iteration_step(
                old_v, *g_state, *f_state, *state)

            it += 1
            if len(history) >= 50:
                first = history.popleft()
                if (abs(first - v) < eps).all() or it > 10000000:
                    satisfied = True
                else:
                    history.append(v.copy())
            else:
                history.append(v.copy())

        return v, policy, it


class MovingObject:
    """
    Class representing objects that can move through the grid, such as the goal, the failure and agent.
    """

    def __init__(self, pos_x, pos_y, type, grid):
        """
        Constructor for MovingObject class. The object can be of type Cell.GOAL, Cell.AGENT or Cell.FAIL. Each object
        is created with specific coordinates and then it's added to the grid. Each object is responsible for tracking
        its path within the grid.

        :param pos_x: row coordinate
        :param pos_y: column coordinate
        :param type: type of object, among Cell.GOAL, Cell.AGENT or Cell.FAIL
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
            val = self.grid.is_reachable(*get_next_coord(direction, self.x, self.y))

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
            idx = random.randint(0, len(moves) - 1)
            valid = self.is_valid(moves[idx])

        self.move(moves[idx])


if __name__ == '__main__':
    rows = 5
    columns = 5
    sync = False
    eps = 1e-6
    # random.seed(1)

    g = Grid(rows, columns)

    start = time.time()
    v, policy, it = g.sync_value_iteration(eps)
    end = time.time()
    print(f'Sync. value iteration - {it} iterations performed ({round(end - start, 3)}) s')

    start = time.time()
    v, policy, it = g.async_value_iteration(eps)
    end = time.time()
    print(f'Async. value iteration - {it} iterations performed ({round(end - start, 3)}) s')

    # if sync:
    #     start = time.time()
    #     v, policy, it = g.sync_value_iteration(eps)
    #     end = time.time()
    # else:
    #     start = time.time()
    #     v, policy, it = g.async_value_iteration(eps)
    #     end = time.time()
    #
    # if sync:
    #     s = 'sync'
    # else:
    #     s = 'async'
    # print(f'{s} value iteration - {it} iterations performed ({round(end - start, 3)}) s')

    # goal = MovingObject(0, 0, Cell.GOAL, g)
    # agent = MovingObject(rows - 1, columns - 1, Cell.AGENT, g)
    # fail = MovingObject(math.floor(rows / 2), math.floor(columns / 2), Cell.FAIL, g)
    #
    # on_goal = agent.x == goal.x and agent.y == goal.y
    # on_fail = agent.x == fail.x and agent.y == fail.y
    #
    # while not on_goal and not on_fail:
    #     direction = policy[goal.x, goal.y, fail.x, fail.y, agent.x, agent.y]
    #     agent.move(Direction[direction])
    #     on_goal = agent.x == goal.x and agent.y == goal.y
    #     on_fail = agent.x == fail.x and agent.y == fail.y
    #     if not on_goal and not on_fail:
    #         goal.random_move()
    #         fail.random_move()
    #         on_goal = agent.x == goal.x and agent.y == goal.y
    #         on_fail = agent.x == fail.x and agent.y == fail.y
    #
    # anim = g.show_grid(policy, 1000)
    #
    #
    # anim.save(f'{s}-VI-{rows}x{columns} grid.gif', writer=animation.PillowWriter(fps=1, bitrate=1800), dpi=600)
    # anim.save(f'{s}-VI-{rows}x{columns} grid.mp4', animation.FFMpegWriter(fps=1), dpi=600)






