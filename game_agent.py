"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


MAX_VAL = float("inf")
MIN_VAL = float("-inf")

DIRECTIONS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2),  (1, 2), (2, -1),  (2, 1)]

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


# def custom_score(game, player):
#     """Calculate the heuristic value of a game state from the point of view
#     of the given player.

#     Note: this function should be called from within a Player instance as
#     `self.score()` -- you should not need to call this function directly.

#     Parameters
#     ----------
#     game : `isolation.Board`
#         An instance of `isolation.Board` encoding the current state of the
#         game (e.g., player locations and blocked cells).

#     player : object
#         A player instance in the current game (i.e., an object corresponding to
#         one of the player objects `game.__player_1__` or `game.__player_2__`.)

#     Returns
#     -------
#     float
#         The heuristic value of the current game state to the specified player.
#     """

def custom_score(game, player):
    # find all legal moves for the current player and their opponent
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # make sure to terminate early if someone is winning or losing
    if not own_moves:
        return float("-inf")
    if not opp_moves:
        return float("inf")

    len_own_moves = len(own_moves)
    len_opp_moves = len(opp_moves)

    # find the best space on the board. This is the space which has the most outer moves
    blank_spaces = game.get_blank_spaces()
    best_reachable = MIN_VAL
    for space in blank_spaces:
        count_reachable = 0
        for direction in DIRECTIONS:
            move = (space[0] + direction[0],  space[1] + direction[1])
            if game.move_is_legal(move):
                count_reachable += 1
        if count_reachable > best_reachable:
            best_reachable = count_reachable
            best_space = space

    # Find the average manhattan distance to the best space from each of the currently legal moves
    # for the current player
    own_average_distance = 0
    for move in own_moves:
        own_average_distance += manhattan_distance(move, best_space)
    own_average_distance /= len_own_moves

    # Find the average manhattan distance to the best space from each of the currently legal moves
    # for the opponent
    opp_average_distance = 0
    for move in opp_moves:
        opp_average_distance += manhattan_distance(move, best_space)
    opp_average_distance /= len_opp_moves

    # Return the difference in number of moves between the current player and the opponent
    # and add a bias towards smaller distances to the best position
    return (len_own_moves - len_opp_moves) + (opp_average_distance - own_average_distance)

def score_with_centre_bias(game, player):
    # find all legal moves for the current player and their opponent
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # make sure to terminate early if someone is winning or losing
    if not own_moves:
        return float("-inf")
    if not opp_moves:
        return float("inf")

    # find the centre square and the current position
    centre = (int(game.width/2), int(game.height/2))
    current_location = game.get_player_location(player)

    # if we're at the centre, the bias towards this position is one
    if (centre == current_location):
        bias = 1
    else:
        bias = 0
        # otherwise, if it's possible to get to the centre
        # and there is a legal move from the current location to it, add a bias of 0.5
        if game.move_is_legal(centre):
            for direction in DIRECTIONS:
                move = (current_location[0] + direction[0], current_location[1] + direction[1])
                if (move == centre):
                    bias = 0.5
                    break
        # alternatively, if there are no legal moves, calculate the average manhattan distance to the
        # centre position. if the distance average is small, add a bias of 0.5
        else:
            distance = 0
            count = 0
            for direction in DIRECTIONS:
                move = (current_location[0] + direction[0], current_location[1] + direction[1])
                if game.move_is_legal(move):
                    count += 1
                    distance += manhattan_distance(move, centre)
            if (count and distance / count) < 1.1:
                bias = 0.5

    # Return the difference in number of moves between the current player and the opponent
    # and add a bias towards smaller distances to centre
    return len(own_moves) - len(opp_moves) + bias

def score_best_square_with_unique_bias(game, player):
    # find all legal moves for the current player and their opponent
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # make sure to terminate early if someone is winning or losing
    if not own_moves:
        return float("-inf")
    if not opp_moves:
        return float("inf")

    len_own_moves = len(own_moves)
    len_opp_moves = len(opp_moves)

    # find the best space on the board. This is the space which has the most outer moves
    blank_spaces = game.get_blank_spaces()
    best_reachable = MIN_VAL
    for space in blank_spaces:
        count_reachable = 0
        for direction in DIRECTIONS:
            move = (space[0] + direction[0],  space[1] + direction[1])
            if game.move_is_legal(move):
                count_reachable += 1
        if count_reachable > best_reachable:
            best_reachable = count_reachable
            best_space = space

    # retrieve the current player's and the opponent's positions
    own_pos = game.get_player_location(player)
    opp_pos = game.get_player_location(game.get_opponent(player))

    # own_average_distance will have a start value of 1 if there is a legal move to the best square for
    # the current player
    own_average_distance = 0
    for direction in DIRECTIONS:
        move = (own_pos[0] + direction[0],  own_pos[1] + direction[1])
        if game.move_is_legal(move) and move == best_space:
            own_average_distance += 1
            break
    # own_average_distance will have a start value of 1 if there is a legal move to the best square for
    # the opponent
    opp_average_distance = 0
    for direction in DIRECTIONS:
        move = (opp_pos[0] + direction[0],  opp_pos[1] + direction[1])
        if game.move_is_legal(move) and move == best_space:
            own_average_distance += 1
            break

    # If the move is unique, to only one of the players, add an extra bias for that
    if opp_average_distance:
        if not own_average_distance:
            opp_average_distance += 1
            own_average_distance -= 1
    else:
        if own_average_distance:
            opp_average_distance -= 1
            own_average_distance += 1

    # calculate the average manhattan distance for the current player
    for move in own_moves:
        own_average_distance += manhattan_distance(move, best_space)
    own_average_distance /= len_own_moves

    # calculate the average manhattan distance for the opponent
    for move in opp_moves:
        opp_average_distance += manhattan_distance(move, best_space)
    opp_average_distance /= len_opp_moves

    # Return the difference in number of moves between the current player and the opponent
    # and add a bias towards smaller distances to the best position
    return (len_own_moves - len_opp_moves) + (opp_average_distance - own_average_distance)


def score_best_square(game, player):
    # find all legal moves for the current player and their opponent
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # make sure to terminate early if someone is winning or losing
    if not own_moves:
        return float("-inf")
    if not opp_moves:
        return float("inf")

    len_own_moves = len(own_moves)
    len_opp_moves = len(opp_moves)

    # find a list of all the best spaces on the board. These are the spaces which have the most outer moves
    blank_spaces = game.get_blank_spaces()
    best_spaces = []
    best_reachable = MIN_VAL
    for space in blank_spaces:
        count_reachable = 0
        for direction in DIRECTIONS:
            move = (space[0] + direction[0],  space[1] + direction[1])
            if game.move_is_legal(move):
                count_reachable += 1
        if count_reachable > best_reachable:
            best_reachable = count_reachable
            best_spaces = [space]
        elif count_reachable == best_reachable:
            best_spaces.append(space)

    # out of all the best spaces, find the one one closest to the center
    smallest_distance = MAX_VAL
    for space in best_spaces:
        distance_to_centre = manhattan_distance(space)
        if distance_to_centre < smallest_distance:
            smallest_distance
            best_space = space

    # calculate the average manhattan distance for the current player
    own_average_distance = 0
    for move in own_moves:
        own_average_distance += manhattan_distance(move, best_space)

    # calculate the average manhattan distance for the opponent
    opp_average_distance = 0
    for move in opp_moves:
        opp_average_distance += manhattan_distance(move, best_space)

    # return the difference between the opponent and the current player's distances.
    # we give better score to lower distances to the best square.
    return opp_average_distance - own_average_distance

def euclidian_distance(pos, pos2=(3, 3)):
    return math.sqrt((pos[0] - pos2[0]) * (pos[0] - pos2[0]) + (pos[1] - pos2[1]) * (pos[1] - pos2[1]))

def manhattan_distance(pos, pos2=(3, 3)):
    return math.fabs(pos2[0] - pos[0]) + math.fabs(pos2[1] - pos[1])



class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        best_move = (-1, -1)
        if (len(legal_moves) == 0):
            return best_move

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            method = getattr(self, self.method)
            if self.iterative:
                depth = 1
                while (True):
                    _, best_move = method(game, depth)
                    depth += 1
            else:
                _, best_move = method(game, self.search_depth)

            return best_move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move

        # Return the best move from the last completed search iteration

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        best_move = (-1, -1)
        best_val = MIN_VAL
        for move in game.get_legal_moves():
            game_copy = game.forecast_move(move)
            current_val = self.minimax_util(game_copy, depth-1, False)
            if (current_val > best_val):
                best_val = current_val
                best_move = move
        return best_val, best_move

    def minimax_util(self, game, depth, maximizing_player):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if (maximizing_player):
            current_player = game.active_player
            minmax_compare = max
            best_val = MIN_VAL
        else:
            current_player = game.inactive_player
            minmax_compare = min
            best_val = MAX_VAL

        if depth == 0:
            return self.score(game, current_player)

        for move in game.get_legal_moves():
            game_copy = game.forecast_move(move)
            best_val = minmax_compare(best_val, self.minimax_util(game_copy, depth-1, not maximizing_player))

        return best_val

        def fill_percentage(game):
            return int((len(game.get_blank_spaces())/(game.width * game.height)) * 100)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        best_move = (-1, -1)
        best_val = MIN_VAL
        for move in game.get_legal_moves():
            game_copy = game.forecast_move(move)
            current_val = self.alphabeta_util(game_copy, depth-1, alpha, beta, False)
            if (current_val > best_val):
                best_val = current_val
                best_move = move
                alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_move

    def alphabeta_util(self, game, depth, alpha, beta, maximizing_player):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, game.active_player if maximizing_player else game.inactive_player)

        if (maximizing_player):
            best_val = MIN_VAL
            for move in game.get_legal_moves():
                game_copy = game.forecast_move(move)
                best_val = max(best_val, self.alphabeta_util(game_copy, depth - 1, alpha, beta, False))
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
        else:
            best_val = MAX_VAL
            for move in game.get_legal_moves():
                game_copy = game.forecast_move(move)
                best_val = min(best_val, self.alphabeta_util(game_copy, depth - 1, alpha, beta, True))
                beta = min(beta, best_val)
                if beta <= alpha:
                    break

        return best_val
