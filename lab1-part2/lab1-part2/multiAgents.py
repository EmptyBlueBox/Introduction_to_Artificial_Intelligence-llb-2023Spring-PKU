# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util
from math import sqrt, log

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        max_val = -float('inf')
        ret_action = None
        for action in gameState.getLegalActions(0):
            tmp_val = self.get_max_or_min_value(
                gameState.generateSuccessor(0, action), 0, 1)
            if tmp_val != None and tmp_val > max_val:
                max_val = tmp_val
                ret_action = action
        return ret_action

    def get_max_or_min_value(self, game_state, depth, agent_index):
      # agent_index取值在[0, game_state.getNumAgents() - 1]
        legal_actions = game_state.getLegalActions(agent_index)
        if depth == self.depth or len(legal_actions) == 0:
            return self.evaluationFunction(game_state)
        ret_val = float('inf') if agent_index else -float('inf')
        if agent_index == 0:
            # 是pacman就找下一层节点的最大值
            for action in legal_actions:
                tmp_val = self.get_max_or_min_value(game_state.generateSuccessor(
                    agent_index, action), depth, agent_index+1)
                if tmp_val != None and tmp_val > ret_val:
                    ret_val = tmp_val
        elif agent_index == game_state.getNumAgents() - 1:
            # 是ghost就找下一层节点的最小值, 如果是最后一个ghost就更新层数，agent_index变为pacman
            for action in legal_actions:
                tmp_val = self.get_max_or_min_value(game_state.generateSuccessor(
                    agent_index, action), depth+1, 0)
                if tmp_val != None and tmp_val < ret_val:
                    ret_val = tmp_val
        else:
            # 是ghost就找下一层节点的最小值, 如果不是最后一个ghost就不更新层数，只增加agent_index
            for action in legal_actions:
                tmp_val = self.get_max_or_min_value(game_state.generateSuccessor(
                    agent_index, action), depth, agent_index+1)
                if tmp_val != None and tmp_val < ret_val:
                    ret_val = tmp_val
        return ret_val


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        max_val = -float('inf')
        ret_action = None
        for action in gameState.getLegalActions(0):
            tmp_val = self.get_max_or_min_value_ab(
                gameState.generateSuccessor(0, action), 0, 1, max_val, float('inf'))
            if tmp_val != None and tmp_val > max_val:
                max_val = tmp_val
                ret_action = action
        return ret_action

    def get_max_or_min_value_ab(self, game_state, depth, agent_index, alpha, beta):
      # agent_index取值在[0, game_state.getNumAgents() - 1]
        legal_actions = game_state.getLegalActions(agent_index)
        if depth == self.depth or len(legal_actions) == 0:
            return self.evaluationFunction(game_state)
        ret_val = float('inf') if agent_index else -float('inf')
        if agent_index == 0:
            # 是pacman就找下一层节点的最大值
            for action in legal_actions:
                tmp_val = self.get_max_or_min_value_ab(game_state.generateSuccessor(
                    agent_index, action), depth, agent_index+1, alpha, beta)
                if tmp_val != None and tmp_val > ret_val:
                    ret_val = tmp_val
                if tmp_val > beta:
                    return tmp_val
                if tmp_val > alpha:
                    alpha = tmp_val
        elif agent_index == game_state.getNumAgents() - 1:
            # 是ghost就找下一层节点的最小值, 如果是最后一个ghost就更新层数，agent_index变为pacman
            for action in legal_actions:
                tmp_val = self.get_max_or_min_value_ab(game_state.generateSuccessor(
                    agent_index, action), depth+1, 0, alpha, beta)
                if tmp_val != None and tmp_val < ret_val:
                    ret_val = tmp_val
                if tmp_val < alpha:
                    return tmp_val
                if tmp_val < beta:
                    beta = tmp_val
        else:
            # 是ghost就找下一层节点的最小值, 如果不是最后一个ghost就不更新层数，只增加agent_index
            for action in legal_actions:
                tmp_val = self.get_max_or_min_value_ab(game_state.generateSuccessor(
                    agent_index, action), depth, agent_index+1, alpha, beta)
                if tmp_val != None and tmp_val < ret_val:
                    ret_val = tmp_val
                if ret_val < alpha:
                    return ret_val
                if tmp_val < beta:
                    beta = tmp_val
        return ret_val


class MCTSAgent(MultiAgentSearchAgent):
    """
      Your MCTS agent with Monte Carlo Tree Search (question 3)
    """

    def getAction(self, gameState):

        class Node:
            '''
            We have provided node structure that you might need in MCTS tree.
            '''

            def __init__(self, data):
                self.north = None
                self.east = None
                self.west = None
                self.south = None
                self.stop = None
                self.parent = None
                self.game_state = data[0]
                self.numerator = data[1]
                self.denominator = data[2]

        data = [gameState, 0, 1]
        mct_root = Node(data)
        epochs = 1000

        def Selection(root):
            "*** YOUR CODE HERE ***"

        def Expansion(cgs, cgstree):
            "*** YOUR CODE HERE ***"
            util.raiseNotDefined()

        def Simulation(cgs, cgstree):
            "*** YOUR CODE HERE ***"
            util.raiseNotDefined()

        def Backpropagation(cgstree, WinorLose):
            "*** YOUR CODE HERE ***"
            util.raiseNotDefined()

        def HeuristicFunction(currentGameState):
            "*** YOUR CODE HERE ***"
            util.raiseNotDefined()

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
