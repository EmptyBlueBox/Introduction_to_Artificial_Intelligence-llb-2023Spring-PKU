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
                self.is_pacman = False
                self.child_node = [None]*5  # 5个动作,N,E,W,S,Stop
                self.parent = None
                self.is_leaf = True  # 相对于已扩展的Monte Carlo Tree来说是否是叶子节点，新扩展的节点都是叶子节点
                self.game_state = data[0]
                self.numerator = data[1]
                self.denominator = data[2]

        data = [gameState, 0, 1]  # data is the data of root node
        mct_root = Node(data)  # mct_root is the root of MCTS tree
        mct_root.is_pacman = True  # root节点是pacman
        # AGENT_NUM is the number of agents in the game
        AGENT_NUM = gameState.getNumAgents()
        C = sqrt(2)  # C is the exploration constant
        EPOCHS = 500  # 模拟次数 500
        DEPTH = 6  # 每次模拟的最大深度 6
        THRESHOLD = 10  # 模拟时取游戏胜利的阈值 10
        NEAREST_K = 10  # 启发函数贪心最近的k个节点 10
        GREEDY_THRESHOLD = 3  # 启发函数贪心的阈值，距离ghost太远就直接搜出来一条距离food最近的路
        index2action = ['North', 'East', 'West', 'South', 'Stop']
        action2index = {'North': 0, 'East': 1,
                        'West': 2, 'South': 3, 'Stop': 4}
        action2dir = {'West': (-1, 0), 'Stop': (0, 0),
                      'East': (1, 0), 'North': (0, 1), 'South': (0, -1)}
        dir = [(-1, 0), (0, 1), (0, -1), (1, 0)]

        ori_pacman_pos = gameState.getPacmanPosition()
        ori_ghost_pos = gameState.getGhostPositions()
        ori_food_pos = gameState.getFood().asList()
        ori_capsule_pos = gameState.getCapsules()
        ori_food_pos += ori_capsule_pos

        def Selection(node, agent_index):
            # 如果当前节点是胜利节点，直接返回1
            if node.game_state.isWin():
                return 1, node, agent_index
            # 如果当前节点是失败节点，直接返回0
            elif node.game_state.isLose():
                return 0, node, agent_index
            # 如果当前节点是叶子节点，直接返回未定的输赢节点
            elif node.is_leaf:
                return -1, node, agent_index

            # 下一个agent的index
            nxt_agent_index = (agent_index + 1) % AGENT_NUM
            # 最大的ucb值
            ucb_val = -float('inf')
            # 当前节点的模拟次数的对数
            ln_t = log(node.denominator)
            # 最大的ucb值对应的节点
            nxt_node = None
            # 如果是pacman就有5个动作（可以Stop），否则就有4个动作
            for child_index in range(5 if agent_index == 0 else 4):
                # 如果子节点已经expand过
                if node.child_node[child_index] is not None:
                    # 如果子节点已经expand但是未simulate，防止计算ucb时除以0
                    if node.child_node[child_index].denominator == 0:
                        return 0, node.child_node[child_index], nxt_agent_index
                    # 如果子节点已经expand并且已经simulate，那么选择ucb最大的节点
                    else:
                        # q是子节点的胜率
                        q = node.child_node[child_index].numerator / \
                            node.child_node[child_index].denominator
                        # n是子节点的模拟次数
                        n = node.child_node[child_index].denominator
                        # 这个子节点的ucb值
                        tmp_ucb_val = q+C*sqrt(ln_t/n)
                        # 如果这个子节点的ucb值大于之前最大的ucb值，那么就更新最大的ucb值和最大的ucb值对应的节点
                        if tmp_ucb_val > ucb_val:
                            ucb_val = tmp_ucb_val
                            nxt_node = node.child_node[child_index]
            return Selection(nxt_node, nxt_agent_index)

        def Expansion(node, agent_index):
            actions = node.game_state.getLegalActions(
                agent_index)  # 获取当前节点的所有合法动作
            if len(actions) == 0:  # 如果没有合法动作，返回False
                return False, node, agent_index
            else:
                node.is_leaf = False  # 如果有合法动作，那么当前节点不是叶子节点

            nxt_agent_index = (agent_index + 1) % AGENT_NUM  # 下一个agent的index
            can_simulate = []  # 可以模拟的动作
            for action in actions:
                # 生成子节点
                child_node = Node(
                    [node.game_state.generateSuccessor(agent_index, action), 0, 0])
                child_node.parent = node
                child_node.is_pacman = True if nxt_agent_index == 0 else False  # 判断子节点是pacman还是ghost
                # 将子节点加入当前节点的子节点列表
                node.child_node[action2index[action]] = child_node
                can_simulate.append(child_node)  # 将子节点加入可以模拟的列表

            # 随机选择一个可以模拟的动作，返回1
            return True, random.choice(can_simulate), nxt_agent_index

        def Simulation(node, agent_index):
            state = node.game_state
            index = agent_index
            for depth in range(DEPTH):
                if state.isWin():
                    return 1  # 这个节点最后得到了确定的结果且是胜利节点
                elif state.isLose():
                    return 0  # 这个节点最后得到了确定的结果且是失败节点

                actions = state.getLegalActions(index)
                if len(actions) == 0:  # 如果没有合法动作，返回失败
                    return 0
                else:  # 随机选择一个合法动作，这里可以进行启发式选择进行优化
                    state = state.generateSuccessor(
                        index, random.choice(actions))
                    index = (index + 1) % AGENT_NUM
            # 这个节点最后没有得到确定的结果，返回启发式函数的结果，函数越大越认为是胜利节点
            return HeuristicFunction(state) >= THRESHOLD

        def Backpropagation(node, is_win):
            while node is not None:
                node.numerator += is_win ^ node.is_pacman
                node.denominator += 1
                node = node.parent

        def HeuristicFunction(game_state):
            current_score = game_state.getScore()
            pacman_pos = game_state.getPacmanPosition()
            ghost_pos = game_state.getGhostPositions()
            food_pos = game_state.getFood().asList()
            min_pacman_ghost_dis = min([manhattanDistance(pacman_pos, ghost_pos[i]) for i in range(
                len(ghost_pos))])  # pacman到ghost的最小距离

            sum_pacman_food_dis = 0
            CNT = 0
            while len(food_pos) > 0:  # 贪心计算pacman到food的总距离
                min_index = None
                min_dis = float('inf')
                for i in range(len(food_pos)):  # 找到距离pacman最近的food
                    tmp_dis = manhattanDistance(
                        pacman_pos, food_pos[i])
                    if tmp_dis < min_dis:
                        min_dis = tmp_dis
                        min_index = i
                sum_pacman_food_dis += min_dis  # 将最近的food的距离加入总距离
                pacman_pos = food_pos[min_index]  # 更新pacman的位置
                del food_pos[min_index]  # 删除已经计算过的food
                if CNT == NEAREST_K:
                    break
                CNT += 1

            return current_score+min_pacman_ghost_dis-sum_pacman_food_dis

        def BFS_using_coordinate():
            frontier = util.Queue()
            vis = [ori_pacman_pos]
            # 先将第一步存起来，方便最后返回答案
            first_actions = gameState.getLegalActions(0)
            x0, y0 = ori_pacman_pos
            for first_action in first_actions:
                dx, dy = action2dir[first_action]
                xx, yy = x0+dx, y0+dy
                frontier.push((xx, yy, first_action))
                vis.append((xx, yy))
                if (xx, yy) in ori_food_pos:
                    return first_action
            walls = gameState.getWalls()
            while frontier.isEmpty() == False:
                x, y, action = frontier.pop()
                if (x, y) in ori_food_pos:
                    return action
                for i in range(4):
                    dx, dy = dir[i]
                    xx, yy = x+dx, y+dy
                    if walls[xx][yy] or (xx, yy) in vis:
                        continue
                    frontier.push((xx, yy, action))
                    vis.append((xx, yy))

        def BFS_find_ghost():
            x0, y0 = ori_pacman_pos
            frontier = util.Queue()
            frontier.push((x0, y0, 0))
            vis = [(x0, y0)]
            walls = gameState.getWalls()

            while frontier.isEmpty() == False:
                x, y, dis = frontier.pop()
                # ghost的坐标是小数
                if (x, y) in ori_ghost_pos or (x+0.5, y) in ori_ghost_pos or (x-0.5, y) in ori_ghost_pos or (x, y+0.5) in ori_ghost_pos or (x, y-0.5) in ori_ghost_pos:
                    return dis
                for i in range(4):
                    dx, dy = dir[i]
                    xx, yy = x+dx, y+dy
                    if walls[xx][yy] or (xx, yy) in vis:
                        continue
                    frontier.push((xx, yy, dis+1))
                    vis.append((xx, yy))

        ori_min_pacman_ghost_dis = BFS_find_ghost()
        if ori_min_pacman_ghost_dis >= GREEDY_THRESHOLD:
            return BFS_using_coordinate()

        for epoch in range(EPOCHS):  # 进行EPOCHS次迭代
            win_or_lose, leaf, leaf_agent_index = Selection(mct_root, 0)  # 选择
            if win_or_lose == -1:  # 不知道叶子节点的胜负
                expand_vaild, leaf, leaf_succ_agent_index = Expansion(
                    leaf, leaf_agent_index)  # 扩展来生成新的叶子节点
                if expand_vaild:
                    win_or_lose = Simulation(leaf, leaf_succ_agent_index)
                else:
                    win_or_lose = 0
            Backpropagation(leaf, win_or_lose)

        ans = None
        max_Q = -float('inf')
        for i in range(5):
            if mct_root.child_node[i] is not None:
                tmp_Q = mct_root.child_node[i].numerator / \
                    mct_root.child_node[i].denominator
                if tmp_Q > max_Q:
                    max_Q = tmp_Q
                    ans = i
        return index2action[ans]
