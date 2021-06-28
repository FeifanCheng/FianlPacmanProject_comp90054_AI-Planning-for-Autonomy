# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
import math
import distanceCalculator
from valueIterationAgents import ValueIterationAgent
from util import nearestPoint

# observe enermies if manhattanDistance <= 5
SIGHT_RANGE = 5

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MyOffensiveAgent', second = 'MyDefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    # get the width, height, middle_X, boundaries of layout at first
    self.start = gameState.getAgentPosition(self.index)
    self.width = gameState.data.layout.width
    self.middle_X = int(self.width / 2)
    self.height = gameState.data.layout.height
    self.boundaries = self.getBoundaries(gameState)

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)
  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class MyOffensiveAgent(DummyAgent):
  # offensive agent

    

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.gameState = gameState

    # nearest food to the agent
    self.nearestFood = None

    # nearest capsule to the agent
    self.nearestCapsule = None

    # nearest enemy to the agent
    self.nearestEnemy = None

    # nearest invader to the agent
    self.nearestInvader = None

    # nearest safe spot to the agent
    self.nearestSafeSpot = None

    # position of ally
    self.allyPos = None

    # total food to eat at the start of a game
    self.totalFood = self.getFood(gameState).count()

  # choose the next action
  def chooseAction(self, gameState):  

    decision = self.decide(gameState)
    if decision == 1:
      valueIterationAgent = ValueIterationAgent(FoodMDP(gameState, self), iterations=20)
      action = valueIterationAgent.getAction(gameState.getAgentPosition(self.index))
      return action
    elif decision == 2:
      valueIterationAgent = ValueIterationAgent(CapsuleMDP(gameState, self), iterations=20)
      action = valueIterationAgent.getAction(gameState.getAgentPosition(self.index))
      return action
    elif decision == 3:
      valueIterationAgent = ValueIterationAgent(GetBackMDP(gameState, self), iterations=20)
      action = valueIterationAgent.getAction(gameState.getAgentPosition(self.index))
      return action
    else:
      valueIterationAgent = ValueIterationAgent(ChaseMDP(gameState, self), iterations=20)
      action = valueIterationAgent.getAction(gameState.getAgentPosition(self.index))
      return action

  def decide(self, gameState):
    foods = self.getFood(gameState)
    walls = gameState.getWalls()
    capsules = self.getCapsules(gameState)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    self.enemyPos = [i.getPosition() for i in enemies if (not i.isPacman) and i.scaredTimer == 0 and i.getPosition() != None]
    self.invaderPos = [i.getPosition() for i in enemies if i.isPacman and i.getPosition() != None]
    position = gameState.getAgentPosition(self.index)
    # minus 1 because the x axis starts from 0
    mid_x = gameState.data.layout.width // 2 - 1  
    agentState = gameState.getAgentState(self.index)
    
    if self.red:
      self.safeSpots = [(mid_x, y) for y in range(gameState.data.layout.height) if not walls[mid_x][y]]
    else:
      self.safeSpots = [(mid_x + 1, y) for y in range(gameState.data.layout.height) if not walls[mid_x + 1][y]]
    timeleft = gameState.data.timeleft
    
    self.nearestEnemy = self.getNearestEnemy(self.enemyPos, position)
    self.nearestFood = self.getNearestFood(foods, position, self.nearestEnemy)
    self.nearestCapsule = self.getNearestCapsule(capsules, position, self.nearestEnemy)
    self.nearestSafeSpot = self.getNearestSafeSpot(self.safeSpots, position, self.nearestEnemy)
    self.nearestInvader = self.getNearestInvader(self.invaderPos, position)
    self.allyPos = [gameState.getAgentPosition(i) for i in self.getTeam(gameState) if i != self.index][0]

    maxCarry = self.totalFood // 4

    
    if agentState.numCarrying == 0:
      if (not agentState.isPacman) and self.nearestInvader != None and agentState.scaredTimer == 0:
        if self.getMazeDistance(position, self.nearestInvader) < self.getMazeDistance(position, self.allyPos) + self.getMazeDistance(self.nearestInvader, self.allyPos):
          # chase the invader
          return 4
      # Eat the nearest food
      return 1
    
    if self.nearestEnemy == None:
      if self.getMazeDistance(position, self.nearestSafeSpot) + 10 > timeleft // 4:
        # go back home
        return 3
      if agentState.numCarrying > maxCarry  or foods.count() <= 2:
        # go back home
        return 3
      else:
        # Eat the nearest food
        return 1
    else:
      if self.getMazeDistance(position, self.nearestSafeSpot) + 10 > timeleft // 4:
        # go back home
        return 3
      if self.nearestCapsule != None:
        if self.getMazeDistance(position, self.nearestCapsule) < self.getMazeDistance(self.nearestEnemy, self.nearestCapsule):
          # eat the capsule
          return 2
      if agentState.numCarrying <= maxCarry and foods.count() > 2:
        if self.getMazeDistance(position, self.nearestFood) + 2 < self.getMazeDistance(self.nearestEnemy, self.nearestFood):
          # eat the food
          return 1
      # go back home
      return 3
      

  # Get the position of the nearest food
  def getNearestFood(self, foods, myPos, enemyPos):
    max_score = -999999
    nearestFood = None
    for food in foods.asList():
      # If there is no enemy, only the distance between food and agent is considered
      if enemyPos == None:
        score = 999999 - self.getMazeDistance(myPos, food)
      else:
        # If the scores of the two foods are the same, take the one closer to the agent
        score = (self.getMazeDistance(enemyPos, food) - self.getMazeDistance(myPos, food)) * 1000 - self.getMazeDistance(myPos, food)
      if score > max_score:
        max_score = score
        nearestFood = food
    return nearestFood

  # Get the position of the nearest enemy
  def getNearestEnemy(self, enemyPos, myPos):
    min_dist = 999999
    # get the latest value of the nearestEnemy
    nearestEnemy = self.nearestEnemy
    if nearestEnemy != None:
      # If the enemy is far away from us, discard the value.
      if self.getMazeDistance(myPos, nearestEnemy) > 10:
        nearestEnemy = None
      
      elif util.manhattanDistance(myPos, nearestEnemy) < 5 and enemyPos == []:
        nearestEnemy = None 
  
    for pos in enemyPos:
      dist = self.getMazeDistance(myPos, pos)
      if dist < min_dist and util.manhattanDistance(myPos, pos) <= 5:
        min_dist = dist
        nearestEnemy = pos
    return nearestEnemy

  # Get the position of the nearest capsule
  def getNearestCapsule(self, capsules, myPos, enemyPos):
    max_score = -999999
    nearestCapsule = None
    for capsule in capsules:
      if enemyPos == None:
        score = 999999 - self.getMazeDistance(myPos, capsule)
      else:
        score = self.getMazeDistance(enemyPos, capsule) - self.getMazeDistance(myPos, capsule)
      if score > max_score:
        max_score = score
        nearestCapsule = capsule
    return nearestCapsule

  # Get the position of the nearest safe spot 
  def getNearestSafeSpot(self, safeSpots, myPos, enemyPos):
    max_score = -999999
    nearestSafeSpot = None
    for safeSpot in safeSpots:
      if enemyPos == None:
        score = 999999 - self.getMazeDistance(myPos, safeSpot)
      else:
        score = (self.getMazeDistance(enemyPos, safeSpot) - self.getMazeDistance(myPos, safeSpot)) * 1000 - self.getMazeDistance(myPos, safeSpot)
      if score > max_score:
        max_score = score
        nearestSafeSpot = safeSpot
    return nearestSafeSpot

  # Get the position of the nearest invader
  def getNearestInvader(self, invaderPos, myPos):
    min_dist = 999999
    nearestInvader = None

    for pos in invaderPos:
      dist = self.getMazeDistance(myPos, pos)
      if dist < min_dist:
        min_dist = dist
        nearestInvader = pos
    return nearestInvader


# MDP models
class FoodMDP():
  def __init__(self, gameState, agent):
    self.gameState = gameState
    self.noise = 0.3
    self.agent = agent
    self.grid = gameState.getWalls()
    self.nearestFood = agent.nearestFood
    self.foods = agent.getFood(gameState)
    self.states = self.getStates()
    self.enemyPos = agent.enemyPos

  def getStates(self):
    states = ['TERMINAL_STATE']
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if not self.grid[x][y] and util.manhattanDistance(self.getStartState(), (x,y)) <= 7:
          states.append((x,y))
    return states

  def getStartState(self):
    return self.gameState.getAgentPosition(self.agent.index)

  def getPossibleActions(self, state):
    if state == 'TERMINAL_STATE':
      return []
    x,y = state
    if self.foods[x][y] or state in self.enemyPos:
      return ['Exit']

    actions = []
    if (x, y+1) in self.states:
      actions.append('North')
    if (x, y-1) in self.states:
      actions.append('South')
    if (x-1, y) in self.states:
      actions.append('West')
    if (x+1, y) in self.states:
      actions.append('East')
    return actions

  def getTransitionStatesAndProbs(self, state, action):
    if self.isTerminal(state):
      return []
    if action == 'Exit':
      return [('TERMINAL_STATE', 1.0)]
    
    x, y = state
    successors = []
    transitions = {'North': (x, y), 'South': (x, y), 'West': (x, y), 'East': (x, y)}
    if (x, y+1) in self.states:
      transitions['North'] = (x, y+1)
    if (x, y-1) in self.states:
      transitions['South'] = (x, y-1)
    if (x-1, y) in self.states:
      transitions['West'] = (x-1, y)
    if (x+1, y) in self.states:
      transitions['East'] = (x+1, y)
    
    if action == 'North':
      successors.append(((x, y+1), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))

    elif action == 'South':
      successors.append(((x, y-1), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    elif action == 'West':
      successors.append(((x-1, y), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    elif action == 'East':
      successors.append(((x+1, y), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    return successors

  def getReward(self, state, action, nextState):
    if state == 'TERMINAL_STATE':
      return 0.0
    if state in self.enemyPos:
      return -2000

    x, y = state
    if self.foods[x][y]:
      return 1000
    if self.nearestFood != None:
      return 10 - self.agent.getMazeDistance(state, self.nearestFood)
    else:
      return 0.0

  def isTerminal(self, state):
    return state == 'TERMINAL_STATE'

class CapsuleMDP():
  def __init__(self, gameState, agent):
    self.gameState = gameState
    self.noise = 0.3
    self.agent = agent
    self.grid = gameState.getWalls()
    self.nearestCapsule = agent.nearestCapsule
    self.capsules = agent.getCapsules(gameState)
    self.states = self.getStates()
    self.enemyPos = agent.enemyPos

  def getStates(self):
    states = ['TERMINAL_STATE']
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if not self.grid[x][y] and util.manhattanDistance(self.getStartState(), (x,y)) <= 7:
          states.append((x,y))
    return states

  def getStartState(self):
    return self.gameState.getAgentPosition(self.agent.index)

  def getPossibleActions(self, state):
    if state == 'TERMINAL_STATE':
      return []
    x,y = state
    if state in self.capsules or state in self.enemyPos:
      return ['Exit']

    actions = []
    if (x, y+1) in self.states:
      actions.append('North')
    if (x, y-1) in self.states:
      actions.append('South')
    if (x-1, y) in self.states:
      actions.append('West')
    if (x+1, y) in self.states:
      actions.append('East')
    return actions

  def getTransitionStatesAndProbs(self, state, action):
    if self.isTerminal(state):
      return []
    if action == 'Exit':
      return [('TERMINAL_STATE', 1.0)]
    
    x, y = state
    successors = []
    transitions = {'North': (x, y), 'South': (x, y), 'West': (x, y), 'East': (x, y)}
    if (x, y+1) in self.states:
      transitions['North'] = (x, y+1)
    if (x, y-1) in self.states:
      transitions['South'] = (x, y-1)
    if (x-1, y) in self.states:
      transitions['West'] = (x-1, y)
    if (x+1, y) in self.states:
      transitions['East'] = (x+1, y)
    
    if action == 'North':
      successors.append(((x, y+1), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))

    elif action == 'South':
      successors.append(((x, y-1), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    elif action == 'West':
      successors.append(((x-1, y), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    elif action == 'East':
      successors.append(((x+1, y), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    return successors

  def getReward(self, state, action, nextState):
    if state == 'TERMINAL_STATE':
      return 0.0
    if state in self.enemyPos:
      return -2000

    
    if state in self.capsules:
      return 1000
    else:
      return 10 - self.agent.getMazeDistance(state, self.nearestCapsule)

  def isTerminal(self, state):
    return state == 'TERMINAL_STATE'

class GetBackMDP():
  def __init__(self, gameState, agent):
    self.gameState = gameState
    self.noise = 0.3
    self.agent = agent
    self.grid = gameState.getWalls()
    self.nearestSafeSpot = agent.nearestSafeSpot
    self.safeSpots = agent.safeSpots
    self.states = self.getStates()
    self.enemyPos = agent.enemyPos

  def getStates(self):
    states = ['TERMINAL_STATE']
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if not self.grid[x][y] and util.manhattanDistance(self.getStartState(), (x,y)) <= 7:
          states.append((x,y))
    return states

  def getStartState(self):
    return self.gameState.getAgentPosition(self.agent.index)

  def getPossibleActions(self, state):
    if state == 'TERMINAL_STATE':
      return []
    x,y = state
    if state in self.safeSpots or state in self.enemyPos:
      return ['Exit']

    actions = []
    if (x, y+1) in self.states:
      actions.append('North')
    if (x, y-1) in self.states:
      actions.append('South')
    if (x-1, y) in self.states:
      actions.append('West')
    if (x+1, y) in self.states:
      actions.append('East')
    return actions

  def getTransitionStatesAndProbs(self, state, action):
    if self.isTerminal(state):
      return []
    if action == 'Exit':
      return [('TERMINAL_STATE', 1.0)]
    
    x, y = state
    successors = []
    transitions = {'North': (x, y), 'South': (x, y), 'West': (x, y), 'East': (x, y)}
    if (x, y+1) in self.states:
      transitions['North'] = (x, y+1)
    if (x, y-1) in self.states:
      transitions['South'] = (x, y-1)
    if (x-1, y) in self.states:
      transitions['West'] = (x-1, y)
    if (x+1, y) in self.states:
      transitions['East'] = (x+1, y)
    
    if action == 'North':
      successors.append(((x, y+1), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))

    elif action == 'South':
      successors.append(((x, y-1), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    elif action == 'West':
      successors.append(((x-1, y), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    elif action == 'East':
      successors.append(((x+1, y), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    return successors

  def getReward(self, state, action, nextState):
    if state == 'TERMINAL_STATE':
      return 0.0
    if state in self.enemyPos:
      return -3000

    
    if state in self.safeSpots:
      return 1000
    else:
      return 10 - self.agent.getMazeDistance(state, self.nearestSafeSpot)

  def isTerminal(self, state):
    return state == 'TERMINAL_STATE'
  
class ChaseMDP():
  def __init__(self, gameState, agent):
    self.gameState = gameState
    self.noise = 0.3
    self.agent = agent
    self.grid = gameState.getWalls()
    self.nearestInvader = agent.nearestInvader
    self.invaderPos = agent.invaderPos
    self.states = self.getStates()

  def getStates(self):
    states = ['TERMINAL_STATE']
    for x in range(self.grid.width):
      for y in range(self.grid.height):
        if not self.grid[x][y] and util.manhattanDistance(self.getStartState(), (x,y)) <= 7:
          states.append((x,y))
    return states

  def getStartState(self):
    return self.gameState.getAgentPosition(self.agent.index)

  def getPossibleActions(self, state):
    if state == 'TERMINAL_STATE':
      return []
    x,y = state
    if state in self.invaderPos:
      return ['Exit']

    actions = []
    if (x, y+1) in self.states:
      actions.append('North')
    if (x, y-1) in self.states:
      actions.append('South')
    if (x-1, y) in self.states:
      actions.append('West')
    if (x+1, y) in self.states:
      actions.append('East')
    return actions

  def getTransitionStatesAndProbs(self, state, action):
    if self.isTerminal(state):
      return []
    if action == 'Exit':
      return [('TERMINAL_STATE', 1.0)]
    
    x, y = state
    successors = []
    transitions = {'North': (x, y), 'South': (x, y), 'West': (x, y), 'East': (x, y)}
    if (x, y+1) in self.states:
      transitions['North'] = (x, y+1)
    if (x, y-1) in self.states:
      transitions['South'] = (x, y-1)
    if (x-1, y) in self.states:
      transitions['West'] = (x-1, y)
    if (x+1, y) in self.states:
      transitions['East'] = (x+1, y)
    
    if action == 'North':
      successors.append(((x, y+1), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))

    elif action == 'South':
      successors.append(((x, y-1), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    elif action == 'West':
      successors.append(((x-1, y), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    elif action == 'East':
      successors.append(((x+1, y), 1-self.noise))
      successors.append((transitions['West'], self.noise / 3))
      successors.append((transitions['East'], self.noise / 3))
      successors.append((transitions['South'], self.noise / 3))
    return successors

  def getReward(self, state, action, nextState):
    if state == 'TERMINAL_STATE':
      return 0.0    
    if state in self.invaderPos:
      return 1000
    else:
      return 10 - self.agent.getMazeDistance(state, self.nearestInvader)

  def isTerminal(self, state):
    return state == 'TERMINAL_STATE'



class MyDefensiveAgent(DummyAgent):
  # The defencive agent
  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.target = None
    self.eatenFoodList = []

  # get a list of pos connecting the red and blue grid
  def getBoundaries(self, gameState):
    boundaries = []
    for y in range(self.height):
      if not gameState.hasWall(self.middle_X - 1, y) and not gameState.hasWall(self.middle_X, y):
        if self.red:
          boundaries.append((self.middle_X - 1, y))
        else:
          boundaries.append((self.middle_X, y))
    return boundaries
  
  # find all the points in own grid of which manhattanDistance <= 3 with all points of boundaries as defensive area to catch invaders
  def setDefensiveArea(self, gameState):
    defensiveArea = set()
    if self.red:
      for x in range(0, self.middle_X - 1):
        for y in range(0, self.height):
          for point in self.boundaries:
            # the points cannot be walls
            if util.manhattanDistance((x, y), point) < SIGHT_RANGE - 1 and not gameState.hasWall(x, y):
              defensiveArea.add((x, y))
    else:
      for x in range(self.middle_X, self.width):
        for y in range(0, self.height):
          for point in self.boundaries:
            if util.manhattanDistance((x, y), point) < SIGHT_RANGE - 1 and not gameState.hasWall(x, y):
              defensiveArea.add((x, y))
    return list(defensiveArea)
  
  # if agent cannot observe invaders in defensive area, it should know their approximate positions through the postions of last eaten food
  def getLastEatenfood(self, gameState):
    # observationHistory records last gameState and current gameState
    if len(self.observationHistory) > 1:
      lastState = self.getPreviousObservation()
      prevFoodList = self.getFoodYouAreDefending(lastState).asList()
      currentFoodList = self.getFoodYouAreDefending(gameState).asList()
      # compare two foodList, find the food last eaten
      if len(prevFoodList) != len(currentFoodList):
        for food in prevFoodList:
          if food not in currentFoodList:
            return food
      else:
        # no food is eaten
        return None

  # if find search lastEatenfood may cause the escape of pacman, defend in escapePoint instead of searching
  def escapePoint(self, gameState, point):
    distances = [self.getMazeDistance(point, boundary) for boundary in self.boundaries]
    escapeDistance = min(distances)
    escapePoint = [b for b, d in zip(self.boundaries, distances) if d == escapeDistance][0]
    return escapePoint, escapeDistance

  def goalRecognitionForFood(self, gameState, eatenFoodList, myPos):
    currentFoodList = self.getFoodYouAreDefending(gameState).asList()
    distances = [self.getMazeDistance(eatenFoodList[-1], food) for food in currentFoodList]
    goals = [food for food, d in zip(currentFoodList, distances) if d == min(distances)]
    dis_to_me = [self.getMazeDistance(myPos, goal) for goal in goals]
    for goal, dist in zip(goals, dis_to_me):
      if dist == min(dis_to_me):
        return goal, dist
    
  def goalRecognition(self, gameState, eatenFoodList, myPos, observedInvaders):
    enermyPos = observedInvaders[0].getPosition()
    goalFood, dist = self.goalRecognitionForFood(gameState, eatenFoodList, myPos)
    escapePoint, escapeDistance = self.escapePoint(gameState, enermyPos)
    capsules = self.getCapsulesYouAreDefending(gameState)
    if len(capsules) != 0:
      distances = [self.getMazeDistance(enermyPos, capsule) for capsule in capsules]
      dis_to_capsule = min(distances)
      for capsule, d in zip(capsules, distances):
        if d == dis_to_capsule:
          goalCapsule = capsule
    else:
      goalCapsule = None
      dis_to_capsule = float('inf')
    return goalFood, dist, escapePoint, escapeDistance, goalCapsule, dis_to_capsule


  def chooseAction(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPos = gameState.getAgentPosition(self.index)
    escapePoint = None
    avoidEscapeDistance = 0
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman]
    if len(invaders) == 0:
      self.eatenFoodList.clear()
    observedEnermies = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    observedInvaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    defensiveArea = self.setDefensiveArea(gameState)
    lastEatenFood = self.getLastEatenfood(gameState)
    if lastEatenFood != None:
      self.eatenFoodList.append(lastEatenFood)
    steps = [] # a path (actions) to target
    if (len(invaders) == 0 and len(observedEnermies) == 0) or (len(invaders) != 0 and len(observedInvaders) == 0 and len(self.eatenFoodList) == 0 and lastEatenFood == None):
      target = random.choice(defensiveArea)
      problem = PositionSearchProblem(gameState, self, self.index, goal = target)
      steps = self.aStarSearch(problem, self.SearchHeuristic)
      if len(steps) != 0:
        # the first action is chosen
        return steps[0]
      else:
        # if no path, stop for nextState
        return Directions.STOP
    # if len(invaders) == 0 and len(observedEnermies) > 0:
    #   return Directions.STOP
    if len(invaders) != 0:
      if len(observedInvaders) == 0 and len(self.eatenFoodList) != 0:
        goal = self.goalRecognitionForFood(gameState, self.eatenFoodList, myPos)[0]
        problem = PositionSearchProblem(gameState, self, self.index, goal)
        steps = self.aStarSearch(problem, self.SearchHeuristic)
        if len(steps) != 0:
          return steps[0]
        else:
          return Directions.STOP
      if len(observedInvaders) != 0 and myState.scaredTimer == 0:
        if len(self.eatenFoodList) == 0:
          problem = CatchInvaders(gameState, self, self.index)
          steps = self.aStarSearch(problem, self.SearchHeuristic)
          if len(steps) != 0:
            return steps[0]
          else:
            return Directions.STOP
        else:
          goalFood, dist, escapePoint, escapeDistance, goalCapsule, dis_to_capsule = self.goalRecognition(gameState, self.eatenFoodList, myPos, observedInvaders)
          avoidEscapeDistance = self.getMazeDistance(myPos, escapePoint)
          if goalCapsule != None:
            avoidCapsule = self.getMazeDistance(myPos, goalCapsule)
          else:
            avoidCapsule = 0
          if escapeDistance > avoidEscapeDistance and dis_to_capsule > avoidCapsule:
            # catch observable invaders if my agent isn't scared, assume only one invader
            problem = CatchInvaders(gameState, self, self.index)
            steps = self.aStarSearch(problem, self.SearchHeuristic)
            if len(steps) != 0:
              return steps[0]
            else:
              return Directions.STOP
          elif escapeDistance == avoidEscapeDistance:
            problem = PositionSearchProblem(gameState, self, self.index, goal = escapePoint)
            steps = self.aStarSearch(problem, self.SearchHeuristic)
            if len(steps) != 0:
              return steps[0]
            else:
              return Directions.STOP
          elif dis_to_capsule == avoidCapsule:
            problem = PositionSearchProblem(gameState, self, self.index, goal = goalCapsule)
            steps = self.aStarSearch(problem, self.SearchHeuristic)
            if len(steps) != 0:
              return steps[0]
            else:
              return Directions.STOP
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def nullHeuristic(self, state, problem = None):    
      """
      A heuristic function estimates the cost from the current state to the nearest    
      goal in the provided SearchProblem.  This heuristic is trivial.    
      """
      return 0

  def aStarSearch(self, problem, heuristic = nullHeuristic):
    PQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, '', 0, [])
    PQ.push(startNode,heuristic(startState, problem))
    visited = set()
    best_g = dict()
    while not PQ.isEmpty():
      node = PQ.pop()
      state, action, cost, path = node
      if (not state in visited) or cost < best_g.get(state):
        visited.add(state)
        best_g[state] = cost
        if problem.isGoalState(state):
          path = path + [(state, action)]
          actions = [action[1] for action in path]
          del actions[0]
          return actions
        for succ in problem.getSuccessors(state):
          succState, succAction, succCost = succ
          newNode = (succState, succAction, cost + succCost, path + [(node, action)])
          PQ.push(newNode, heuristic(succState, problem) + cost + succCost)
    return []

  def SearchHeuristic(self, state, problem):
    x, y = state
    heuristic = 0
    distance = 0
    # see mazeDistance as heuristic
    distance = self.getMazeDistance(state, problem.goal)
    if isinstance(problem, CatchInvaders) and problem.food[x][y]:
      heuristic += 100
    # if distance = 1, catch it
    if distance == 1:
      return 99999
    return heuristic

class PositionSearchProblem:
    """
    This search problem can be used to find paths to a particular point on the pacman board.
    The state space consists of (x,y) positions in a pacman game.
    """

    def __init__(self, gameState, agent, index = 0, goal = (1,1), costFn = lambda x: 1):
        """
        gameState: A GameState object
        goal: A position in the gameState
        """
        self.food = agent.getFoodYouAreDefending(gameState)
        self.walls = gameState.getWalls()
        self.middle_X = gameState.data.layout.width / 2
        self.startState = gameState.getAgentPosition(index)
        self.costFn = costFn
        self.goal = goal
        self.agent = agent
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return state == self.goal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
        """
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if self.agent.red:
              if not self.walls[nextx][nexty] and nextx < self.middle_X:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))
            else:
              if not self.walls[nextx][nexty] and nextx >= self.middle_X:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class CatchInvaders(PositionSearchProblem):
  """
  The goal is to catch observable invaders
  """
  def __init__(self, gameState, agent, index = 0, goal = (1,1)):
    # Store the food for later reference
    self.food = agent.getFoodYouAreDefending(gameState)
    self.capsule = agent.getCapsulesYouAreDefending(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.middle_X = gameState.data.layout.width / 2
    self.walls = gameState.getWalls()
    self.startState = gameState.getAgentPosition(index)
    self.costFn = lambda x: 1
    self.agent = agent
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.enemies = [gameState.getAgentState(index) for index in agent.getOpponents(gameState)]
    self.invaders = [a for a in self.enemies if a.isPacman and a.getPosition() != None]
    if len(self.invaders) > 0:
      # assume only one invader
      self.goal = [invader.getPosition() for invader in self.invaders][0]
    else:
      self.goal = None

  def isGoalState(self, state):
    return state == self.goal
  
class SearchLastEatenFoodProblem(PositionSearchProblem):
  """
  The goal is to look for last eatenfood to get the approximate location of unobservable invaders
  """
  def __init__(self, gameState, agent, index = 0, goal = (1,1)):
    # Store the food for later reference
    self.food = agent.getFoodYouAreDefending(gameState)
    self.capsule = agent.getCapsulesYouAreDefending(gameState)
    # Store info for the PositionSearchProblem (no need to change this)
    self.middle_X = gameState.data.layout.width / 2
    self.walls = gameState.getWalls()
    self.startState = gameState.getAgentPosition(index)
    self.costFn = lambda x: 1
    self.agent = agent
    self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
    self.goal = goal
  
  def isGoalState(self, state):
    return state == self.goal

