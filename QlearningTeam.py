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


class MyOffensiveAgent(CaptureAgent):

    def __init__(self, *args, **options):
        CaptureAgent.__init__(self, *args)
        if options.get('numTraining') is None:
            self.numTraining = 0
        else:
            self.numTraining = int(options['numTraining'])


    def registerInitialState(self, state):
        """
        get the register from captureagent and print the number of training
        """
        CaptureAgent.registerInitialState(self, state)
        self.episodesSoFar = 0
        self.total_train_reward = 0.0
        self.total_test_reward = 0.0
        self.epsilon = 0.05
        self.alpha = 0.2
        self.discount = 0.9
        # list to store the previous action
        self.store_previous_action = []
        # record the last state
        self.Last_state = None
        # record the last action
        self.Last_action = None
        # create the Qvalue to store the Q value in different state and action
        self.Qvalue = util.Counter()
        width_red, width_blue = self.getmiddleposition(state)
        # Store the coordinates of the entry point
        self.entries = self.entry(state, width_blue)
        # Store dead end coordinates
        self.deadEnds = self.wall(state, width_red)
        # Record the last point of ghost
        self.closestGhost = None
        self.bestEntry = None
        self.episodeRewards = 0.0
        if self.episodesSoFar == 0:
            print((self.numTraining)," times training")

    """
    Q-learning
    """

    def update(self, state, action, nextState, reward):
        """
        Q-Value update
        """
        interest_discount = self.discount * self.computeValueFromQValues(nextState)
        estimate = reward + interest_discount
        rate_learning = self.alpha * estimate
        interest = (1 - self.alpha) * self.getQValue(state, action)
        self.Qvalue[(state, action)] += interest + rate_learning


    def getQValue(self, state, action):
        """
        Computes the Qvalue
        """
        features = self.getFeatures(state, action)
        weights = self.getWeights()
        self.Qvalue[(state, action)] = features * weights
        return features * weights

    def computeValueFromQValues(self, state):
        """
        get the max Qvalue in this state
        """
        List = []
        legal_actions = state.getLegalActions(self.index)
        num_legal = len(legal_actions)
        if num_legal == 0:
            return 0.0
        for legal in legal_actions:
            Qvalue = self.getQValue(state, legal)
            List = List + [Qvalue]
        max_value = max(List)
        return max_value

    def computeActionFromQValues(self, state):
        """
        Compute and get the best action to take in a state.
        """
        List = []
        max_value = self.computeValueFromQValues(state)
        # get the list of legal action
        legal_action = state.getLegalActions(self.index)
        # get the number of legal action
        num_legal = len(legal_action)
        if num_legal == 0:
            return None
        for legal in legal_action:
            Qvalue = self.Qvalue[(state, legal)]
            if max_value == Qvalue:
                List = List + [legal]
        get_action = random.choice(List)
        return get_action

    def getPolicy(self, state):
        """
        return the action from the choose
        """
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        """
        return the max Q value
        """
        return self.computeValueFromQValues(state)

    def getSuccessor(self, state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = state.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getMyState(self, state):
        """
        get the state of mine
        """
        Mystate = state.getAgentState(self.index)
        return Mystate

    def getMyPosition(self, state):
        """
        get the position of mine
        """
        Myposition = self.getMyState(state).getPosition()
        return Myposition

    def getEnemyState(self, state):
        """
        get the list of enemy state
        """
        enemy = []
        opponentState = self.getOpponents(state)
        for i in opponentState:
            Enemystate = state.getAgentState(i)
            enemy = enemy + [Enemystate]
        return enemy

    def getreward(self, state):
        """
        get the value of reward
        """
        start_postion = state.getAgentState(self.index).getPosition()
        reward = state.getScore()
        current_state = self.getMyState(state)
        current_position = self.getMyPosition(state)
        last_state = self.getMyState(self.Last_state)
        last_position = self.getMyPosition(self.Last_state)
        if not current_state.isPacman:
            reward = reward - 60
        # Return to the origin, the situation of being eaten
        if last_state.isPacman:
            if current_position == start_postion:
                if not current_state.isPacman:
                    reward = reward - 120
        # Walk in place
        if current_position == last_position:
            reward = reward - 3
        # Have eaten food and did not return to the origin
        if last_state.numCarrying > current_state.numCarrying:
            if current_position != start_postion:
                reward = reward + 70
        return reward

    def observationFunction(self, state):
        """
        observation the result
        """
        if self.Last_state is not None:
            self.episodeRewards = self.episodeRewards + self.getreward(state)
            self.update(self.Last_state, self.Last_action, state, self.getreward(state))
        return state.makeObservation(self.index)

    def stopepisode(self):
        """
        stopEpisode
        """
        if self.episodesSoFar >= self.numTraining:
            self.total_test_reward = self.total_test_reward + self.episodeRewards
        else:
            self.total_train_reward = self.total_train_reward + self.episodeRewards
        self.episodesSoFar = self.episodesSoFar + 1
        if self.episodesSoFar >= self.numTraining:
            self.resetalpha()
            self.resetepsilon()

    def resetepsilon(self):
        self.epsilon = 0.0

    def resetalpha(self):
        self.alpha = 0.0

    def final(self, state):
        """
        Call at the end of each game
        """
        # get the reward
        reward = self.getreward(state)
        self.update(self.Last_state, self.Last_action, state, reward)
        self.episodeRewards = self.episodeRewards + reward
        self.stopepisode()
        if self.episodesSoFar == self.numTraining:
            msg = 'Training finished (reset epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))
            print(self.getWeights(state))

    def judgeOppositeDirction(self, List, a):
        """
        judge the opposite dirction
        """
        if List[a] == "West" and List[a - 1] == "East":
            return True
        if List[a] == "East" and List[a - 1] == "West":
            return True
        if List[a] == "South" and List[a - 1] == "North":
            return True
        if List[a] == "North" and List[a - 1] == "South":
            return True
        else:
            return False

    def chooseAction(self, state):
        """
        pick one legal action and check if there are legal action
        """
        legalActions = state.getLegalActions(self.index)
        num_legal = len(legalActions)
        reward = self.getScore(state)
        if num_legal == 0:
            return None
        # check the loop action in the game
        loopAction = None
        # check if the list store previous actions larger than four
        num_previous = len(self.store_previous_action)
        # In the case of more than four actions stored in the table
        if num_previous >= 4:
            # Because it is a round trip, to judge whether the pacman is moving in place,
            # it is to judge the different situations of the previous four actions,
            # and the separate actions are in the same direction
            if self.store_previous_action[-1] == self.store_previous_action[-3]:
                if self.store_previous_action[-2] == self.store_previous_action[-4]:
                    # Adjacent actions are opposite
                    if self.judgeOppositeDirction(self.store_previous_action, -1):
                        if self.judgeOppositeDirction(self.store_previous_action, -3):
                            loopAction = self.store_previous_action[-4]
                            self.store_previous_action = []
        # Determine if loopaction is None
        if loopAction != None:
            # if loop action is in legal, remove it, and choose one legal action
            if loopAction in legalActions:
                legalActions.remove(loopAction)
                return random.choice(legalActions)
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        # use the result of Qlearning choose one action

        check = util.flipCoin(self.epsilon)
        if check == True:
            choose_action = random.choice(legalActions)
        else:
            # use qlearning function
            choose_action = self.computeActionFromQValues(state)
        # update last_state, last_action and the list which store previous action
        self.Last_state = state
        self.Last_action = choose_action
        self.store_previous_action = self.store_previous_action + [choose_action]
        return choose_action

    def wall(self, state, X):
        """
        Store the coordinates of a dead end with walls on three sides
        """
        layout = state.data.layout
        width = layout.width
        height = layout.height
        wallend = []
        # Get the starting abscissa of the active area
        if self.red:
            boundary_meddle = X
            boundary_side = width
        else:
            boundary_meddle = 0
            boundary_side = X
        while True:
            # Update the layout of the new wall
            Wall = layout.walls
            num_endPlace = len(wallend)
            # Perform a prescribed search on the scope of the offense
            for y in range(0, height):
                for x in range(boundary_meddle, boundary_side):
                    # Judge whether all points are walls
                    if layout.walls[x][y] == False:
                        # Determine whether this point is a wall on three sides or a dead end,
                        # and add this point to the dead end list
                        if self.wallendPlace(x, y, state):
                            Wall[x][y] = True
                            wallend = wallend + [(x, y)]
            layout.walls = Wall
            # If there is no dead end point in this cycle, break
            if num_endPlace == len(wallend):
                break
        return wallend

    def entry(self, state, X):
        """
        get the list of entry place
        """
        entries = []
        height = state.data.layout.height
        for Y in range(0, height):
            if state.data.layout.isWall((X, Y)) == False:
                entries = entries + [(X, Y)]
        return entries

    def getmiddleposition(self, state):
        """
        get the middle position in different side(red or blue team)
        """
        width = state.data.layout.width
        if self.red:
            width_red = int(width / 2)
            width_blue = int(width / 2) - 1
        else:
            width_red = int(width / 2) - 1
            width_blue = int(width / 2)
        return (width_red, width_blue)

    def getWeights(self):
        return {
            'Defensive': -2,
            'Offense': 1,
            'dead-back': -10000,
            'Stop': -100,
            'under-attack': 0,
            'food-eaten': 300,
            'invincible-capsule': 350,
            'Keep-eating': -3,
            'near-capsule': -5,
            'back-side': -2,
            'carry-food': 400,
            'dead-end': -1000,
            'distance-enemy': 5,
            'distance-entry': -5,
        }


    def wallendPlace(self, x, y, state):
        """
        judge the dead end place
        """
        walls = state.data.layout.walls
        num_walls = 0
        judge = False
        if walls[x][y+1]:
            num_walls = num_walls + 2
        if walls[x][y-1]:
            num_walls = num_walls + 2
        if walls[x+1][y]:
            num_walls = num_walls + 2
        if walls[x-1][y]:
            num_walls = num_walls + 2
        if num_walls < 6:
            judge = False
        else:
            judge = True
        return judge


    def getFeatures(self, state, action):
        """
        get features
        """
        features = util.Counter()
        # get the position of start position
        startPosition = state.getAgentState(self.index).getPosition()
        # get the successor place
        successor = self.getSuccessor(state, action)
        successor_state = self.getMyState(successor)
        successor_position = successor_state.getPosition()
        # get the current place
        current_state = self.getMyState(state)
        current_position = current_state.getPosition()
        dangerous = False
        # get the capsules
        capsules = self.getCapsules(state)
        num_capsules = len(capsules)
        # get the next point coordinates
        position = (int(successor_position[0]), int(successor_position[1]))
        # get the information of layout
        width_wall = state.getWalls().width
        height_wall = state.getWalls().height
        area = width_wall * height_wall
        # get the information of foods in the game
        foods = self.getFood(state).asList()
        num_foods = len(foods)
        judge_food = self.getFood(state)[int(successor_position[0])][int(successor_position[1])]
        # get ghost state
        enemies = self.getEnemyState(state)
        # situation the pacman don't move
        if action == Directions.STOP:
            features['Stop'] = 1
        # the list store the closest ghost
        ghosts = []
        exampledist = 99999
        # Determine if it is ghost
        for enemy in enemies:
            if enemy.isPacman == False:
                # remaining invincible time
                if enemy.scaredTimer < 3:
                    if enemy.getPosition() != None:
                        ghosts = ghosts + [enemy]
        num_ghost = len(ghosts)
        # If there are ghosts nearby, update the self.closestGhost
        if num_ghost > 0:
            for close in ghosts:
                postion = close.getPosition()
                Mazedist = self.getMazeDistance(successor_position, postion)
                if Mazedist <= 5:
                    # Update the latest ghost in self.closestGhost
                    if exampledist >= Mazedist:
                        self.closestGhost = close
                        exampledist = Mazedist
        # The next step is in his own half, just didn't attack
        if successor_state.isPacman == False:
            choose_example = -99999
            for entry in self.entries:
                # if there is ghost nearby, record the distance
                if self.closestGhost is not None:
                    # get the distance to ghost
                    ghost_position = self.closestGhost.getPosition()
                    ghostdistance = self.getMazeDistance(ghost_position, entry)
                else:
                    ghostdistance = 0
                # if there are foods
                if num_foods > 0:
                    # find the shortest distance to food
                    fooddistance = []
                    for food in foods:
                        food_distance = self.getMazeDistance(entry, food)
                        fooddistance = fooddistance + [food_distance]
                    # calculate whether the food is in or near the ghost
                    escape_eat = ghostdistance - min(fooddistance)
                # no foods situation
                else:
                    escape_eat = ghostdistance
                # if it comes to food
                if escape_eat > choose_example:
                    choose_example = escape_eat
                    self.bestEntry = entry
            successor_entry = self.getMazeDistance(successor_position, self.bestEntry)
            features['distance-entry'] = float(successor_entry)
            # if there is the nearest ghost
            if self.closestGhost is not None:
                ghost_position = self.closestGhost.getPosition()
                current_position = current_state.getPosition()
                # real-time distance to ghost
                current_ghost = self.getMazeDistance(ghost_position, current_position)
                if current_ghost <= 2:
                    dangerous = True
            # determine if I become a pacman in the opponent's half
            if current_state.isPacman is True:
                # the next point is at the starting point, eaten by ghosts
                if successor_position == startPosition:
                    features['dead-back'] = area
                # the next point is not at the starting point, not eaten by ghosts
                else:
                    # no food carry
                    if current_state.numCarrying == 0:
                        # it’s not dangerous at this time, the ghost is no longer within two feet
                        if dangerous == False:
                            features['Defensive'] = area
                    # food carry
                    if current_state.numCarrying > 0:
                        features['carry-food'] = area
        # if on the opponent's side
        else:
            # if the current point is not pacman, come back from the opposite half this time
            if current_state.isPacman == False:
                current_entry = self.getMazeDistance(self.bestEntry, current_position)
                # the distance from the current point to the entrance is less than 1, start the attack
                if current_entry <= 1:
                    features['Offense'] = area

            # only consider nearby enemies
            closestGhost = None
            # store enemy locations in ghosts
            ghosts = []
            for enemy in enemies:
                # determine the enemy is not Pacman, and can get to the enemy position
                if enemy.isPacman == False:
                    if enemy.getPosition() != None:
                        ghosts = ghosts + [enemy]
            # number of enemies
            num_ghost = len(ghosts)
            # can get the enemy
            if num_ghost > 0:
                exampledist = 99999
                for enemy in ghosts:
                    # get the distance between where I am going next and where the enemy is
                    Mazedist = self.getMazeDistance(successor_position, enemy.getPosition())
                    # if the next position and the enemy position are less than 5,
                    # update the value of the closestghost
                    if Mazedist <= 5:
                        if Mazedist < exampledist:
                            closestGhost = enemy
                            exampledist = Mazedist
            # when there are ghost nearby
            if closestGhost is not None:
                if current_state.numCarrying == 0:
                    ghost_position = closestGhost.getPosition()
                    ghost_tosuccessor = self.getMazeDistance(successor_position, ghost_position)
                    # when the nearest enemy’s position and my next position are larger than 5,
                    # eat food
                    if ghost_tosuccessor > 1:
                        features['Keep-eating'] = float(ghost_tosuccessor)
                # the remaining scared time is less than 5
                if closestGhost.scaredTimer <= 5:
                    # distance is less than 5
                    ghost_position = closestGhost.getPosition()
                    ghost_tosuccessor = self.getMazeDistance(successor_position, ghost_position)
                    # when the nearest enemy’s position and my next position are less than or equal to 5,
                    # it is dangerous
                    if ghost_tosuccessor <= 5:
                        ghost_successor = self.getMazeDistance(successor_position, ghost_position)
                        features['under-attack'] = area
                        features['distance-enemy'] = float(ghost_successor)
                        # the distance is less than 1, was eaten
                        if ghost_successor <= 1:
                            features['dead-back'] = area
                        # enter the dead end
                        if position in self.deadEnds:
                            features['dead-end'] = area
                            # turn around and be eaten quickly so it can move to the ghost
                            # and try to escape instead of hiding in a dead end
                            features['distance-enemy'] = -features['distance-enemy']
                # find the nearest entry point
                current_entry = []
                for entry in self.entries:
                    successor_entry = self.getMazeDistance(successor_position, entry)
                    current_entry = current_entry + [successor_entry]
                # shortest time to entrance
                current_entries = min(current_entry)
                # determine if there are capsules on the field
                if num_capsules > 0:
                    # Give a judgment that the difference
                    # between the number of remaining steps
                    # and the number of steps to the entry point is within 10
                    if int(state.data.timeleft) / 4 - 13 >= current_entries:
                        # if the next step is to eat a capsule
                        if successor_position in capsules:
                            features['invincible-capsule'] = area
                        # in most cases, do not take capsules in the next step
                        else:
                            # the location of the capsule on the storage field
                            distance_current_capsules = []
                            # select the nearest capsule
                            choose_capsules = []
                            # find the capsule with the shortest distance so far
                            for capsule in capsules:
                                successor_capsule = self.getMazeDistance(successor_position, capsule)
                                distance_current_capsules = distance_current_capsules + [successor_capsule]
                            distance_capsule = min(distance_current_capsules)
                            # select the coordinates of a capsule
                            for capsule in capsules:
                                successor_capsule = self.getMazeDistance(successor_position, capsule)
                                if successor_capsule == distance_capsule:
                                    choose_capsules = choose_capsules + [capsule]
                            capsule_position = random.choice(choose_capsules)
                            features['near-capsule'] = float(distance_capsule)
                            # determine if the previous position is empty
                            if self.Last_state:
                                # Get the last point coordinate saved in chooseaction function
                                last_state = self.getMyState(self.Last_state)
                                last_position = last_state.getPosition()
                                last_capsule = self.getMazeDistance(last_position, capsule_position)
                                # Get ghost coordinates
                                ghost_position = closestGhost.getPosition()
                                ghost_capsule = self.getMazeDistance(ghost_position, capsule_position)
                                # Don’t worry about dead ends if you get the capsules
                                # Determine when the distance between the ghost
                                # and the capsule is greater than the distance between the ghost and the capsule
                                if ghost_capsule > distance_capsule:
                                    # In the process of approaching the capsule
                                    if last_capsule > distance_capsule:
                                        # in a dead end
                                        if position in self.deadEnds:
                                            # If the capsule is in a dead end
                                            if capsule_position in self.deadEnds:
                                                # Can be close to the capsule
                                                features['dead-end'] = 0.0
                                                features['distance-enemy'] = -features['distance-enemy']
                # no capsule on the field
                else:
                    current_back = self.getMazeDistance(startPosition, successor_position)
                    features['back-side'] = float(current_back + 10)
            # real-time situation, no ghosts nearby
            else:
                current_entry = self.getMazeDistance(self.bestEntry, current_position)
                if current_entry <= 5:
                    if current_state.numCarrying > 2:
                        current_back = self.getMazeDistance(startPosition, successor_position)
                        features['back-side'] = float(current_back + 10)
                if current_entry <= 7:
                    if current_state.numCarrying > 4:
                        current_back = self.getMazeDistance(startPosition, successor_position)
                        features['back-side'] = float(current_back + 10)
                if current_state.numCarrying > 6:
                    current_back = self.getMazeDistance(startPosition, successor_position)
                    features['back-side'] = float(current_back + 10)
                # have encountered ghosts nearby before
                if self.closestGhost is not None:
                    # Record the distance between the ghost point and the next point before
                    ghost_lastpostion = self.closestGhost.getPosition()
                    successor_ghost = self.getMazeDistance(successor_position, ghost_lastpostion)
                    # If the distance to ghost is less than 5, reset the value of the nearest ghost to None
                    if successor_ghost <= 5:
                        self.closestGhost = None
            # in the case of being eaten
            if features['distance-enemy'] == 0.0:
                # Because the previous setting is calculated within five steps,
                # now use a number greater than 5 to calculate
                features['distance-enemy'] = float(8)
            # there are capsules on the field, and the next step is to eat him
            if num_capsules > 0:
                if successor_position in capsules:
                    features['invincible-capsule'] = area
            # the situation is not dangerous
            if features['under-attack'] != 1:
                # get the shortest distance home
                current_entry = []
                for entry in self.entries:
                    successor_entry = self.getMazeDistance(successor_position, entry)
                    current_entry = current_entry + [successor_entry]
                # shortest time to entrance
                current_entries = min(current_entry)
                # when the food on the field is less than 2 or the difference between the remaining steps
                # and the steps to the entrance is less than 10, go home and prepare
                if int(state.data.timeleft) / 4 - 10 < current_entries:
                    current_back = self.getMazeDistance(startPosition, successor_position)
                    features['back-side'] = float(current_back)
                elif num_foods <= 2:
                    current_back = self.getMazeDistance(startPosition, successor_position)
                    features['back-side'] = float(current_back)
                # if the number of food carry is less than 5, back to side
                elif current_state.numCarrying >= 5:
                    current_back = self.getMazeDistance(startPosition, successor_position)
                    features['back-side'] = float(current_back)
                # continue to prepare for eating
                else:
                    back_eat = -99999
                    choose_food = None
                    # determine whether the food on the field is greater than 0,
                    # compare the weight to go home or continue to eat food
                    if num_foods > 0:
                        for food in foods:
                            # record the distance between the food at this time and the next point
                            current_food = self.getMazeDistance(successor_position, food)
                            # determine whether the nearest close of the record is empty,
                            # and set the distance to zero
                            if self.closestGhost is not None:
                                ghost_position = self.closestGhost.getPosition()
                                current_ghost = self.getMazeDistance(food, ghost_position)
                            # get distance when there is ghost nearby
                            else:
                                current_ghost = 0
                            # calculate the difference between the distance between the food
                            # and the ghost and the distance to the next point.
                            # a positive number means that it is closer to my next point.
                            calculate_ghost_food = current_ghost - current_food
                            if back_eat < calculate_ghost_food:
                                back_eat = calculate_ghost_food
                                # when the food is closer to me,
                                # record the choice of food at this time
                                choose_food = food
                    if choose_food is None:
                        current_food = None
                    else:
                        # record the distance from the next point to the food at this time
                        current_food = self.getMazeDistance(successor_position, choose_food)
                    # choose one food to eat
                    if choose_food is not None:
                        # if the food carried is less than half of the food in the scene
                        if current_state.numCarrying < int(num_foods / 2):
                            # if the distance to the food is shorter than the distance to the entrance
                            # or there is no food on the body at the next step, prepare to continue eating food
                            if current_food < current_entries or successor_state.numCarrying == 0:
                                features['Keep-eating'] = float(current_food)
                            if current_state.numCarrying > 4:
                                features['back-side'] = float(current_entries)
                    # at this time, no food is chosen, it is close to the border
                    else:
                        features['back-side'] = float(current_entries)
                    if judge_food == True:
                        features['food-eaten'] = area
        features.divideAll(area)
        return features


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

    # find all the points in own grid of which manhattanDistance <= 5 with all points of boundaries as defensive area to catch invaders
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
    def escapePoint(self, gameState, lastEatenFood):
        distances = [self.getMazeDistance(lastEatenFood, boundary) for boundary in self.boundaries]
        minDistance = min(distances)
        escapePoint = [b for b, d in zip(self.boundaries, distances) if d == minDistance][0]
        return escapePoint, minDistance

    def chooseAction(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = gameState.getAgentPosition(self.index)
        escapePoint = None
        minDistance = 0
        avoidEscapeDistance = 0
        defensiveArea = self.setDefensiveArea(gameState)
        lastEatenFood = self.getLastEatenfood(gameState)
        if (lastEatenFood != None):
            self.eatenFoodList.append(lastEatenFood)
            escapePoint, minDistance = self.escapePoint(gameState, lastEatenFood)
            avoidEscapeDistance = self.getMazeDistance(myPos, escapePoint)
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman]
        observedInvaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        steps = []  # a path (actions) to target
        if len(invaders) == 0 or (
                len(observedInvaders) == 0 and lastEatenFood == None and len(self.eatenFoodList) == 0):
            # wander in defensiveArea, target is a random point in defensiveArea (PositionSearchProblem)
            target = random.choice(defensiveArea)
            problem = PositionSearchProblem(gameState, self, self.index, goal=target)
            steps = self.aStarSearch(problem, self.SearchHeuristic)
            if len(steps) != 0:
                # the first action is chosen
                return steps[0]
            else:
                # if no path, stop for nextState
                return Directions.STOP
        elif (len(observedInvaders) == 0 and lastEatenFood == None and len(self.eatenFoodList) != 0) or (
                lastEatenFood != None and minDistance > avoidEscapeDistance):
            # coincidently not observe invaders during wandering, so search the last eaten food in list
            problem = SearchLastEatenFoodProblem(gameState, self, self.index, goal=self.eatenFoodList[-1])
            # target / goal is eatenFoodList[-1] (SearchLastEatenFoodProblem)
            steps = self.aStarSearch(problem, self.SearchHeuristic)
            if len(steps) != 0:
                return steps[0]
            else:
                return Directions.STOP
        elif lastEatenFood != None and minDistance <= avoidEscapeDistance:
            # go to escapePoint instead of searching lastEatenFood
            target = escapePoint
            problem = PositionSearchProblem(gameState, self, self.index, goal=target)
            steps = self.aStarSearch(problem, self.SearchHeuristic)
            if len(steps) != 0:
                return steps[0]
            else:
                return Directions.STOP
        if len(observedInvaders) != 0 and myState.scaredTimer == 0:
            # catch observable invaders if my agent isn't scared, assume only one invader
            problem = CatchInvaders(gameState, self, self.index)
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

    def nullHeuristic(self, state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest
        goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0

    def aStarSearch(self, problem, heuristic=nullHeuristic):
        PQ = util.PriorityQueue()
        startState = problem.getStartState()
        startNode = (startState, '', 0, [])
        PQ.push(startNode, heuristic(startState, problem))
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

    def __init__(self, gameState, agent, index=0, goal=(1, 1), costFn=lambda x: 1):
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
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

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
            x, y = state
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
        self._expanded += 1  # DO NOT CHANGE
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
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class CatchInvaders(PositionSearchProblem):
    """
    The goal is to catch observable invaders
    """

    def __init__(self, gameState, agent, index=0, goal=(1, 1)):
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

    def __init__(self, gameState, agent, index=0, goal=(1, 1)):
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





