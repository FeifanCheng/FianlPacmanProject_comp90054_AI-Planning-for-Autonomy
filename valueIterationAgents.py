import util
import collections

class ValueIterationAgent():
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        
        states = self.mdp.getStates()
        for i in range(self.iterations):
            new_values = self.values.copy()
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                actions = self.mdp.getPossibleActions(state)
                q_values = util.Counter()
                for action in actions:
                    q_values[action] = self.computeQValueFromValues(state, action) 
                new_values[state] = q_values[q_values.argMax()]
            self.values = new_values.copy()
    
    # Return the value of the state
    def getValue(self, state):
        return self.values[state]


    # ompute the Q-value of action in state from the value function stored in self.values.
    def computeQValueFromValues(self, state, action):     
        q = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return q

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        q_values = util.Counter()
        for action in actions:
            q_values[action] = self.computeQValueFromValues(state, action)
        return q_values.argMax()

    # Returns the policy at the state
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state"
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)