'''
Implementation of a basic RL environment.

Rewards are all normal.
Transitions are multinomial.

'''

import numpy as np

#-------------------------------------------------------------------------------


class Environment(object):
    '''General RL environment'''

    def __init__(self):
        pass

    def reset(self):
        pass

    def get_states(self):
        '''
        Return current timestep, state

        '''

        return 0, 0

    def advance(self, action):
        '''
        Moves one step in the environment.

        Args:
            action

        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        return 0, 0, 0


#-------------------------------------------------------------------------------


class TabularMDP(Environment):
    '''
    Tabular MDP

    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    '''

    def __init__(self, nState, nAction, epLen):
        '''
        Initialize a tabular episodic MDP

        Args:
            nState  - int - number of states
            nAction - int - number of actions
            epLen   - int - episode length

        Returns:
            Environment object
        '''

        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen

        self.timestep = 0
        self.state = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        for state in range(nState):
            for action in range(nAction):
                self.R[state, action] = (1, 1)
                self.P[state, action] = np.ones(nState) / nState


    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = 0

    def get_states(self):
        '''Return current timestep, state'''

        return self.timestep, self.state


    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action

        Returns:
        reward - double - reward
        newState - int - new state
        pContinue - 0/1 - flag for end of the episode
        '''
        if self.R[self.state, action][1] < 1e-9:
            # Hack for no noise
            reward = self.R[self.state, action][0]
        else:
            reward = self.R[self.state, action][0]
        newState = np.random.choice(self.nState, p=self.P[self.state, action])

        # Update the environment
        self.state = newState
        self.timestep += 1

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1

        return reward, newState, pContinue

    def compute_optVals(self):
        '''
        Compute optimal Q and V values of the environment by value iteation

        Args:
            NULL - works on the TabularMDP

        Returns:
            qVals - qVals[state, timestep] is vector of (optimal) Q values for each action
            vVals - vVals[timestep] is the vector of (optimal) V values at timestep
        '''
        qVals = {}
        vVals = {}

        vVals[self.epLen] = np.zeros(self.nState)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            vVals[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    qVals[s, j][a] = self.R[s, a][0] + np.dot(self.P[s, a], vVals[j + 1])

                vVals[j][s] = np.max(qVals[s, j])
        return qVals, vVals


#-------------------------------------------------------------------------------
# Benchmark environments


def make_riverSwim(epLen=20, nState=6):
    '''
    Makes the benchmark RiverSwim MDP
    See Page 6 in https://arxiv.org/pdf/1306.0940.pdf

    Args:
        NULL - works for default implementation

    Returns:
        riverSwim - Tabular MDP environment
    '''
    nAction = 2 # 0-left,  1-right
    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (5. / 1000, 0)
    R_true[nState - 1, 1] = (1, 0)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1. # left action always succeed

    for s in range(1, nState - 1):
        P_true[s, 1][min(nState - 1, s + 1)] = 0.35
        P_true[s, 1][s] = 0.6
        P_true[s, 1][max(0, s-1)] = 0.05

    P_true[0, 1][0] = 0.4
    P_true[0, 1][1] = 0.6
    P_true[nState - 1, 1][nState - 1] = 0.6
    P_true[nState - 1, 1][nState - 2] = 0.4

    riverSwim = TabularMDP(nState, nAction, epLen)
    riverSwim.R = R_true
    riverSwim.P = P_true
    riverSwim.reset()

    return riverSwim
