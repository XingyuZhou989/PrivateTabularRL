'''
Finite Horizon Tabular Agent.

'''

import numpy as np


class Agent(object):

    def __init__(self):
        pass

    def update_obs(self, obs, action, reward, newObs):
        '''Add observation to records'''

    def update_policy(self, h):
        '''Update internal policy based upon records'''

    def pick_action(self, obs):
        '''Select an observation based upon the observation and policy'''


class FiniteHorizonTabularAgent(Agent):
    def __init__(self, nState, nAction, epLen, nEps, **kwargs):
        '''
        Tabular episodic learner for time-homoegenous MDP.

        Args:
            nState - int - number of states
            nAction - int - number of actions
            epLen - int - length of each episode
            nEps - int - number of episodes
        Returns:
            tabular learner, to be inherited from
        '''

        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen
        self.nEps = nEps

        # value functions
        # qVals - qVals[state, timestep] is vector of Q values for each action
        # vVals - vVals[timestep] is the vector of values at timestep
        self.qVals = {}
        self.vVals = {}


        self.R_stats = {}
        self.P_stats = {}

        # internal policy and update rate
        self.pi = {}
        self.pi_old = {}
        self.eta = np.sqrt(2./1000)

        # make the initial statistics
        for state in range(nState):
            for action in range(nAction):
                self.R_stats[state, action] = (1, 1)
                self.P_stats[state, action] = (
                    1 * np.ones(self.nState, dtype=np.float32))

        # random initial policy
        for i in range(self.epLen):
            j = self.epLen - i - 1
            for s in range(self.nState):
                self.pi[s, j] = np.zeros(self.nAction, dtype=np.float64)
                self.pi_old[s, j] = np.zeros(self.nAction, dtype=np.float64)
                for a in range(self.nAction):
                    self.pi[s,j][a] = 1./self.nAction
                    self.pi_old[s,j][a] = 1./self.nAction

    def update_obs(self, oldState, action, reward, newState, pContinue, h):

        '''
        Update the stats based on one transition.
        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1 - episode ending or not
            h - int - time within episode (not used)
        Returns:
            NULL - updates in place
        '''
        oldMean, oldCounts = self.R_stats[oldState, action]

        newCounts = oldCounts + 1
        newMean = (oldMean * oldCounts + reward) / (oldCounts + 1)
        self.R_stats[oldState, action] = (newMean, newCounts)

        if pContinue == 1:
            self.P_stats[oldState, action][newState] += 1


    def compute_estModelBonus(self,ep):
        '''
        Compute estimates and bonus terms

        ep - int - the current episode number

        '''

        R_hat = {}
        P_hat = {}
        R_bonus = {}
        P_bonus = {}
        # hyperparameter
        scaling = 2 * np.log(3)

        for s in range(self.nState):
            for a in range(self.nAction):
                R_hat[s, a] = self.R_stats[s, a][0]
                P_hat[s, a] = self.P_stats[s, a] / np.sum(self.P_stats[s, a])
                R_sum = self.R_stats[s, a][1]
                R_bonus[s, a] = np.sqrt(scaling / R_sum)

                P_sum = self.P_stats[s, a].sum()

                P_bonus[s, a] = np.sqrt(scaling / P_sum)

        return R_hat, P_hat, R_bonus, P_bonus


    def compute_policy(self, R, P, R_bonus, P_bonus, eval = False):
        '''
        Compute the optimal value functions or update the current policy based
        on input eval
        Args:
            R - R[s,a] = estimated mean rewards
            P - P[s,a] = estimated probability vector of transitions
            R_bonus - R_bonus[s,a] = bonus for rewards
            P_bonus - P_bonus[s,a] = bonus for transitions
            eval - False for VI, True for PO
        '''
        qVals = {}
        vVals = {}

        vVals[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            vVals[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = (R[s, a] + R_bonus[s, a]
                                      + np.dot(P[s, a], vVals[j + 1])
                                      + P_bonus[s, a] * i)
                if eval == True:
                    qVals[s,j] = np.clip(qVals[s,j], 0,self.epLen) # truncate
                    vVals[j][s] = np.inner(self.pi[s,j], qVals[s,j])
                    self.pi_old[s,j] = self.pi[s,j]

                    # mirror ascent
                    weight = np.exp(self.eta * qVals[s,j])
                    unnorm_prob = np.multiply(self.pi[s,j],weight)
                    prob = unnorm_prob / unnorm_prob.sum()
                    self.pi[s,j] = prob

                else:
                    # simple value iteration
                    vVals[j][s] = np.max(qVals[s, j])


        self.qVals = qVals


#-----------------------------------------------------------------------------
# UCBVI
#-----------------------------------------------------------------------------
class UCBVI(FiniteHorizonTabularAgent):
    '''UCBVI '''

    def update_policy(self, ep = False, eval = False):
        '''
        update new policy
        '''

        R_hat, P_hat, R_bonus, P_bonus = self.compute_estModelBonus(ep)

        self.compute_policy(R_hat, P_hat, R_bonus, P_bonus, eval)

    def pick_action(self, state, timestep):
        '''
        Greedy with respect to current Q values
        '''

        Q = self.qVals[state, timestep]
        action = np.argmax(Q)
        return action


#-----------------------------------------------------------------------------
# UCBVI-LDP
#-----------------------------------------------------------------------------
class UCBVI_LDP(UCBVI):
    '''UCBVI-LDP '''

    def __init__(self, nState, nAction, epLen, nEps, privEps):
        '''
        As per the tabular learner, but added tunable privEps.

        Args:
            privEps - double - privacy budget
        '''
        super(UCBVI_LDP, self).__init__(nState, nAction, epLen, nEps)
        self.privEps = privEps
        # noise at each local side
        self.noiseN = np.random.laplace(0,1./self.privEps,nEps)
        self.noiseR = np.random.laplace(0,1./self.privEps,nEps)
        self.noiseP = np.random.laplace(0,1./self.privEps,(nEps,self.nState))

    def compute_estModelBonus(self, ep):
        '''
        LDP version of updating estimates and bonus
        '''

        R_hat = {}
        P_hat = {}
        R_bonus = {}
        P_bonus = {}
        scaling = 2 * np.log(3)
        for s in range(self.nState):
            for a in range(self.nAction):


                counts = self.P_stats[s, a].sum()
                n1 = self.noiseN[0:ep] # the total noise is ep
                nosyCounts = counts + n1.sum()

                n2 = self.noiseP[0:ep]
                P_vec_noise = self.P_stats[s, a] + n2.sum(axis=0)
                eDelta = (1./self.privEps) * np.sqrt(ep) # the total noise is ep
                denom = np.maximum(1.0, nosyCounts + eDelta)

                P_hat[s, a] = P_vec_noise / denom
                P_bonus[s, a] = np.sqrt(scaling / denom)


                n3 = self.noiseR[0:ep] # the total noise is ep
                R_sum = R_sum = self.R_stats[s, a][1]
                R_noise = self.R_stats[s, a][0] * R_sum + n3.sum()
                denomR = np.maximum(1.0, R_sum + n1.sum() + eDelta)
                R_hat[s, a] = R_noise / denomR
                R_bonus[s, a] = np.sqrt(scaling / denomR) + (self.nState * eDelta / denomR)

        return R_hat, P_hat, R_bonus, P_bonus

#-----------------------------------------------------------------------------
# UCBVI-JDP
#-----------------------------------------------------------------------------
class UCBVI_JDP(UCBVI):
    '''UCBVI-JDP '''

    def __init__(self, nState, nAction, epLen, nEps, privEps):
        '''
        As per the tabular learner, but added tunable privEps.

        Args:
            privEps - double - privacy budget
        '''
        super(UCBVI_JDP, self).__init__(nState, nAction, epLen, nEps)
        self.privEps = privEps

    def compute_estModelBonus(self, ep):
        '''
        JDP version of updating estimates and bonus
        '''
        R_hat = {}
        P_hat = {}
        R_bonus = {}
        P_bonus = {}
        scaling = 2 * np.log(3)
        K = int(np.floor(np.log(ep)))+1
        for s in range(self.nState):
            for a in range(self.nAction):
                # fresh noise to mimic the tree-based algorithm
                # # the total noise is log(ep)
                n1 = np.random.laplace(0,1./self.privEps,K)
                n2 = np.random.laplace(0,1./self.privEps,(K,self.nState))
                n3 = np.random.laplace(0,1./self.privEps,K)


                counts = self.P_stats[s, a].sum()
                P_vec_noise = self.P_stats[s, a] + n2.sum(axis=0)
                eDelta = (1./self.privEps) * np.sqrt(K)
                denom = np.maximum(1.0, counts + n1.sum() + eDelta)

                P_hat[s, a] = P_vec_noise / denom
                P_bonus[s, a] = np.sqrt(scaling / denom) +  (self.nState * eDelta / denom)


                R_sum = R_sum = self.R_stats[s, a][1]
                R_noise = self.R_stats[s, a][0] * R_sum + n3.sum()
                denomR = np.maximum(1.0, R_sum + n1.sum() + eDelta)
                R_hat[s, a] = R_noise / denomR
                R_bonus[s, a] = np.sqrt(scaling / denomR) + (self.nState * eDelta / denomR)


        return R_hat, P_hat, R_bonus, P_bonus

#-----------------------------------------------------------------------------
# UCBPO
#-----------------------------------------------------------------------------
class UCBPO(UCBVI):
    '''UCBPO '''
    def update_policy(self, ep = False, eval = True):
        '''
        update new policy
        '''

        R_hat, P_hat, R_bonus, P_bonus = self.compute_estModelBonus(ep)

        self.compute_policy(R_hat, P_hat, R_bonus, P_bonus, eval)


    def pick_action(self, state, timestep):
        '''
        Use old policy to act and collect samples
        '''

        action = np.random.choice(self.nAction,1,p=self.pi_old[state,timestep])[0]

        return action


#-----------------------------------------------------------------------------
# UCBPO-LDP
#-----------------------------------------------------------------------------
class UCBPO_LDP(UCBVI_LDP):

    def update_policy(self, ep = False, eval = True):
        '''
        update new policy
        '''

        R_hat, P_hat, R_bonus, P_bonus = self.compute_estModelBonus(ep)

        self.compute_policy(R_hat, P_hat, R_bonus, P_bonus, eval)


    def pick_action(self, state, timestep):
        '''
        Use old policy to act and collect samples
        '''

        action = np.random.choice(self.nAction,1,p=self.pi_old[state,timestep])[0]

        return action


#-----------------------------------------------------------------------------
# UCBPO-JDP
#-----------------------------------------------------------------------------
class UCBPO_JDP(UCBVI_JDP):

    def update_policy(self, ep = False, eval = True):
        '''
        update new policy
        '''

        R_hat, P_hat, R_bonus, P_bonus = self.compute_estModelBonus(ep)

        self.compute_policy(R_hat, P_hat, R_bonus, P_bonus, eval)


    def pick_action(self, state, timestep):
        '''
        Use old policy to act and collect samples
        '''

        action = np.random.choice(self.nAction,1,p=self.pi_old[state,timestep])[0]

        return action
