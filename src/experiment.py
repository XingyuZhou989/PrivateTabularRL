'''
Script to run simple tabular RL experiments.

'''

import numpy as np


def run_experiment(agent, env, nEps, fileFreq=1000, targetPath='tmp.csv'):

    qVals, vVals = env.compute_optVals()


    cumRegret = 0
    cumReward = 0
    empRegret = 0

    data = []

    for ep in range(1,nEps+1):
        # Reset the environment

        env.reset()
        epMaxVal = vVals[env.timestep][env.state]

        epReward = 0
        epRegret = 0
        pContinue = 1

        agent.update_policy(ep)

        while pContinue > 0:
            # Step through the episode
            h, oldState = env.get_states()

            action = agent.pick_action(oldState, h)
            epRegret += qVals[oldState, h].max() - qVals[oldState, h][action]

            reward, newState, pContinue = env.advance(action)
            epReward += reward

            agent.update_obs(oldState, action, reward, newState, pContinue, h)


        cumReward += epReward
        cumRegret += epRegret
        avgRegret = cumRegret / ep

        
        data.append(avgRegret)

        if ep < 1e4:
            recFreq = 100
        elif ep < 1e5:
            recFreq = 1000
        else:
            recFreq = 10000

        if ep % recFreq == 0:
            print('episode:', ep, 'epReward:', epReward, 'cumRegret:', cumRegret)


    return data
