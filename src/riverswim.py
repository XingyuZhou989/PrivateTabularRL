import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import environment
import agent

from experiment import run_experiment


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Run tabular RL experiment')

    parser.add_argument('alg', help='Agent constructor', type=str)
    parser.add_argument('nEps', help='number of episodes', type=int)
    parser.add_argument('nRuns', help='number of independent runs', type=int)
    parser.add_argument('privEps', help='privacy budget for private algorithms', type=float)


    args = parser.parse_args()

    # Make a filename to identify flags
    fileName = ('alg=' + str(args.alg)
                + '_nEps=' + str(args.nEps)
                + '_nRuns=' + str(args.nRuns)
                + '_priv=' + '%03.2f' % args.privEps
                + '.csv')

    folderName = './'
    targetPath = folderName + fileName
    print('******************************************************************')
    print(fileName)
    print('******************************************************************')



    env = environment.make_riverSwim()

    nRuns = args.nRuns
    nEps = args.nEps
    privEps = args.privEps


    alg_dict = {'UCBVI': agent.UCBVI,
                'UCBVI_LDP': agent.UCBVI_LDP,
                'UCBVI_JDP': agent.UCBVI_JDP,
                'UCBPO': agent.UCBPO,
                'UCBPO_LDP': agent.UCBPO_LDP,
                'UCBPO_JDP': agent.UCBPO_JDP
            }

    agent_constructor = alg_dict[args.alg]

    res = np.zeros(nEps)

    data_frame = np.zeros((nEps,nRuns))


    for run in range(nRuns):
        np.random.seed(run+1)

        # np.random.seed(1)
        agent = agent_constructor(env.nState, env.nAction, env.epLen,
                                  args.nEps, privEps = args.privEps)


        print('**************************************************')
        print('Run:', run+1, 'start')
        print('**************************************************')



        data = run_experiment(agent, env, nEps)
        data_frame[:,run] = data

        print('**************************************************')
        print('Run:', run+1,'end')
        print('**************************************************')

        res = res + (data - res) / (run + 1)



    dt = pd.DataFrame(data_frame)
    print('Writing to file ' + targetPath)
    dt.to_csv(targetPath, index=False, header = None, float_format='%.2f')


    eps = [i for i in range(nEps)]
    plt.plot(eps, res)
    plt.show()
