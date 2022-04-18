import gym
import numpy as np
import matplotlib.pyplot as plt
from gym_gamblers.envs import GamblersEnv
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, show, colorbar
from numpy import sqrt
import random

from BoltzmannPolicies import BoltzmannPolicy
from frozen_lake import FrozenLakeEnv

dire_map = {0: '←', 1: '↓', 2: '↑', 3: '→'}


def argmax(env, V, pi, action, s, gamma):
    e = np.zeros(env.nA)
    for a in range(env.nA):  # iterate for every action possible
        q = 0
        P = np.array(env.P(s, a))
        (x, y) = np.shape(P)  # for Bellman Equation

        for i in range(x):  # iterate for every possible states
            s_ = int(P[i][0])  # S' - Sprime - possible succesor states
            p = P[i][1]  # Transition Probability P(s'|s,a)
            r = P[i][2]  # Reward

            q += p * (r + gamma * V[s_])  # calculate action_ value q(s|a)
            e[a] = q

    m = np.argmax(e)
    action[s] = m  # Take index which has maximum value
    pi[s][m] = 1  # update pi(a|s)

    return pi


def bellman_optimality_update(env, V, s, gamma):  # update the stae_value V[s] by taking
    pi = np.zeros((env.nS, env.nA))  # action which maximizes current value
    e = np.zeros(env.nA)
    # STEP1: Find
    for a in range(env.nA):
        q = 0  # iterate for all possible action
        P = np.array(env.P(s,a))
        (x, y) = np.shape(P)

        for i in range(x):
            s_ = int(P[i][0])
            p = P[i][1]
            r = P[i][2]
            q += p * (r + gamma * V[s_])
            e[a] = q

    m = np.argmax(e)
    pi[s][m] = 1

    value = 0
    for a in range(env.nA):
        u = 0
        P = np.array(env.P(s,a))
        (x, y) = np.shape(P)
        for i in range(x):
            s_ = int(P[i][0])
            p = P[i][1]
            r = P[i][2]

            u += p * (r + gamma * V[s_])

        value += pi[s, a] * u

    V[s] = value
    return V[s]


def value_iteration_helper(env, gamma, theta):
    V = np.zeros(env.nS)  # initialize v(0) to arbitory value, my case "zeros"
    i = 0
    while True:
        delta = 0
        i = i + 1
        for s in range(env.nS):  # iterate for all states
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)  # update state_value with bellman_optimality_update
            delta = max(delta, abs(v - V[s]))  # assign the change in value per iteration to delta
        if delta < theta:
            print("Value Iteration Converged {}".format(i))
            break  # if change gets to negligible

            # --> converged to optimal value
    pi = np.zeros((env.nS, env.nA))
    action = np.zeros((env.nS))
    for s in range(env.nS):
        pi = argmax(env, V, pi, action, s, gamma)  # extract optimal policy using action value

    return V, pi, action, i  # optimal value funtion, optimal policy


def value_iteration(env, gamma, theta):
    V, pi, action, itr = value_iteration_helper(env, gamma, theta)

    fig = plt.figure(figsize=(10, 5))

    plt.bar(range(len(V)), V, color='maroon',
            width=0.4)
    plt.xlabel("States")
    plt.ylabel("Values")
    plt.savefig('value_over_states.png')
    fig = plt.figure(figsize=(10, 5))

    plt.xlabel('State')
    plt.ylabel('Action')
    plt.imshow(pi.T)
    plt.colorbar()
    plt.savefig('action_prob.png')

    # row = int(sqrt(len(V)))
    # a = np.reshape(action, (row, row))
    print(action)  # discrete action to take in given state

    e = 0
    for i_episode in range(100):
        c = env.reset()
        for t in range(10000):
            res = env.step(action[int(c)])
            print(res)
            (c, a), reward, done, info = res
            if done:
                if reward == 1:
                    e += 1
                break
    print(" agent succeeded to reach goal {} out of 100 Episodes using this policy ".format(e + 1))
    env.close()
    return V, itr


def execute(env, policy, gamma=1.0, render=False):
    start = env.reset()
    totalReward = 0
    stepIndex = 0
    while True:
        if render:
            env.render()
        start, reward, done, _ = env.step(int(policy[start]))
        totalReward += (gamma ** stepIndex * reward)
        stepIndex += 1
        if done:
            break
    return


# Evaluation
def evaluatePolicy(env, policy, gamma=1.0, n=100):
    scores = [execute(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


# Extract the policy given a value-function
def extractPolicy(env, v, gamma=1.0):
    policy = np.zeros(env.nS)
    for s in range(1, env.nS):
        q_sa = np.arange(0, s+1,1)
        for a in range(len(q_sa)):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for s_,p, r, _ in env.P(s, a)])
            policy[s] = np.argmax(q_sa)
    return policy


# Iteratively calculates value-function under policy
def CalcPolicyValue(env, policy, gamma=1.0, theta=0.0001):
    value = np.zeros(env.nS)

    while True:
        previousValue = np.copy(value)
        for states in range(env.nS):
            policy_a = policy[states]
            value[states] = sum([p * (r + gamma * previousValue[int(s_)]) for s_, p, r, _ in env.P(states, policy_a)])
        if (np.sum((np.fabs(previousValue - value))) <= theta):
            break
            print("breaked")
    return value


# Policy Iteration algorithm
def policyIteration(env, gamma=1.0, theta=0.1):
    policy = np.arange(0, env.nS, 1)
    maxIterations = 1000
    policy_values = None
    itr = 0
    for i in range(maxIterations):
        oldPolicyValue = CalcPolicyValue(env, policy, gamma, theta)
        newPolicy = extractPolicy(env, oldPolicyValue, gamma)
        if (np.all(policy == newPolicy)):
            print('Policy Iteration converged at %d' % (i + 1))
            itr = i + 1
            policy_values = oldPolicyValue
            break
        policy = newPolicy

    fig = plt.figure(figsize=(10, 5))
    plt.bar(range(len(policy_values)), policy_values, color='maroon',
            width=0.4)
    plt.xlabel("States")
    plt.ylabel("Values")
    plt.savefig('value_over_states_policy_itr.png')
    fig = plt.figure(figsize=(10, 5))
    return policy, itr


def q_learning(env, alpha, discount_factor, epsilon, max_epsilon, min_epsilon, decay, train_episodes, test_episodes,
               max_steps):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # Training the agent

    # Creating lists to keep track of reward and epsilon values
    training_rewards = []
    epsilons = []

    for episode in range(train_episodes):
        # Reseting the environment each time as per requirement
        state = env.reset()
        # Starting the tracker for the rewards
        total_training_rewards = 0
        delta = 0
        for step in range(10000):
            # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)

            ### STEP 2: SECOND option for choosing the initial action - exploit
            # If the random number is larger than epsilon: employing exploitation
            # and selecting best action
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(Q[state, :])

                ### STEP 2: FIRST option for choosing the initial action - explore
            # Otherwise, employing exploration: choosing a random action
            else:
                action = env.action_space.sample()

            ### STEPs 3 & 4: performing the action and getting the reward
            # Taking the action and getting the reward and outcome state
            new_state, reward, done, info = env.step(action)
            val = Q[state, action]
            ### STEP 5: update the Q-table
            # Updating the Q-table using the Bellman equation
            Q[state, action] = Q[state, action] + alpha * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
            delta = max(delta, abs(Q[state, action] - val))
            # Increasing our total reward and updating the state
            total_training_rewards += reward
            state = new_state

            # Ending the episode
            if done == True:
                # print ("Total reward for episode {}: {}".format(episode, total_training_rewards))
                break

        # Cutting down on exploration by reducing the epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        # Adding the total reward and reduced epsilon values
        training_rewards.append(total_training_rewards)
        epsilons.append(epsilon)

    print("Training score over time: " + str(sum(training_rewards) / train_episodes))

    print("Find Optimal Actions in a state")
    action = np.chararray(len(Q), unicode=True)
    for st in range(len(Q)):
        action[st] = dire_map[int(np.argmax(Q[st, :]))]

    row = int(sqrt(len(Q)))
    action1 = action.reshape((row, row))
    print(action1)

    action = np.ones(len(Q))
    for st in range(len(Q)):
        action[st] = int(np.argmax(Q[st, :]))

    return action

    # Visualizing results and total reward over all episodes
    fig = plt.figure(figsize=(10, 5))
    x = range(train_episodes)
    plt.plot(x, training_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Training total reward')
    plt.title('Total rewards over all episodes in training')
    plt.savefig("Episodesvsrewards.png")

    # Visualizing the epsilons over all episodes
    fig = plt.figure(figsize=(10, 5))
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title("Epsilon for episode")
    plt.savefig("episodesvsepsilon.png")


def q_learning_boltzmann(env, alpha, discount_factor, tau1, max_epsilon, min_epsilon, decay, train_episodes,
                         test_episodes, max_steps):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # Training the agent
    exp_explr_strategy = BoltzmannPolicy(actions=range(env.nA), tau=tau1, tau_decay=False)
    # Creating lists to keep track of reward and epsilon values
    training_rewards = []
    epsilons = []

    for episode in range(train_episodes):
        # Reseting the environment each time as per requirement
        state = env.reset()
        # Starting the tracker for the rewards
        total_training_rewards = 0
        delta = 0
        for step in range(10000):
            # Choosing an action given the states based on a random number
            exp_exp_tradeoff = random.uniform(0, 1)

            ### STEP 2: SECOND option for choosing the initial action - exploit
            # If the random number is larger than epsilon: employing exploitation
            # and selecting best action
            # if exp_exp_tradeoff > epsilon:
            #     action = np.argmax(Q[state, :])
            #
            #     ### STEP 2: FIRST option for choosing the initial action - explore
            # # Otherwise, employing exploration: choosing a random action
            # else:
            #     action = env.action_space.sample()

            q_list = [Q[state, action1] for action1 in range(env.nA)]
            action = exp_explr_strategy.compute_action(q_list)

            ### STEPs 3 & 4: performing the action and getting the reward
            # Taking the action and getting the reward and outcome state
            new_state, reward, done, info = env.step(action)
            val = Q[state, action]
            ### STEP 5: update the Q-table
            # Updating the Q-table using the Bellman equation
            Q[state, action] = Q[state, action] + alpha * (
                    reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
            delta = max(delta, abs(Q[state, action] - val))
            # Increasing our total reward and updating the state
            total_training_rewards += reward
            state = new_state

            # Ending the episode
            if done == True:
                # print ("Total reward for episode {}: {}".format(episode, total_training_rewards))
                break

        # Cutting down on exploration by reducing the epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

        # Adding the total reward and reduced epsilon values
        training_rewards.append(total_training_rewards)
        epsilons.append(epsilon)

    print("Training score over time: " + str(sum(training_rewards) / train_episodes))

    print("Find Optimal Actions in a state")
    action = np.chararray(len(Q), unicode=True)
    for st in range(len(Q)):
        action[st] = dire_map[int(np.argmax(Q[st, :]))]

    row = int(sqrt(len(Q)))
    action1 = action.reshape((row, row))
    print(action1)

    action = np.ones(len(Q))
    for st in range(len(Q)):
        action[st] = int(np.argmax(Q[st, :]))

    return action

    # Visualizing results and total reward over all episodes
    fig = plt.figure(figsize=(10, 5))
    x = range(train_episodes)
    plt.plot(x, training_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Training total reward')
    plt.title('Total rewards over all episodes in training')
    plt.savefig("Episodesvsrewards.png")

    # Visualizing the epsilons over all episodes
    fig = plt.figure(figsize=(10, 5))
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title("Epsilon for episode")
    plt.savefig("episodesvsepsilon.png")


def play(env, optimal_actions):
    state = 0
    done = False
    win_count = 0
    for i in range(100):
        state = 0
        env.reset()
        while True:

            observation, reward, done, info = env.step(int(optimal_actions[state]))
            if done and reward > 0:
                win_count = win_count + 1
                break
            if done and reward == 0:
                break
            state = observation
    return win_count / 100


if __name__ == '__main__':
    # env = gym.make('FrozenLake-v0')
    dire_map = {0: '←', 1: '↓', 2: '↑', 3: '→'}
    print(dire_map)
    print(chr(int('0X2190', 16)))

    env = GamblersEnv()
    gamma = 0.75
    theta = 0.000001
    env.render()
    V, v_itr = value_iteration(env, gamma, theta)

    row = int(sqrt(len(V)))

    # print("Value Iterations - {}".format(V.reshape(row, row)))
    policy, p_itr = policyIteration(env, gamma, theta)
    policy, p_itr = policyIteration(env, gamma, theta)
    win_percentage = play(env, policy)
    print("Policy and Value Iterations percentage victory for 4*4 and gamma = 0.75 {}".format(win_percentage))
    row = int(sqrt(len(policy)))
    reshape_policy = policy.reshape(row, row)

    actions = np.chararray(reshape_policy.shape, unicode=True)

    for i in range(reshape_policy.shape[0]):
        for j in range(reshape_policy.shape[1]):
            actions[i][j] = reshape_policy[i][j]

    print(actions)
    #
    #
    #
    # impact of rewards
    # print("IMpact of Rewards on Function")
    # env = FrozenLakeEnv(map_name='4x4', reward=(0, 0, 10))
    # gamma = 0.99
    # theta = 0.000001
    # env.render()
    # V, v_itr = value_iteration(env, gamma, theta)
    #
    # row = int(sqrt(len(V)))
    #
    # print("Value Iterations - {}".format(V.reshape(row, row)))
    # policy, p_itr = policyIteration(env, gamma, theta)
    # win_percentage = play(env, policy)
    # print("Policy and Value Iterations percentage victory for 4*4 {}".format(win_percentage))
    # row = int(sqrt(len(policy)))
    # reshape_policy = policy.reshape(row, row)
    #
    # actions = np.chararray(reshape_policy.shape, unicode=True)
    #
    # for i in range(reshape_policy.shape[0]):
    #     for j in range(reshape_policy.shape[1]):
    #         actions[i][j] = dire_map[int(reshape_policy[i][j])]
    #
    # print(actions)
    # #
    #
    #
    # env = FrozenLakeEnv(map_name='8x8', reward=(0, 0, 10))
    # gamma = 0.99
    # theta = 0.000001
    # env.render()
    # V, v_itr = value_iteration(env, gamma, theta)
    #
    # row = int(sqrt(len(V)))
    #
    # print("Value Iterations - {}".format(V.reshape(row, row)))
    # policy, p_itr = policyIteration(env, gamma, theta)
    # row = int(sqrt(len(policy)))
    # reshape_policy = policy.reshape(row, row)
    #
    # actions = np.chararray(reshape_policy.shape, unicode=True)
    #
    # for i in range(reshape_policy.shape[0]):
    #     for j in range(reshape_policy.shape[1]):
    #         actions[i][j] = dire_map[int(reshape_policy[i][j])]
    #
    # print(actions)
    #
    # print("Impact of change of theta")
    # env = FrozenLakeEnv(map_name='4x4')
    # gamma = 0.99
    # theta = 0.01
    # env.render()
    # V, v_itr = value_iteration(env, gamma, theta)
    #
    # row = int(sqrt(len(V)))
    #
    # print("Value Iterations - {}".format(V.reshape(row, row)))
    # policy, p_itr = policyIteration(env, gamma, theta)
    # row = int(sqrt(len(policy)))
    # reshape_policy = policy.reshape(row, row)
    #
    # actions = np.chararray(reshape_policy.shape,unicode=True)
    #
    # for i in range(reshape_policy.shape[0]):
    #     for j in range(reshape_policy.shape[1]):
    #         actions[i][j] = dire_map[int(reshape_policy[i][j])]

    # print("Impact of change of Increase of Reward for Frozen State")
    # env = FrozenLakeEnv(map_name='4x4' , reward = (5,0,10))
    # gamma = 0.99
    # theta = 0.01
    # env.render()
    # V, v_itr = value_iteration(env, gamma, theta)
    #
    # row = int(sqrt(len(V)))
    #
    # print("Value Iterations - {}".format(V.reshape(row, row)))
    # policy, p_itr = policyIteration(env, gamma, theta)
    # row = int(sqrt(len(policy)))
    # reshape_policy = policy.reshape(row, row)
    #
    # actions = np.chararray(reshape_policy.shape,unicode=True)
    #
    # for i in range(reshape_policy.shape[0]):
    #     for j in range(reshape_policy.shape[1]):
    #         actions[i][j] = dire_map[int(reshape_policy[i][j])]
    #
    # print(actions)

    # print("Impact of negative Rewards for Frozen State and Hole State")
    # env = FrozenLakeEnv(map_name='4x4' , reward = (-1, -1, 10))
    # gamma = 0.99
    # theta = 0.01
    # env.render()
    # V, v_itr = value_iteration(env, gamma, theta)
    #
    # row = int(sqrt(len(V)))
    #
    # print("Value Iterations - {}".format(V.reshape(row, row)))
    # policy, p_itr = policyIteration(env, gamma, theta)
    # row = int(sqrt(len(policy)))
    # reshape_policy = policy.reshape(row, row)
    #
    # actions = np.chararray(reshape_policy.shape,unicode=True)
    #
    # for i in range(reshape_policy.shape[0]):
    #     for j in range(reshape_policy.shape[1]):
    #         actions[i][j] = dire_map[int(reshape_policy[i][j])]
    #
    # print(actions)

    # env = FrozenLakeEnv(map_name='8x8')
    # gamma = 0.99
    # theta = 0.01
    # env.render()
    # V, v_itr = value_iteration(env, gamma, theta)
    #
    # row = int(sqrt(len(V)))
    #
    # print("Value Iterations - {}".format(V.reshape(row, row)))
    # policy, p_itr = policyIteration(env, gamma, theta)
    # row = int(sqrt(len(policy)))
    # reshape_policy = policy.reshape(row, row)
    #
    # actions = np.chararray(reshape_policy.shape,unicode=True)
    #
    # for i in range(reshape_policy.shape[0]):
    #     for j in range(reshape_policy.shape[1]):
    #         actions[i][j] = dire_map[int(reshape_policy[i][j])]
    #
    # print(actions)

    # Setting the hyperparameters
    # print("Q Learning")
    # env = FrozenLakeEnv(map_name='4x4',reward = (0, 0, 1))
    # alpha = 0.7  # learning rate
    # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # discount_factor = 0.618
    # epsilon = 1
    # max_epsilon = 1
    # min_epsilon = 0.01
    # decay = 0.01
    #
    # train_episodes = 20000
    # test_episodes = 1000
    # max_steps = 1000
    # wins = []
    # for alpha in alphas:
    #     print("Working with alpha - {}".format(alpha))
    #     max_win = -99999.0
    #     for i in range(3):
    #         env.reset()
    #         action = q_learning(env, alpha, discount_factor, epsilon, max_epsilon, min_epsilon, decay, train_episodes, test_episodes, max_steps)
    #         win_percentage = play(env, action)
    #         max_win = max(win_percentage, max_win)
    #     wins.append(max_win)
    #     print("percentage victory {}".format(max_win))
    #
    #
    # fig = plt.figure(figsize=(10, 5))
    # x = range(train_episodes)
    # plt.plot(alphas, wins)
    # plt.xlabel('Alpha')
    # plt.ylabel('Win Percentage')
    # plt.title('Win Percentage as Function of Alpha')
    # plt.savefig("winpercentagevsalpha.png")

    # alpha = 0.8  # learning rate
    # # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # # discount_factor = 0.618
    # discount_factors = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # epsilon = 1
    # max_epsilon = 1
    # min_epsilon = 0.01
    # decay = 0.01
    #
    # train_episodes = 20000
    # test_episodes = 1000
    # max_steps = 1000
    # wins = []
    # for discount_factor in discount_factors:
    #     print("Working with discount_factor - {}".format(discount_factor))
    #     max_win = -99999.0
    #     for i in range(3):
    #         env.reset()
    #         action = q_learning(env, alpha, discount_factor, epsilon, max_epsilon, min_epsilon, decay, train_episodes, test_episodes, max_steps)
    #         win_percentage = play(env, action)
    #         max_win = max(win_percentage, max_win)
    #     wins.append(max_win)
    #     print("percentage victory {}".format(max_win))
    #
    #
    # fig = plt.figure(figsize=(10, 5))
    # x = range(train_episodes)
    # plt.plot(discount_factors, wins)
    # plt.xlabel('Discount Factor(Gamma)')
    # plt.ylabel('Win Percentage')
    # plt.title('Win Percentage as Function of Gamma')
    # plt.savefig("winpercentagevsgamma.png")

    # alpha = 0.8  # learning rate
    # # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # # discount_factor = 0.618
    # # discount_factors = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # discount_factor = 0.9
    # epsilon = 1
    # max_epsilon = 1
    # min_epsilon = 0.01
    # decays = [0.0001, 0.001, 0.01, 0.1]
    #
    # train_episodes = 20000
    # test_episodes = 1000
    # max_steps = 1000
    # wins = []
    # for decay in decays:
    #     print("Working with Decay - {}".format(decay))
    #     max_win = -99999.0
    #     for i in range(3):
    #         env.reset()
    #         action = q_learning(env, alpha, discount_factor, epsilon, max_epsilon, min_epsilon, decay, train_episodes, test_episodes, max_steps)
    #         win_percentage = play(env, action)
    #         max_win = max(win_percentage, max_win)
    #     wins.append(max_win)
    #     print("percentage victory {}".format(max_win))
    #
    # fig = plt.figure()
    # width = 0.35
    #
    # x_pos = [i for i, _ in enumerate(decays)]
    #
    # plt.bar(x_pos, wins, color='green')
    # plt.xlabel("Epsilon Decay")
    # plt.ylabel("Win Percentage")
    # plt.title("Win Percentage as Function of Epsilon Decay")
    #
    # plt.xticks(x_pos, decays)
    #
    # plt.savefig("winpercentagevsepsilon.png")

    # print("BoltZmannnnnnnnnnnnnnnnnnnnn")
    # alpha = 0.8  # learning rate
    # # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # # discount_factor = 0.618
    # # discount_factors = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # discount_factor = 0.9
    # epsilon = 1
    # max_epsilon = 1
    # min_epsilon = 0.01
    # # decays = [0.0001, 0.001, 0.01, 0.1]
    # decay = 0.001
    # taus = [10, 100, 1000, 10000, 100000]
    # train_episodes = 20000
    # test_episodes = 1000
    # max_steps = 1000
    # wins = []
    # for tau in taus:
    #     print("Working with Tau  - {}".format(tau))
    #     max_win = -99999.0
    #     for i in range(3):
    #         env.reset()
    #         action = q_learning_boltzmann(env, alpha, discount_factor, tau, max_epsilon, min_epsilon, decay,
    #                                       train_episodes, test_episodes, max_steps)
    #         win_percentage = play(env, action)
    #         max_win = max(win_percentage, max_win)
    #     wins.append(max_win)
    #     print("percentage victory {}".format(max_win))
    #
    # fig = plt.figure()
    # width = 0.35
    #
    # x_pos = [i for i, _ in enumerate(taus)]
    #
    # plt.bar(x_pos, wins, color='green')
    # plt.xlabel("Tau")
    # plt.ylabel("Win Percentage")
    # plt.title("Win Percentage as Function of Tauy")
    #
    # plt.xticks(x_pos, taus)
    #
    # plt.savefig("winpercentagevstau_boltzman.png")

# creating the bar plot
