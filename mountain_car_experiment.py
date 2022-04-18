"""
Original work by Ankit Choudhary
(https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)
and by Genevieve Hayes (https://gist.github.com/gkhayes/3d154e0505e31d6367be22ed3da2e955#file-mountain_car-py)
Modified by Matthieu Divet (mdivet3@gatech.edu)
"""
from math import sqrt

import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import copy
# Import and initialize Mountain Car Environment
from mountain_car import MountainCarEnv

env = MountainCarEnv()
env.reset()
mountain_car_nA = 3
# discretization of the environment's state space into a 91 x 71 space = 6461 possible states
# (position in the range [-60, 30]/50 and velocity in the range [-35, 35]/500)
# mountain_car_discretized_nS = np.prod(np.round((env.observation_space.high - env.observation_space.low)
#                                                * np.array([50, 500]), 0).astype(int) + 1)
num_states = (env.observation_space.high - env.observation_space.low) * \
             np.array([10, 100])
num_states = np.round(num_states, 0).astype(int)

mountain_car_discretized_nS = num_states[0] * num_states[1] + 1

def convertIndexToState(index):
    return np.array([index/10 , index/100]) + env.observation_space.low

def convertStateToIndex(state):
    next_state_index = (state - env.observation_space.low) * np.array([10, 100])
    row, col = np.round(next_state_index, 0).astype(int)
    return max(0, row - 1) * num_states[1] + max(0, col - 1)



def one_step_lookahead(env, state, utility, discount_factor):
    """
    max(action_values) = max(sum [(1/nb_of_possible_s') * r(s) + T(s, a, s') * gamma * U(s')])
                       = r(s) + gamma * max(sum [T(s, a, s') * U(s')])
    with max over the possible actions at state s and the sum over the possible s'
    and where T(s, a, s') = 1 because of the following piece of code from the definition of env.step(action):
            position, velocity = self.state
            velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)
            position += velocity
            position = np.clip(position, self.min_position, self.max_position)
            if (position==self.min_position and velocity<0): velocity = 0
    which means that s' (given by [position, velocity]) is entirely defined by s and a.
    This is Bellman's equation, seen in class. Therefore we can use max(action_values) in value_iteration()
    and argmax(action_values) in policy_iteration().


    Also, this environment does not have a transition matrix like Frozen Lake. So, to analyze state s, we convert its
    index into the corresponding values (position, velocity) and we then set the environment's state to be
    [position, velocity]. Therefore, calling environment.step(action) will now give the next state, the reward,
    if the next state is final and info based on the state we set.
    """
    action_values = np.zeros(mountain_car_nA)
    # position = float((state // 71) - 60) / 50
    # velocity = float(state - 71 * (state // 71) - 35) / 500
    position, velocity = convertIndexToState(state)
    env.state = np.array([position, velocity])
    for action in range(mountain_car_nA):
        next_state, reward, done, info = env.step(action)
        if done:
            reward = 1
        # next_state_index = ((np.round(next_state[0] * 50, 0).astype(int) + 60) * 71) \
        #                    + np.round(next_state[1] * 500, 0).astype(int) + 35
        # action_values[action] += reward + discount_factor * utility[next_state_index]

        next_state_index = convertStateToIndex(next_state)
        action_values[action] += reward + discount_factor * utility[next_state_index]

    return action_values


# def value_iteration(env, discount_factor=0.9, theta=1e-9, max_iterations=1e9):
#     # Initialize state-value function with zeros for each environment state
#     utility = np.zeros(mountain_car_discretized_nS)
#     t0 = time.time()
#     converged = False
#     deltas = []
#     iteration = 0
#     for i in range(int(max_iterations)):
#         # Early stopping condition
#         delta = 0.0
#         # Update each state
#         for state in range(mountain_car_discretized_nS):
#             # Do a one-step lookahead to calculate state-action values
#             action_value = one_step_lookahead(env, state, utility, discount_factor)
#             # Select best action to perform based on the highest state-action value
#             best_action_value = np.max(action_value)
#             # Calculate change in value
#             delta = max(delta,  abs(utility[state] - best_action_value))
#             # Update the value function for current state
#             utility[state] = best_action_value
#             # Check if we can stop
#             # the utility function has converged - another iteration wouldn't improve its estimation by more than theta)
#         deltas.append(delta)
#         print(utility)
#         if delta < theta:
#             print('Value-iteration converged at iteration #{}.'.format(i))
#             print('Converged after {} seconds.'.format(round(time.time() - t0, 2)))
#             converged = True
#             iteration = i
#             break
#
#     if not converged:
#         print('Did not converge after {} iterations.'.format(max_iterations))
#
#     plt.figure()
#     plt.plot((np.arange(len(deltas)) + 1), deltas)
#     plt.xlabel('Iterations')
#     plt.ylabel('Average Delta')
#     plt.title('Delta vs Iterations')
#     plt.show()
#
#     # Create a deterministic policy using the optimal value function
#     pol = np.zeros([mountain_car_discretized_nS, mountain_car_nA])
#     for state in range(mountain_car_discretized_nS):
#         # One step lookahead to find the best action for this state
#         action_value = one_step_lookahead(env, state, utility, discount_factor)
#         # Select best action based on the highest state-action value
#         best_action = np.argmax(action_value)
#         # Update the policy to perform a better action at a current state
#         pol[state, best_action] = 1.0
#     return utility, iteration

def bellman_optimality_update(env, V, s, gamma):  # update the stae_value V[s] by taking
    pi = np.zeros((env.nS, env.nA))  # action which maximizes current value
    e = np.zeros(env.nA)
    # STEP1: Find
    for a in range(env.nA):
        q = 0  # iterate for all possible action
        next_state, reward, done, info = env.step(a)
        next_state_index = convertStateToIndex(next_state)

        e[a] = (reward + gamma * V[next_state_index])


    m = np.argmax(e)
    pi[s][m] = 1

    value = 0
    for a in range(env.nA):
        u = 0
        next_state, reward, done, info = env.step(a)
        next_state_index = convertStateToIndex(next_state)
        u += (reward + gamma * V[next_state_index])
        value += pi[s, a] * u

    V[s] = value
    return V[s]

def argmax(env, V, pi, action, s, gamma):
    e = np.zeros(env.nA)
    for a in range(env.nA):  # iterate for every action possible
        q = 0
        next_state, reward, done, info = env.step(a)

        s_ = convertStateToIndex(next_state)


        q += (reward + gamma * V[s_])  # calculate action_ value q(s|a)
        e[a] = q

    m = np.argmax(e)
    action[s] = m  # Take index which has maximum value
    pi[s][m] = 1  # update pi(a|s)

    return pi

def value_iteration_helper(env, gamma, theta):
    V = np.zeros(env.nS)  # initialize v(0) to arbitory value, my case "zeros"
    i = 0
    while True:
        print("Value Iterations - {}".format(i))
        delta = 0
        i = i + 1
        for s in range(env.nS):  # iterate for all states
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)  # update state_value with bellman_optimality_update
            delta = max(delta, abs(v - V[s]))  # assign the change in value per iteration to delta
        print("Delta - {}".format(delta))
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

    row = int(sqrt(len(V)))
    a = np.reshape(action, (num_states[0], num_states[1]))
    print(a)  # discrete action to take in given state

    e = 0
    for i_episode in range(100):
        c = convertStateToIndex(env.reset())
        for t in range(10000):
            ns, reward, done, info = env.step(int(action[c]))
            c = convertStateToIndex(ns)
            env.state = ns
            if done:
                if reward == 1:
                    e += 1
                break
    print(" agent succeeded to reach goal {} out of 100 Episodes using this policy ".format(e + 1))
    env.close()
    return V, itr



def policy_evaluation(pol, env, discount_factor=0.9, theta=1e-9, max_iterations=1e7):
    # Number of evaluation iterations
    evaluation_iterations = 1
    # Initialize a value function for each state as zero
    utility = np.zeros(mountain_car_discretized_nS)
    # Repeat until change in value is below the threshold
    for i in range(int(max_iterations)):
        # Initialize a change of value function as zero
        delta = 0
        # Iterate though each state
        for state in range(mountain_car_discretized_nS):
            position, velocity = convertIndexToState(state)
            env.state = np.array([position, velocity])
            # Initial a new value of current state
            u = 0
            # Try all possible actions which can be taken from this state
            for action, action_probability in enumerate(pol[state]):
                # Check how good next state will be
                # Calculate the expected value (action_probability will actually just be 0 or 1, except in the
                # very first iteration of policy_iteration() - therefore we will indeed be evaluating the utility of
                # the state as if the action to take was indeed the one given by the policy we're evaluating. This
                # is linked to the absence of a max over the possible actions in the formula for the U_t(s))
                next_state, reward, done, info = env.step(action)
                if done:
                    reward = 1
                next_state_index = convertStateToIndex(next_state)
                u += action_probability * (reward + discount_factor * utility[next_state_index])
            # Calculate the absolute change of value function
            delta = max(delta, np.abs(utility[state] - u))
            # Update value function
            utility[state] = u
        evaluation_iterations += 1
        # Terminate if value change is insignificant
        if delta < theta:
            print('Policy evaluated in {} iterations.'.format(evaluation_iterations))
            return utility


def policy_iteration(envi, discount_factor=0.9, max_iterations=1e3):
    # Start with a random policy
    # num states x num actions / num actions
    poli = np.ones([mountain_car_discretized_nS, mountain_car_nA]) / mountain_car_nA
    # Initialize counter of evaluated policies
    evaluated_policies = 0
    # Repeat until convergence or critical number of iterations reached
    t0 = time.time()
    converged = False
    itr = 0
    list_of_changes_in_policy = []

    for i in range(int(max_iterations)):
        stable_policy = True
        changes_in_policy = 0
        # Evaluate current policy
        U = policy_evaluation(poli, envi, discount_factor=discount_factor)
        # Go through each state and try to improve actions that were taken (policy Improvement)
        for state in range(mountain_car_discretized_nS):
            # Choose the best action in a current state under current policy
            current_action = np.argmax(poli[state])
            # Look one step ahead and evaluate if current action is optimal
            # We will try every possible action in a current state
            action_value = one_step_lookahead(envi, state, U, discount_factor)
            # Select a better action
            best_action = np.argmax(action_value)
            # If action changed
            if current_action != best_action:
                stable_policy = False
                changes_in_policy += 1
                # Greedy policy update
                # set all actions to 0 and the best action to 1 for current state in the matrix that represents
                # the policy
                poli[state] = np.eye(mountain_car_nA)[best_action]
        evaluated_policies += 1
        list_of_changes_in_policy.append(changes_in_policy)
        print('{} actions changed.'.format(changes_in_policy))
        # If the algorithm converged and policy is not changing anymore, then return final policy and value function
        if stable_policy:
            print('Evaluated {} policies.'.format(evaluated_policies))
            print('Evaluated in {} seconds.'.format(round(time.time() - t0, 2)))
            converged = True
            itr = i + 1
            break
    if not converged:
        print('Did not converge after {} iterations.'.format(max_iterations))
    plt.figure()
    plt.plot((np.arange(len(list_of_changes_in_policy)) + 1), list_of_changes_in_policy)
    plt.xlabel('Iterations')
    plt.ylabel('Number of changes in policy')
    plt.title('Number of changes in policy vs Iterations')
    plt.show()
    return poli, U, itr


def play_episodes(env, nb_episodes, pol):
    win = 0
    tot_reward = 0
    for episode in range(nb_episodes):
        terminated = False
        state = env.reset()
        state_index = convertStateToIndex(state)
        while not terminated:
            # Select best action to perform in a current state
            action = np.argmax(pol[state_index])
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = env.step(action)
            next_state_index = convertStateToIndex(next_state)
            # Summarize total reward
            tot_reward += reward
            # Update current state
            state_index = next_state_index
            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                win += 1
    avg_reward = tot_reward / nb_episodes
    return win/nb_episodes, tot_reward, avg_reward


# # Number of episodes to play
# n_episodes = 10000
# # Functions to find best policy
# solvers = [('Value Iteration', value_iteration),
#            ('Policy Iteration', policy_iteration)]
# for iteration_name, iteration_func in solvers:
#     # Load a Mountain Car environment
#     env = gym.make('MountainCar-v0')
#     # Search for an optimal policy using policy iteration
#     policy, V = iteration_func(env.env)
#     print('{} :: policy found = {}'.format(iteration_name, policy))
#     print('{} :: values found = {}'.format(iteration_name, V))
#     # Apply best policy to the real environment
#     wins, total_reward, average_reward = play_episodes(env, n_episodes, policy)
#     print('{} :: number of wins over {} episodes = {}'.format(iteration_name, n_episodes, wins))
#     print('{} :: average reward over {} episodes = {} \n\n'.format(iteration_name, n_episodes, average_reward))
import random
def q_learning(env, alpha, discount_factor, epsilon, max_epsilon, min_epsilon, decay, train_episodes, test_episodes,
               max_steps):
    Q = np.zeros((mountain_car_discretized_nS, mountain_car_nA))

    # Training the agent

    # Creating lists to keep track of reward and epsilon values
    training_rewards = []
    epsilons = []

    for episode in range(train_episodes):
        # Reseting the environment each time as per requirement
        state = convertStateToIndex(env.reset())
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
            new_state = convertStateToIndex(new_state)
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

    # print("Find Optimal Actions in a state")
    # action = np.chararray(len(Q), unicode=True)
    # for st in range(len(Q)):
    #     action[st] = dire_map[int(np.argmax(Q[st, :]))]
    #
    # row = int(sqrt(len(Q)))
    # action1 = action.reshape((row, row))
    # print(action1)
    #
    # action = np.ones(len(Q))
    # for st in range(len(Q)):
    #     action[st] = int(np.argmax(Q[st, :]))



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
    return sum(training_rewards) / train_episodes



# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes, exploration="linear-decay"):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    # Calculate episodic reduction in epsilon in case exploration == "linear-decay"
    reduction = (epsilon - min_eps) / episodes

    # If exploration == "greedy" then epsilon needs to be 0
    if exploration == "greedy":
        epsilon = 0

    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low) * np.array([100, 1000])
        state_adj = np.round(state_adj, 0).astype(int)

        while done is not True:
            # Render environment for last twenty episodes
            if i >= (episodes - 20):
                env.render()

            # Determine next action - epsilon greedy strategy (if epsilon == 0 this is just a greedy exploration)
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low) * np.array([100, 1000])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states and render successful state
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                env.render()

            # Adjust Q value for current state
            else:
                delta = learning * (reward +
                                    discount * np.max(Q[state2_adj[0],
                                                        state2_adj[1]]) -
                                    Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_eps:
            if exploration == "linear-decay":
                epsilon -= reduction
            elif exploration == "exp-decay":
                epsilon *= 1/2
            # elif exploration == "greedy" or exploration == "epsilon-greedy" epsilon mudt not change

        # Track rewards
        reward_list.append(tot_reward)

        if (i + 1) % 1000 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []

        if (i + 1) % 1000 == 0:
            print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))

    env.close()

    return np.mean(ave_reward_list)


# Iteratively calculates value-function under policy
def CalcPolicyValue(env, policy, gamma=1.0, theta=0.0001):
    value = np.zeros(env.nS)
    env_cp = copy.deepcopy(env)
    while True:
        previousValue = np.copy(value)
        for states in range(env.nS):
            position, velocity = convertIndexToState(states)
            env_cp.state = np.array([position, velocity])
            policy_a = policy[states]
            next_state, reward, done, info = env_cp.step(int(policy_a))
            value[states] = reward + gamma * previousValue[convertStateToIndex(next_state)]
        if (np.sum((np.fabs(previousValue - value))) <= theta):
            break
            print("breaked")
    return value


# Extract the policy given a value-function
def extractPolicy(env, v, gamma=1.0):
    policy = np.zeros(env.nS)
    env_cp = copy.deepcopy(env)
    for s in range(env.nS):
        position, velocity = convertIndexToState(s)
        env_cp.state = np.array([position, velocity])

        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            next_state, reward, done, info = env_cp.step(a)
            q_sa[a] = reward + gamma * v[convertStateToIndex(next_state)]
            policy[s] = np.argmax(q_sa)
    return policy


# Policy Iteration algorithm
def policyIteration(env, gamma=1.0, theta=0.1):
    policy = np.random.choice(env.nA, size=(env.nS))
    maxIterations = 1000
    policy_values = None
    itr = 0
    for i in range(maxIterations):
        oldPolicyValue = CalcPolicyValue(env, policy, gamma, theta)
        newPolicy = extractPolicy(env, oldPolicyValue, gamma)
        if (np.allclose(policy , newPolicy)):
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


def play(env, optimal_actions):
    state = 0
    done = False
    win_count = 0
    total_reward = 0
    for i in range(100):
        state = 0
        state = env.reset()

        for i in range(9000):

            observation, reward, done, info = env.step(int(optimal_actions[convertStateToIndex(state)]))
            total_reward = total_reward + reward
            if done and reward > 0:
                win_count = win_count + 1
                break
            if done and reward == 0:
                break
            state = observation
            env.state = observation
    return win_count / 100, total_reward/100

# Run Q-learning algorithm
# rewards = QLearning(environment, 0.2, 0.9, 0.04, 0, 100000, exploration="greedy")
#
# # Plot Rewards
# plt.plot(1000 * (np.arange(len(rewards)) + 1), rewards)
# plt.xlabel('Episodes')
# plt.ylabel('Average Reward')
# plt.title('Average Reward vs Episodes')
# plt.savefig('Rewards_graphs/rewards_mountain_car_greedy.jpg')
# plt.close()
# print("Gamma Values 1")
# env.reset()
# gamma = 0.75
# theta = 0.000001
# V, v_itr = value_iteration(env, gamma, theta)
#
# row = int(sqrt(len(V)))
#
# print("Value Iterations - {}".format(v_itr))
# env.state = env.reset()
# policy, p_itr = policyIteration(env, gamma, theta)
# print("Policy Iterations - {}".format(p_itr))
# win_percentage = play(env, policy)
# print("Policy and Value Iterations percentage victory for 4*4 and gamma = 0.75 {}".format(win_percentage))
# # row = int(sqrt(len(policy)))
# # reshape_policy = policy.reshape(row, row)
# #
# # actions = np.chararray(reshape_policy.shape, unicode=True)
#
#
#
# print("IMpact of Rewards on Function")
# env.state = env.reset()
# gamma = 0.99
# theta = 0.000001
# V, v_itr = value_iteration(env, gamma, theta)
#
#
#
# print("Value Iterations - {}".format(v_itr))
# env.state = env.reset()
# policy, p_itr = policyIteration(env, gamma, theta)
# print("Policy Iterations - {}".format(p_itr))
# win_percentage = play(env, policy)
# print("Policy and Value Iterations percentage victory for 4*4 {}".format(win_percentage))
#
#
#
#
#
#
# print("Theta Value 1")
# env.state = env.reset()
# gamma = 0.99
# theta = 0.01
#     # env.render()
# V, v_itr = value_iteration(env, gamma, theta)
#
# row = int(sqrt(len(V)))
#
# print("Value Iterations - {}".format(v_itr))
# env.state = env.reset()
# policy, p_itr = policyIteration(env, gamma)
#
# win_percentage = play(env, policy)
# print("2. Results Value Iterations - {} Policy Iterations - {} , win_percentage - {}".format(v_itr, p_itr, win_percentage))
#
#
# print("Theta Value 2")
# env.state = env.reset()
# gamma = 0.99
# theta = 0.000001
#     # env.render()
# V, v_itr = value_iteration(env, gamma, theta)
#
# row = int(sqrt(len(V)))
#
# print("Value Iterations - {}".format(v_itr))
# env.state = env.reset()
# policy, p_itr = policyIteration(env, gamma)
#
# win_percentage = play(env, policy)
# print("2. Results Value Iterations - {} Policy Iterations - {} , win_percentage - {}".format(v_itr, p_itr, win_percentage))


# print("Impact of rewards")
# gamma = 0.99
# theta = 0.006
#     # env.render()
# env.reset()
# V, v_itr = value_iteration(env, gamma, theta)
# env.reset()
# policy, p_itr = policyIteration(env, gamma)
# win_percentage , reward = play(env, policy)
# print("Value Iterations - {}, Policy Iterations - {}, avergae reward - {}".format(v_itr,p_itr,reward))












# # Setting the hyperparameters
print("Q Learning")
env.reset()
alpha = 0.7  # learning rate
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
discount_factor = 0.618
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_episodes = 20000
test_episodes = 1000
max_steps = 1000
wins = []
for alpha in alphas:
    print("Working with alpha - {}".format(alpha))
    max_win = -99999.0

    env.reset()
    avg_rewards_list = q_learning(env, alpha, discount_factor, epsilon, max_epsilon, min_epsilon, decay, train_episodes, test_episodes,
               max_steps)
    wins.append(avg_rewards_list)
    print("percentage victory {}".format(avg_rewards_list))


fig = plt.figure(figsize=(10, 5))
x = range(train_episodes)
plt.plot(alphas, wins)
plt.xlabel('Alpha')
plt.ylabel('Average Reward')
plt.title('Average Reward as Function of Alpha')
plt.savefig("rewardpercentagevsalpha.png")


