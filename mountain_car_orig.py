"""
Original work by Ankit Choudhary
(https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)
and by Genevieve Hayes (https://gist.github.com/gkhayes/3d154e0505e31d6367be22ed3da2e955#file-mountain_car-py)
Modified by Matthieu Divet (mdivet3@gatech.edu)
"""
import random

import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Import and initialize Mountain Car Environment
from BoltzmannPolicies import BoltzmannPolicy
from mountain_car import MountainCarEnv

environment = gym.make('MountainCar-v0')
environment.reset()
mountain_car_nA = 3
# discretization of the environment's state space into a 91 x 71 space = 6461 possible states
# (position in the range [-60, 30]/50 and velocity in the range [-35, 35]/500)
mountain_car_discretized_nS = np.prod(np.round((environment.observation_space.high - environment.observation_space.low)
                                               * np.array([50, 500]), 0).astype(int) + 1)


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
    position = float((state // 71) - 60) / 50
    velocity = float(state - 71 * (state // 71) - 35) / 500
    env.state = np.array([position, velocity])
    for action in range(mountain_car_nA):
        next_state, reward, done, info = env.step(action)
        next_state_index = ((np.round(next_state[0] * 50, 0).astype(int) + 60) * 71) \
                           + np.round(next_state[1] * 500, 0).astype(int) + 35
        action_values[action] += reward + discount_factor * utility[next_state_index]
    return action_values


def value_iteration(env, discount_factor=0.9, theta=1e-9, max_iterations=1e9):
    # Initialize state-value function with zeros for each environment state
    utility = np.zeros(mountain_car_discretized_nS)
    t0 = time.time()
    converged = False
    deltas = []
    for i in range(int(max_iterations)):
        # Early stopping condition
        delta = 0.0
        # Update each state
        for state in range(mountain_car_discretized_nS):
            # Do a one-step lookahead to calculate state-action values
            action_value = one_step_lookahead(env, state, utility, discount_factor)
            # Select best action to perform based on the highest state-action value
            best_action_value = np.max(action_value)
            # Calculate change in value
            delta += (utility[state] - best_action_value)/mountain_car_discretized_nS
            # Update the value function for current state
            utility[state] = best_action_value
            # Check if we can stop
            # the utility function has converged - another iteration wouldn't improve its estimation by more than theta)
        deltas.append(delta)
        print(utility)
        if delta < theta:
            print('Value-iteration converged at iteration #{}.'.format(i))
            print('Converged after {} seconds.'.format(round(time.time() - t0, 2)))
            converged = True
            break

    if not converged:
        print('Did not converge after {} iterations.'.format(max_iterations))

    plt.figure()
    plt.plot((np.arange(len(deltas)) + 1), deltas)
    plt.xlabel('Iterations')
    plt.ylabel('Average Delta')
    plt.title('Delta vs Iterations')
    plt.show()

    # Create a deterministic policy using the optimal value function
    pol = np.zeros([mountain_car_discretized_nS, mountain_car_nA])
    for state in range(mountain_car_discretized_nS):
        # One step lookahead to find the best action for this state
        action_value = one_step_lookahead(env, state, utility, discount_factor)
        # Select best action based on the highest state-action value
        best_action = np.argmax(action_value)
        # Update the policy to perform a better action at a current state
        pol[state, best_action] = 1.0
    return pol, utility


def policy_evaluation(pol, env, discount_factor=0.9, theta=1e-9, max_iterations=1e9):
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
            position = float((state // 71) - 60) / 50
            velocity = float(state - 71 * (state // 71) - 35) / 500
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
                next_state_index = ((np.round(next_state[0] * 50, 0).astype(int) + 60) * 71) \
                                   + np.round(next_state[1] * 500, 0).astype(int) + 35
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


def policy_iteration(envi, discount_factor=0.9, max_iterations=1e9):
    # Start with a random policy
    # num states x num actions / num actions
    poli = np.ones([mountain_car_discretized_nS, mountain_car_nA]) / mountain_car_nA
    # Initialize counter of evaluated policies
    evaluated_policies = 0
    # Repeat until convergence or critical number of iterations reached
    t0 = time.time()
    converged = False
    list_of_changes_in_policy = []
    num_itr = 0
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
            num_itr = i
            converged = True
            break
    if not converged:
        print('Did not converge after {} iterations.'.format(max_iterations))
    plt.figure()
    plt.plot((np.arange(len(list_of_changes_in_policy)) + 1), list_of_changes_in_policy)
    plt.xlabel('Iterations')
    plt.ylabel('Number of changes in policy')
    plt.title('Number of changes in policy vs Iterations')
    plt.show()
    return poli, itr


def play_episodes(env, nb_episodes, pol):
    win = 0
    tot_reward = 0
    for episode in range(nb_episodes):
        terminated = False
        state = env.reset()
        state_index = ((np.round(state[0] * 50, 0).astype(int) + 60) * 71)\
                      + np.round(state[1] * 500, 0).astype(int) + 35
        while not terminated:
            # Select best action to perform in a current state
            action = np.argmax(pol[state_index])
            # Perform an action an observe how environment acted in response
            next_state, reward, terminated, info = env.step(action)
            next_state_index = ((np.round(next_state[0] * 50, 0).astype(int) + 60) * 71)\
                               + np.round(next_state[1] * 500, 0).astype(int) + 35
            # Summarize total reward
            tot_reward += reward
            # Update current state
            state_index = next_state_index
            # Calculate number of wins over episodes
            if terminated and reward == 1.0:
                win += 1
    avg_reward = tot_reward / nb_episodes
    return win, tot_reward, avg_reward


# Number of episodes to play
n_episodes = 10000
# Functions to find best policy
# solvers = [('Value Iteration', value_iteration),
#            ('Policy Iteration', policy_iteration)]
# for iteration_name, iteration_func in solvers:
#     # Load a Mountain Car environment
#     environment = gym.make('MountainCar-v0')
#     # Search for an optimal policy using policy iteration
#     policy, V = iteration_func(environment.env)
#     print('{} :: policy found = {}'.format(iteration_name, policy))
#     print('{} :: values found = {}'.format(iteration_name, V))
#     # Apply best policy to the real environment
#     wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)
#     print('{} :: number of wins over {} episodes = {}'.format(iteration_name, n_episodes, wins))
#     print('{} :: average reward over {} episodes = {} \n\n'.format(iteration_name, n_episodes, average_reward))

# print("Experiment 1")
# gamma = 0.75
# theta = 0.000001
# environment = gym.make('MountainCar-v0')
# environment.reset()
# policy, utility = value_iteration(environment,gamma, theta)
# pol, itr = policy_iteration(environment,gamma)
# wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)
# print("Results - {} , {}, {}, {}".format(wins, total_reward, average_reward, itr))
#
#
# print("Experiment 2")
# gamma = 0.99
# theta = 0.000001
# environment = gym.make('MountainCar-v0')
# environment.reset()
# # policy, utility = value_iteration(environment,gamma, theta)
# pol, itr = policy_iteration(environment,gamma)
# wins, total_reward, average_reward = play_episodes(environment, n_episodes, policy)
# print("Results - {} , {}, {}, {}".format(wins, total_reward, average_reward, itr))



























# Define Q-learning function
def QLearning(env, learning, discount, epsilon, decay, episodes,  exploration="linear-decay"):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([100, 1000])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []



    # If exploration == "greedy" then epsilon needs to be 0
    if exploration == "greedy":
        epsilon = 0

    # Run Q learning algorithm
    for i in range(episodes):
        print("Episodes - {}".format(i))
        # Initialize parameters
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low) * np.array([100, 1000])
        state_adj = np.round(state_adj, 0).astype(int)
        count = 0
        while done is not True and count < 100000 :
            # print(count)
            count = count + 1
            # Render environment for last twenty episodes
            # if i >= (episodes - 20):
            #     # env.render()
            exp_exp_tradeoff = random.uniform(0, 1)
            # Determine next action - epsilon greedy strategy (if epsilon == 0 this is just a greedy exploration)
            if exp_exp_tradeoff > epsilon:
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
                # env.render()

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

        if exploration == "linear-decay":
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * i)
        elif exploration == "exp-decay":
            epsilon *= 1/2
            # elif exploration == "greedy" or exploration == "epsilon-greedy" epsilon mudt not change

        # Track rewards
        reward_list.append(tot_reward)

        ave_reward = np.mean(reward_list)
        ave_reward_list.append(ave_reward)
        reward_list = []


    env.close()

    return np.mean(ave_reward_list)


def QLearning_boltzmann(env, learning, discount, epsilon, min_eps, episodes, tau1, exploration="linear-decay"):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([100, 1000])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))
    exp_explr_strategy = BoltzmannPolicy(actions=range(env.action_space.n), tau=tau1, tau_decay=False)

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
        print("Episodes - {}".format(i))
        # Initialize parameters
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low) * np.array([100, 1000])
        state_adj = np.round(state_adj, 0).astype(int)
        count = 0
        while done is not True and count < 100000 :
            # print(count)
            count = count + 1
            # Render environment for last twenty episodes
            # if i >= (episodes - 20):
            #     # env.render()

            # Determine next action - epsilon greedy strategy (if epsilon == 0 this is just a greedy exploration)
            q_list = [Q[state_adj[0], state_adj[1], action1] for action1 in range(env.action_space.n)]
            action = exp_explr_strategy.compute_action(q_list)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low) * np.array([100, 1000])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # Allow for terminal states and render successful state
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                # env.render()

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
        print("Total Reward - {}".format(tot_reward))
        reward_list.append(tot_reward)


    env.close()

    return np.mean(reward_list)

print("Q Learning")
environment = MountainCarEnv()
environment.reset()
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

    environment.reset()
    avg_rewards_list = QLearning(environment, alpha, discount_factor, 0.04, 0, 1000, exploration="linear-decay")
    wins.append(avg_rewards_list)
    print("percentage victory {}".format(avg_rewards_list))


fig = plt.figure(figsize=(10, 5))
x = range(train_episodes)
plt.plot(alphas, wins)
plt.xlabel('Alpha')
plt.ylabel('Average Reward')
plt.title('Average Reward as Function of Alpha')
plt.savefig("rewardpercentagevsalpha.png")






print("Q Learning Discount Factor")
environment = MountainCarEnv()
environment.reset()
alpha = 1.0  # learning rate
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
discount_factor = 0.618
discount_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_episodes = 20000
test_episodes = 1000
max_steps = 1000
wins = []
for discount_factor in discount_factors:
    print("Working with discount_factor - {}".format(discount_factor))
    max_win = -99999.0

    environment.reset()
    avg_rewards_list = QLearning(environment, alpha, discount_factor, 0.04, 0, 1000, exploration="linear-decay")
    wins.append(avg_rewards_list)
    print("percentage victory {}".format(avg_rewards_list))


fig = plt.figure(figsize=(10, 5))
x = range(train_episodes)
plt.plot(discount_factors, wins)
plt.xlabel('Discount Factor')
plt.ylabel('Average Reward')
plt.title('Average Reward as Function of Discount Factor')
plt.savefig("dfpercentagevsalpha.png")








print("Q Learning BOLTZMann")
environment = MountainCarEnv()
environment.reset()
alpha = 1.0  # learning rate
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
discount_factor = 0.4
discount_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
taus = [10, 100, 1000, 10000, 100000]
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01
taus = [10, 100, 1000, 10000, 100000]
train_episodes = 20000
test_episodes = 1000
max_steps = 1000
wins = []
for tau in taus:
    print("Working with Tau - {}".format(tau))
    max_win = -99999.0

    environment.reset()
    avg_rewards_list = QLearning_boltzmann(environment, alpha, discount_factor, 0.04, 0, 200, tau, exploration="linear-decay")
    wins.append(avg_rewards_list)
    print("percentage victory {}".format(avg_rewards_list))

fig = plt.figure()
width = 0.35

x_pos = [i for i, _ in enumerate(taus)]

plt.bar(x_pos, wins, color='green')
plt.xlabel("Tau")
plt.ylabel("Rewards")
plt.title("Rewards as Function of Tau")

plt.xticks(x_pos, taus)

plt.savefig("rewardspercentagevstau_boltzman.png")




print("Q Learning Epsilon Decay")
environment = MountainCarEnv()
environment.reset()
alpha = 1.0  # learning rate
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
discount_factor = 0.4
discount_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
taus = [10, 100, 1000, 10000, 100000]
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01
decays = [0.0001, 0.001, 0.01, 0.1]
taus = [10, 100, 1000, 10000, 100000]
train_episodes = 20000
test_episodes = 1000
max_steps = 1000
wins = []
for decay in decays:
    print("Working with decay - {}".format(decay))
    max_win = -99999.0

    environment.reset()
    avg_rewards_list = QLearning(environment, alpha, discount_factor, 0.04, decay, 200, exploration="linear-decay")
    wins.append(avg_rewards_list)
    print("percentage victory {}".format(avg_rewards_list))

fig = plt.figure()
width = 0.35

x_pos = [i for i, _ in enumerate(decays)]

plt.bar(x_pos, wins, color='green')
plt.xlabel("Decay")
plt.ylabel("Rewards")
plt.title("Rewards as Function of Decay")

plt.xticks(x_pos, decays)

plt.savefig("rewardspercentagevsepisilon.png")