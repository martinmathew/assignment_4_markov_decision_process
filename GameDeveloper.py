import gym
import numpy as np
import matplotlib.pyplot as plt
from gym_gamblers.envs import GamblersEnv
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, show, colorbar





def argmax(env, V, pi, action, s, gamma):
    e = np.zeros(env.nA)
    for a in range(env.nA):  # iterate for every action possible
        q = 0
        P = np.array(env.P[s][a])
        (x, y) = np.shape(P)  # for Bellman Equation

        for i in range(x):  # iterate for every possible states
            s_ = int(P[i][1])  # S' - Sprime - possible succesor states
            p = P[i][0]  # Transition Probability P(s'|s,a)
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
        P = np.array(env.P[s][a])
        (x, y) = np.shape(P)

        for i in range(x):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            q += p * (r + gamma * V[s_])
            e[a] = q

    m = np.argmax(e)
    pi[s][m] = 1

    value = 0
    for a in range(env.nA):
        u = 0
        P = np.array(env.P[s][a])
        (x, y) = np.shape(P)
        for i in range(x):
            s_ = int(P[i][1])
            p = P[i][0]
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

    return V, pi, action  # optimal value funtion, optimal policy


def value_iteration(env, gamma, theta):

    V, pi, action = value_iteration_helper(env, gamma, theta)

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

    a = np.reshape(action, (8, 8))
    print(a)  # discrete action to take in given state

    e = 0
    for i_episode in range(100):
        c = env.reset()
        for t in range(10000):
            c, reward, done, info = env.step(action[c])
            if done:
                if reward == 1:
                    e += 1
                break
    print(" agent succeeded to reach goal {} out of 100 Episodes using this policy ".format(e + 1))
    env.close()

def execute(env, policy, gamma=1.0, render=False):
  start = env.reset()
  totalReward = 0
  stepIndex = 0
  while True:
    if render:
      env.render()
    start, reward, done,_ = env.step(int(policy[start]))
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
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
            policy[s] = np.argmax(q_sa)
    return policy

#Iteratively calculates value-function under policy
def CalcPolicyValue(env, policy, gamma=1.0, theta=0.0001):
  value = np.zeros(env.nS)

  while True:
    previousValue = np.copy(value)
    for states in range(env.nS):
      policy_a = policy[states]
      value[states] = sum([p * (r + gamma * previousValue[s_]) for p,s_, r, _ in env.P[states][policy_a]])
    if (np.sum((np.fabs(previousValue - value))) <= theta):
      break
      print( "breaked")
  return value


#Policy Iteration algorithm
def policyIteration(env, gamma=1.0,  theta = 0.1):
  policy = np.random.choice(env.nA, size=(env.nS))
  maxIterations = 1000
  policy_values = None
  for i in range(maxIterations):
    oldPolicyValue = CalcPolicyValue(env, policy, gamma, theta)
    newPolicy = extractPolicy(env,oldPolicyValue, gamma)
    if (np.all(policy == newPolicy)):
      print('Policy Iteration converged at %d' %(i+1))
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
  return policy



if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0')
    env = GamblersEnv()
    gamma = 0.99
    theta = 0.000001
    env.render()
    value_iteration(env, gamma, theta)
    policyIteration(env, gamma, theta)


# creating the bar plot



