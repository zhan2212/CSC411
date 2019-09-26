from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done
    
class Policy128(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=128, output_size=9):
        super(Policy128, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,output_size),
                nn.Softmax()
                )
        
    def forward(self, x):
        y = self.net(x)
        return y

class Policy64(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy64, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,output_size),
                nn.Softmax()
                )
        
    def forward(self, x):
        y = self.net(x)
        return y
    
class Policy32(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=32, output_size=9):
        super(Policy32, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,output_size),
                nn.Softmax()
                )
        
    def forward(self, x):
        y = self.net(x)
        return y

class Policy16(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=16, output_size=9):
        super(Policy16, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size,hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,output_size),
                nn.Softmax()
                )
        
    def forward(self, x):
        y = self.net(x)
        return y

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    result = np.zeros(len(rewards))
    for i in range(len(rewards)):
        val = 0.0
        discount = 1.0
        for j in range(i,len(rewards)):
            val += discount * rewards[j]
            discount *= gamma
        result[i] = val
    return result

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0, # TODO
            Environment.STATUS_INVALID_MOVE: -5000,
            Environment.STATUS_WIN         : 10,
            Environment.STATUS_TIE         : 5,
            Environment.STATUS_LOSE        : -10
    }[status]

def train(policy, env, gamma=0.9, log_interval=10, max_episode = 50000, policyIndx = 64):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    
    rewards = np.array([])
    invalid = np.array([])
    moveCount = 0.0
    invalidCount = 0.0
    
    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            moveCount += 1.0
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            if reward == -5000:
                invalidCount += 1.0
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)
        
        
        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)
            

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            rewards = np.append(rewards,running_reward / log_interval)
            invalid = np.append(invalid, invalidCount/moveCount)
            print(invalidCount/moveCount)
            running_reward = 0
            invalidCount = 0.0
            moveCount = 0.0

        
        if i_episode % (log_interval) == 0 and policyIndx == 128:
            torch.save(policy.state_dict(),
                       "ttt_128/policy-%d.pkl" % i_episode)
            
        if i_episode % (log_interval) == 0 and policyIndx == 64:
            torch.save(policy.state_dict(),
                       "ttt_64/policy-%d.pkl" % i_episode)
        
        if i_episode % (log_interval) == 0 and policyIndx == 32:
            torch.save(policy.state_dict(),
                       "ttt_32/policy-%d.pkl" % i_episode)
            
        if i_episode % (log_interval) == 0 and policyIndx == 16:
            torch.save(policy.state_dict(),
                       "ttt_16/policy-%d.pkl" % i_episode)
            

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if i_episode == max_episode:
            return rewards, invalid


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt_64/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)
    
def load_weights32(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt_32/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

def load_weights16(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt_16/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

def load_weights128(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt_128/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)    


trainNum = 200000

# Using 64 layers(default)
# Initialize the environment
policy128 = Policy128()
env4 = Environment()
# Train the model
rewards4,invalid4 = train(policy128, env4, max_episode = trainNum, policyIndx = 128)

# Using 64 layers(default)
# Initialize the environment
policy64 = Policy64()
env = Environment()
# Train the model
rewards,invalid = train(policy64, env, max_episode = trainNum, policyIndx = 64)


# Using 32 layers
# Initialize the environment
policy32 = Policy32()
env2 = Environment()
# Train the model
rewards2,invalid2 = train(policy32, env2, max_episode = trainNum, policyIndx = 32)


# Using 16 layers
# Initialize the environment
policy16 = Policy16()
env3 = Environment()
# Train the model
rewards3,invalid3 = train(policy16, env3, max_episode = trainNum, policyIndx = 16)


######################
##### Part 5 (a)
######################
# smooth the data
rewardPlot = np.array([])
for i in range(len(rewards)):
    if i % 100 == 0:
        rewardPlot = np.append(rewardPlot,np.mean(rewards[i:i+100]))

# plot the performance
fig = plt.figure()
plt.plot(range(0,trainNum,1000),rewardPlot.tolist(),'r',label='Win')
plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.title('The Training Curve')
plt.legend(loc='lower right')
plt.show()


######################
##### Part 5 (b)
######################
# Initialize the environment
policy128 = Policy128()
load_weights128(policy128, trainNum) # load the file

# count win/tie/lose games
winCount = 0
tieCount = 0
loseCount = 0

# play 100 games
for i in range(10000):
    game = Environment()
    state = game.reset()
    done = False
    while not done:
        action, logprob = select_action(policy128, state)
        state, status, done = game.play_against_random(action)
        
    if status == 'win':
        winCount += 1
    elif status == 'tie':
        tieCount += 1
    else:
        loseCount += 1
        
# print the result     
print('128 Hidden Layers:')
print('The number of win games out of 10000 games is: %d'%winCount)
print('The number of tie games out of 10000 games is: %d'%tieCount)
print('The number of lose games out of 10000 games is: %d'%loseCount)
print('\n')

# Initialize the environment
policy64 = Policy64()
load_weights(policy64, trainNum) # load the file

# count win/tie/lose games
winCount = 0
tieCount = 0
loseCount = 0

# play 100 games
for i in range(10000):
    game = Environment()
    state = game.reset()
    done = False
    while not done:
        action, logprob = select_action(policy64, state)
        state, status, done = game.play_against_random(action)
        
    if status == 'win':
        winCount += 1
    elif status == 'tie':
        tieCount += 1
    else:
        loseCount += 1
        
# print the result     
print('64 Hidden Layers:')
print('The number of win games out of 10000 games is: %d'%winCount)
print('The number of tie games out of 10000 games is: %d'%tieCount)
print('The number of lose games out of 10000 games is: %d'%loseCount)
print('\n')


# Initialize the environment
policy32 = Policy32()
load_weights32(policy32, trainNum) # load the file

# count win/tie/lose games
winCount = 0
tieCount = 0
loseCount = 0

# play 100 games
for i in range(10000):
    game = Environment()
    state = game.reset()
    done = False
    while not done:
        action, logprob = select_action(policy32, state)
        state, status, done = game.play_against_random(action)
        
    if status == 'win':
        winCount += 1
    elif status == 'tie':
        tieCount += 1
    else:
        loseCount += 1
        
# print the result
print('32 Hidden Layers:')
print('The number of win games out of 10000 games is: %d'%winCount)
print('The number of tie games out of 10000 games is: %d'%tieCount)
print('The number of lose games out of 10000 games is: %d'%loseCount)
print('\n')

# Initialize the environment
policy16 = Policy16()
load_weights16(policy16, trainNum) # load the file

# count win/tie/lose games
winCount = 0
tieCount = 0
loseCount = 0

# play 100 games
for i in range(10000):
    game = Environment()
    state = game.reset()
    done = False
    while not done:
        action, logprob = select_action(policy16, state)
        state, status, done = game.play_against_random(action)
        
    if status == 'win':
        winCount += 1
    elif status == 'tie':
        tieCount += 1
    else:
        loseCount += 1
        
# print the result  
print('16 Hidden Layers:')
print('The number of win games out of 10000 games is: %d'%winCount)
print('The number of tie games out of 10000 games is: %d'%tieCount)
print('The number of lose games out of 10000 games is: %d'%loseCount)
print('\n')

rewardPlot = np.array([])
for i in range(len(rewards)):
    if i % 100 == 0:
        rewardPlot = np.append(rewardPlot,np.mean(rewards[i:i+100]))

rewardPlot2 = np.array([])
for i in range(len(rewards2)):
    if i % 100 == 0:
        rewardPlot2 = np.append(rewardPlot2,np.mean(rewards2[i:i+100]))
        
rewardPlot3 = np.array([])
for i in range(len(rewards3)):
    if i % 100 == 0:
        rewardPlot3 = np.append(rewardPlot3,np.mean(rewards3[i:i+100]))

rewardPlot4 = np.array([])
for i in range(len(rewards4)):
    if i % 100 == 0:
        rewardPlot4 = np.append(rewardPlot4,np.mean(rewards4[i:i+100]))

# plot the performance
fig = plt.figure()
plt.plot(range(0,trainNum,1000),rewardPlot4.tolist(),'y',label='128')
plt.plot(range(0,trainNum,1000),rewardPlot.tolist(),'r',label='64')
plt.plot(range(0,trainNum,1000),rewardPlot2.tolist(),'g',label='32')
plt.plot(range(0,trainNum,1000),rewardPlot3.tolist(),'b',label='16')
plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.title('The Training Curve')
plt.legend(loc='lower right')
plt.show()



######################
##### Part 5 (c)
######################
# smooth the data
invalidPlot = np.array([])
for i in range(len(invalid)):
    if i % 100 == 0:
        invalidPlot = np.append(invalidPlot,np.mean(invalid[i:i+100]))
        
# plot the performance
fig = plt.figure()
plt.plot(range(0,trainNum,1000),invalidPlot.tolist(),'r',label='Percentage of Invalid Moves')
plt.xlabel('Episodes')
#plt.xlim(0,15000)
plt.ylabel('Percentage of Invalid Moves')
plt.title('Invalid Moves')
plt.legend(loc='upper right')
plt.show()


######################
##### Part 5 (d)
######################
# Initialize the environment
policy64 = Policy64()
load_weights(policy64, trainNum) # load the file

# count win/tie/lose games
winCount = 0
tieCount = 0
loseCount = 0

# play 100 games
for i in range(100):
    game = Environment()
    #print('round %d'%i)
    state = game.reset()
    #if i < 5:
     #   print('Game: %d'%(i+1))
      #  game.render()
    done = False
    while not done:
        action, logprob = select_action(policy64, state)
        state, status, done = game.play_against_random(action)
       # game.render()
        #if i < 5:
          #  game.render()
        
    if status == 'win':
        winCount += 1
    elif status == 'tie':
        tieCount += 1
    else:
        loseCount += 1
        #ame.render()
        
# print the result        
print('The number of win games out of 100 games is: %d'%winCount)
print('The number of tie games out of 100 games is: %d'%tieCount)
print('The number of lose games out of 100 games is: %d'%loseCount)
print('\n')






