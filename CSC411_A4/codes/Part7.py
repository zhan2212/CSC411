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
    
def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def load_weights64(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt_64/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

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

def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data   

######################
##### Part 7
######################
trainNum = 200000
policy64 = Policy64()
env = Environment()

grid11 = np.array([])
grid12 = np.array([])
grid13 = np.array([])
grid21 = np.array([])
grid22 = np.array([])
grid23 = np.array([])
grid31 = np.array([])
grid32 = np.array([])
grid33 = np.array([])


for ep in range(10,trainNum,10):
    load_weights64(policy64, ep)
    firstMove = first_move_distr(policy64, env).tolist()
    #print(firstMove)
    grid11 = np.append(grid11,firstMove[0][0])
    grid12 = np.append(grid12,firstMove[0][1])
    grid13 = np.append(grid13,firstMove[0][2])
    grid21 = np.append(grid21,firstMove[0][3])
    grid22 = np.append(grid22,firstMove[0][4])
    grid23 = np.append(grid23,firstMove[0][5])
    grid31 = np.append(grid31,firstMove[0][6])
    grid32 = np.append(grid32,firstMove[0][7])
    grid33 = np.append(grid33,firstMove[0][8])

# plot the result 
fig = plt.figure()
plt.plot(range(10,trainNum,10),grid11.tolist(),'r',label='(1,1)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (1,1)')
plt.show()

fig = plt.figure()
plt.plot(range(10,trainNum,10),grid12.tolist(),'r',label='(1,2)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (1,2)')
plt.show()

fig = plt.figure()
plt.plot(range(10,trainNum,10),grid13.tolist(),'r',label='(1,3)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (1,3)')
plt.show()

fig = plt.figure()
plt.plot(range(10,trainNum,10),grid21.tolist(),'r',label='(2,1)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (2,1)')
plt.show()

fig = plt.figure()
plt.plot(range(10,trainNum,10),grid22.tolist(),'r',label='(2,2)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (2,2)')
plt.show()

fig = plt.figure()
plt.plot(range(10,trainNum,10),grid23.tolist(),'r',label='(2,3)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (2,3)')
plt.show()

fig = plt.figure()
plt.plot(range(10,trainNum,10),grid31.tolist(),'r',label='(3,1)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (3,1)')
plt.show()

fig = plt.figure()
plt.plot(range(10,trainNum,10),grid32.tolist(),'r',label='(3,2)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (3,2)')
plt.show()

fig = plt.figure()
plt.plot(range(10,trainNum,10),grid33.tolist(),'r',label='(3,3)')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.title('First Step Probability Distribution on grid (3,3)')
plt.show()
     
 
a = np.array([[np.mean(grid11[-1000:-1]),np.mean(grid12[-1000:-1]),np.mean(grid13[-1000:-1])],
              [np.mean(grid21[-1000:-1]),np.mean(grid22[-1000:-1]),np.mean(grid23[-1000:-1])],
              [np.mean(grid31[-1000:-1]),np.mean(grid32[-1000:-1]),np.mean(grid33[-1000:-1])]])

plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    