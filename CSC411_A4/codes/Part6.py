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
    

######################
##### Part 6
######################
winR = np.array([])
tieR = np.array([])
loseR = np.array([])
trainNum = 200000

policy64 = Policy64()
for ep in range(200,trainNum,200):
    print(ep)
    load_weights64(policy64, ep) # load the file
    # count win/tie/lose games
    winCount = 0
    tieCount = 0
    loseCount = 0
    # play 100 games
    for i in range(100):
        game = Environment()
        #print('round %d'%i)
        state = game.reset()
        #game.render()
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
        
    winR = np.append(winR,winCount/100.0)
    tieR = np.append(tieR,tieCount/100.0)
    loseR = np.append(loseR,loseCount/100.0)
        
# plot the win/tie/lose rate
fig = plt.figure()
plt.plot(range(200,trainNum,200),winR.tolist(),'r',label='Win')
plt.plot(range(200,trainNum,200),tieR.tolist(),'b',label='Tie')
plt.plot(range(200,trainNum,200),loseR.tolist(),'g',label='Lose')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Probability')
plt.show() 