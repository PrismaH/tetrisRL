# -*- coding: utf-8 -*-
import math
import random
import sys
import os
import shutil
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=UserWarning)
    import torch
    
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CyclicLR
from ranger import Ranger

from engine import TetrisEngine

width, height = 10, 20 # standard tetris friends rules
engine = TetrisEngine(width, height)

# set up matplotlib
#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
    #from IPython import display

#plt.ion()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
if use_cuda:print("....Using Gpu...")
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
#Tensor = FloatTensor


######################################################################
# Replay Memory
# -------------
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class MBConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, expand_ratio, stride=1, padding=1):
        super(MBConv, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.stride = stride

        self.conv1 = nn.Conv2d(channels_in, channels_in*expand_ratio, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_in*expand_ratio)

        self.conv2 = nn.Conv2d(channels_in*expand_ratio, channels_in*expand_ratio, groups=channels_in*expand_ratio, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_in*expand_ratio)

        self.conv3 = nn.Conv2d(channels_in*expand_ratio, channels_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels_out)

        self.mish = Mish()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mish(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mish(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.channels_in == self.channels_out and self.stride == 1:
            x = x + shortcut
        
        return x

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        #activ func
        self.mish = Mish()

        #First Conv
        channels_in = 1
        self.conv1 = nn.Conv2d(channels_in, int(round(32 * 1.4)), kernel_size=3, stride=2, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(int(round(32 * 1.4)))

        self.layer1 = self.build_block(channels_in=32, channels_out=16, kernel_size=3, depth=1, stride=1, expand_ratio=1, padding=1)
        self.layer2 = self.build_block(channels_in=16, channels_out=24, kernel_size=3, depth=2, stride=2, padding=1)
        self.layer3 = self.build_block(channels_in=24, channels_out=40, kernel_size=5, depth=2, stride=1, padding=2)
        self.layer4 = self.build_block(channels_in=40, channels_out=80, kernel_size=3, depth=3, stride=1, padding=1)
        self.layer5 = self.build_block(channels_in=80, channels_out=112, kernel_size=5, depth=3, stride=1, padding=2)
        self.layer6 = self.build_block(channels_in=112, channels_out=192, kernel_size=5, depth=4, stride=2, padding=2)
        self.layer7 = self.build_block(channels_in=192, channels_out=320, kernel_size=3, depth=1, stride=1, padding=1)

        self.conv2 = nn.Conv2d(int(round(320 * 1.4)), int(round(1280 * 1.4)), kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(round(1280 * 1.4)))

        self.fc1 = nn.Linear(int(round(1280 * 1.4)), engine.nb_actions)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def build_block(self, channels_in, channels_out, kernel_size, depth, stride, expand_ratio=6, padding=1):
        block_list = []
        for _ in range(int(round(depth * 1.8))):
            block_list.append(MBConv(int(round(channels_in * 1.4)), int(round(channels_out * 1.4)), kernel_size=kernel_size, expand_ratio=expand_ratio, stride=stride, padding=padding))
            channels_in = channels_out
            stride = 1
        return nn.Sequential(*block_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.fc1(x.view(x.size(0), -1))

        return x

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``Variable`` - this is a simple wrapper around
#    ``torch.autograd.Variable`` that will automatically send the data to
#    the GPU every time we construct a Variable.
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
#

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
CHECKPOINT_FILE = 'checkpoint.pth.tar'


steps_done = 0

model = DQN()
print(model)

if use_cuda:
    model.cuda()

loss = nn.MSELoss()
optimizer = Ranger(model.parameters(), lr=.001)
scheduler = CyclicLR(optimizer, base_lr=0.1, max_lr=0.6, mode='triangular', cycle_momentum=False)
memory = ReplayMemory(3000)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(state, requires_grad=False).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return FloatTensor([[random.randrange(engine.nb_actions)]])


episode_durations = []


'''
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
'''


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state.


last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]))
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
    with torch.no_grad():
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    if len(loss.data.size())>0 : return loss.data[0] 
    else : return loss

def optimize_supervised(pred, targ):
    optimizer.zero_grad()

    diff = loss(pred, targ)
    diff.backward()
    optimizer.step()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    try: # If these fail, its loading a supervised model
        optimizer.load_state_dict(checkpoint['optimizer'])
        memory = checkpoint['memory']
    except Exception as e:
        pass
    # Low chance of random action
    #steps_done = 10 * EPS_DECAY

    return checkpoint['epoch'], checkpoint['best_score']

if __name__ == '__main__':
    # Check if user specified to resume from a checkpoint
    start_epoch = 0
    best_score = 0
    if len(sys.argv) > 1 and sys.argv[1] == 'resume':
        if len(sys.argv) > 2:
            CHECKPOINT_FILE = sys.argv[2]
        if os.path.isfile(CHECKPOINT_FILE):
            print("=> loading checkpoint '{}'".format(CHECKPOINT_FILE))
            start_epoch, best_score = load_checkpoint(CHECKPOINT_FILE)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(CHECKPOINT_FILE, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(CHECKPOINT_FILE))

    ######################################################################
    #
    # Below, you can find the main training loop. At the beginning we reset
    # the environment and initialize the ``state`` variable. Then, we sample
    # an action, execute it, observe the next screen and the reward (always
    # 1), and optimize our model once. When the episode ends (our model
    # fails), we restart the loop.

    f = open('log.out', 'w+')
    for i_episode in count(start_epoch):
        # Initialize the environment and state
        state = FloatTensor(engine.clear()[None,None,:,:])

        score = 0
        for t in count():
            # Select and perform an action
            action = select_action(state).type(LongTensor)

            # Observations
            last_state = state
            state, reward, done = engine.step(action[0,0])
            state = FloatTensor(state[None,None,:,:])
            
            # Accumulate reward
            score += int(reward)

            reward = FloatTensor([float(reward)])
            # Store the transition in memory
            memory.push(last_state, action, state, reward)

            # Perform one step of the optimization (on the target network)
            if done:
                # Train model
                if i_episode % 10 == 0:
                    log = 'epoch {0} score {1}'.format(i_episode, score)
                    print(log)
                    f.write(log + '\n')
                    loss = optimize_model()
                    print('loss: {}'.format(loss))
                # Checkpoint
                if i_episode % 100 == 0:
                    is_best = True if score > best_score else False
                    save_checkpoint({
                        'epoch' : i_episode,
                        'state_dict' : model.state_dict(),
                        'best_score' : best_score,
                        'optimizer' : optimizer.state_dict(),
                        'memory' : memory
                        }, is_best)
                break

    f.close()
    print('Complete')
    #env.render(close=True)
    #env.close()
    #plt.ioff()
    #plt.show()
