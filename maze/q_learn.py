import numpy as np
import itertools as itr
import pprint
from functools import reduce

pp = pprint.PrettyPrinter()

class MDP:
	def __init__(self, init, actlist, terminals, gamma=0.9):
		self.init = init
		self.actlist = actlist
		self.terminals = terminals
		self.gamma = gamma
		self.states = set()
		self.reward = {}

	def R(self, state):
		return self.reward[state]

	def T(self, state, action):
		raise NotImplementedError

	def actions(self, state):
		if state in self.terminals:
			return [None]
		else:
			return self.actlist

class MazeMDP(MDP):
  def __init__(self, init, gamma=0.9):
    self.size = (4, 4)
    actlist = ((0,1), (1,0), (0,-1), (-1,0))
    terminals = [(self.size[0]-1, self.size[1]-1)]
    MDP.__init__(self, init, actlist, terminals, gamma)

  def move(self, state, act):
    state1 = (state[0] + act[0], state[1] + act[1])
    
    if state1[0] < 0 or state1[1] < 0 or self.size[0]-1 < state1[0] or self.size[1]-1 < state1[1]:
      return state
    else:
      return state1

  def T(self, state, act):
    return [(1.0, self.move(state, act))]
  
  def R(self, state, act, state1):
    if state1 in self.terminals:
      return 100
    else:
      return -10

class QTable:
  def __init__(self):
    self.table = {}
  
  def safe_init(self, s, a):
    if not s in self.table:
      self.table[s] = {}
    
    if not a in self.table[s]:
      self.table[s][a] = np.random.rand(1,1)[0][0]
  
  def get(self, s, a):
    self.safe_init(s,a)
    return self.table[s][a]
  
  def set(self, s, a, r):
    self.safe_init(s, a)
    self.table[s][a] = r
  
  def act_max(self, s, actlist):
    return max(actlist, key = lambda a: self.get(s, a))
  
  def reward_max(self, s, actlist):
    return self.get(s, self.act_max(s, actlist))

def q_learning(mdp, iteration):
  q = QTable()
  
  alpha = 0.5
  
  for i in range(iteration):
    if i%1000 == 0: print(i)
    state = mdp.init
    
    while not state in mdp.terminals:
      if np.random.rand(1,1)[0][0] < 0.3:
        ai = np.random.choice(len(mdp.actlist), 1)[0]
        act = mdp.actlist[ai]
      else:
        act = q.act_max(state, mdp.actlist)
      
      transition = mdp.T(state, act)
      t_index = np.random.choice(len(transition), 1, p=[t[0] for t in transition])
      state1 = transition[t_index][1]
      reward = mdp.R(state, act, state1)
      
      q.set(state, act, (1-alpha) * q.get(state, act) + alpha*(reward + mdp.gamma*q.reward_max(state1, mdp.actlist)))
      state = state1
  
  return q.table

mdp = MazeMDP((0,0))
q_table = q_learning(mdp, 10000)

pp.pprint(q_table)
