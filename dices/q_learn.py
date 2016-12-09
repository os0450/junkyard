import numpy as np
import itertools as itr
import pprint
from functools import reduce
import pickle

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

def roll(n):
	return tuple(np.bincount(np.random.randint(0,6,n), minlength=6))

class DiceMDP(MDP):
  def __init__(self, gamma=0.9):
    MDP.__init__(self, None, None, None, gamma)
  
  def is_terminate(self, state):
    return not any(a is not None for a in self.available_actlist(state))
  
  def T(self, state, act):
		if act is None or self.is_terminate(state):
			return [(0.0, state)]
    
		rds, eds = map(list, state)
		eds[act] = rds[act]
		rds[act] = 0
		
		nd = reduce(lambda a,b:a+b, rds, 0)
		counts = {}
		total = 0
		dvs = (0,1,2,3,4,5)
		for c in  itr.product(dvs, repeat=nd):
			dvCount = tuple(np.bincount(c, minlength=6))
			
			nextState = (dvCount, tuple(eds))
			
			if nextState not in counts:
				counts[nextState] = 0
			
			counts[nextState] += 1
			total += 1
		
		def toProb(x):
			return float(x)/float(total)
		
		probs = {vs:toProb(x) for vs,x in counts.items()}
		
		return list(map(lambda x: (probs[x], x), probs.keys()))
  
  def next_state(self, state, act):
		if act is None or self.is_terminate(state):
			return state
    
		rds, eds = map(list, state)
		eds[act] = rds[act]
		rds[act] = 0
		
		return (roll(sum(rds)), tuple(eds))
  
  def R(self, state, act, state1):
    if all([r == 0 for r in state1[0]]):
      return sum([e * (i+1) for e,i in zip(state1[1], range(6))])
    
    return 0
  
  def available_actlist(self, state):
    aa = [i for r,e,i in zip(state[0], state[1], range(6)) if e == 0 and r != 0]
    
    if len(aa) == 0:
      return [None]
    else:
      return aa

class QTable:
  def __init__(self, table = {}):
    self.table = table
  
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
    if i % 1000 == 0: print(i)
    state = (roll(2), (0,0,0,0,0,0))
    
    while not mdp.is_terminate(state):
      actlist = mdp.available_actlist(state)
      
      if np.random.rand(1,1)[0][0] < 0.3:
        ai = np.random.choice(len(actlist), 1)[0]
        act = actlist[ai]
      else:
        act = q.act_max(state, actlist)
      
      state1 = mdp.next_state(state, act)
      reward = mdp.R(state, act, state1)
      
      q.set(state, act, (1-alpha) * q.get(state, act) + alpha*(reward + mdp.gamma*q.reward_max(state1, mdp.available_actlist(state1))))
      state = state1
      
  
  return q.table

if __name__ == '__main__':
	mdp = DiceMDP()
	q_table = q_learning(mdp, 10000000)
	
	with open('q_lean.pick', mode='wb') as f:
		pickle.dump(q_table, f)
