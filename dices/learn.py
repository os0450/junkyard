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

class DiceMDP(MDP):
	def __init__(self, states, init, terminals, reward, gamma=0.9):
		actlist = (None, 0,1,2,3,4,5)
		MDP.__init__(self, init, actlist, terminals, gamma = gamma)
		
		self.states = states
		self.reward = reward
	
	def R(self, state):
		return self.reward[state] if state in self.reward else -0.0
	
	def T(self, state, action):
		if action is None or state in (State.Invalid, State.Burst):
			return [(0.0, state)]
		
		if state[0][action] == 0 or state[1][action] != 0:
			return [(1.0, State.Invalid)]
		
		if State.is_burst(state[0], state[1]):
			return [(1.0, State.Burst)]

		rds, eds = map(list, state)
		eds[action] = rds[action]
		rds[action] = 0
		
		nd = reduce(lambda a,b:a+b, rds, 0)
		counts = {}
		total = 0
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

class State:
	Invalid = 0
	Burst = 1
	
	@classmethod
	def all(cls):
		states = set((cls.Invalid, cls.Burst))
		for e in range(0,num_dice+1):
			r = num_dice-e
			for rs in itr.combinations_with_replacement(dvs,r):
				for es in itr.combinations_with_replacement(dvs,e):
					states.add((tuple(np.bincount(rs, minlength=6)), tuple(np.bincount(es, minlength=6))))
		
		return states
	
	@classmethod
	def is_burst(cls, nds, eds):
		return not any([a != 0 and b == 0 for (a,b) in zip(nds, eds)])

	@classmethod
	def terminals(cls):
		terminals = set()
		for state in cls.state():
			None		

def roll(n):
	return tuple(np.bincount(np.random.randint(0,6,n), minlength=6))

def value_iteration(mdp, epsilon=0.001):
	U1 = {s:0 for s in mdp.states};
	R, T, gamma = mdp.R, mdp.T, mdp.gamma
	
	while True:
		U = U1.copy()
		delta = 0
		cnt = 0
		for s in mdp.states:
			U1[s] = R(s) + gamma * max([sum([p*U[s1] for (p,s1) in T(s, a)]) for a in mdp.actions(s)])
			delta = max(delta, abs(U1[s]-U[s]))
			
			cnt += 1
			print(cnt) if cnt%100 == 0 else None
		
		print("delta:" + str(delta))
		if delta < epsilon * (1-gamma)/gamma:
			return U

def best_policy(mdp, U):
	pi = {}
	for s in mdp.states:
		pi[s] = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))

	return pi

def expected_utility(a, s, U, mdp):
	return sum([p * U[s1] for (p,s1) in mdp.T(s,a)])

def policy_iteration(mdp, epsilon=0.001):
  pi = {s:0 for s in mdp.states};
  R, T, gamma = mdp.R, mdp.T, mdp.gamma
	
  list_states = list(mdp.states)
  num_states = len(mdp.states)
  indices = {}
  idx = 0
  for s1 in list_states:
    indices[s1] = idx
    idx += 1

  def probability(s, a, s1):
    transients = T(s, a)
    t = next(filter(lambda t: t[1] == s1, transients), None)
    if t is None:
      return 0
    else:
      return t[0]
	
  def solve_value(pi):
    # tmp = np.array([[probability(s1, pi[s1], s2)*-gamma for s2 in list_states] for s1 in list_states])
    tmp = np.zeros((len(list_states), len(list_states)))

    for s1 in list_states:
      for transients in (T(s1, a) for a in mdp.actlist):
        for t in transients:
          tmp[indices[s1]][indices[t[1]]] = t[0]*-gamma
    
    for i in range(num_states):
      tmp[i][i] += 1
    
    r = np.array([R(s1) for s1 in list_states])
    
    x = np.linalg.solve(tmp, r)

    return dict((k, v) for k,v in zip(list_states, x))
  
  def v(s1, a):
    return R(s1)+gamma*sum([t[0]*gamma*V[t[1]] for t in T(s1, a)])
  
  for i in range(100):
    print(i)
    
    ppi = pi.copy()
    
    V = solve_value(pi)
    
    pi = {s1: max(mdp.actlist, key=lambda a: v(s1, a)) for s1 in mdp.states}
    
    if all([pi[s] == ppi[s] for s in mdp.states]):
      break

  return pi



num_dice = 2
empty_state = tuple([0]*6)

terminals = set()
reward = {}
dvs = (0,1,2,3,4,5)
for c in itr.product(dvs,repeat=num_dice):
	state = (empty_state, tuple(np.bincount(c, minlength=6)))
	terminals.add(state)
	reward[state] = sum(c)+num_dice

terminals.add(State.Invalid)
terminals.add(State.Burst)
reward[State.Invalid] = 0
reward[State.Burst] = 0


states = State.all()

mdp = DiceMDP(states, (roll(num_dice), empty_state), terminals, reward, gamma=0.99)

print("Start value iteration")
# pp.pprint(states)

pi = policy_iteration(mdp)

with open('policy.pick', 'wb') as f:
  pickle.dump(pi, f)

# U = value_iteration(mdp)
# pp.pprint(U)
# pi = best_policy(mdp, U)
# pp.pprint(pi)
