import pickle
from q_learn import DiceMDP, roll, QTable

with open('policy.pick', 'rb') as f:
	pi = pickle.load(f)

mdp = DiceMDP()
rewards = []
for i in range(100000):
	state = (roll(2), (0,0,0,0,0,0))
	while not mdp.is_terminate(state):
		act = pi[state]
		state = mdp.next_state(state, act)
	
	rewards.append(mdp.R(None, None, state))

print(float(sum(rewards))/float(len(rewards)))
