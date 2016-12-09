import pickle
from q_learn import DiceMDP, roll, QTable

with open('q_learn.pick', 'rb') as f:
	q = pickle.load(f)

mdp = DiceMDP()
q_table = QTable(q)

rewards = []
for i in range(100000):
	state = (roll(2), (0,0,0,0,0,0))
	while not mdp.is_terminate(state):
		act = q_table.act_max(state, mdp.available_actlist(state))
		state = mdp.next_state(state, act)
	
	rewards.append(mdp.R(None, None, state))

print(float(sum(rewards))/float(len(rewards)))
