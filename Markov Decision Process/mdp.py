# Markov Decision Process to automate my handwritten homework :)
# https://inst.eecs.berkeley.edu/~cs188/sp10/written/w4.pdf

states = ['PoorHungry', 'PoorFull', 'RichHungry', 'RichFull']
actions = ['Play', 'Eat']

def transition_model(s, a, s_prime):
	if s == 'PoorHungry' and a == 'Play' and s_prime == 'PoorHungry':
		return 0.8
	if s == 'PoorHungry' and a == 'Play' and s_prime == 'RichHungry':
		return 0.2
	if s == 'PoorHungry' and a == 'Eat' and s_prime == 'PoorHungry':
		return 0.8
	if s == 'PoorHungry' and a == 'Eat' and s_prime == 'PoorFull':
		return 0.2

	if s == 'PoorFull' and a == 'Play' and s_prime == 'PoorFull':
		return 0.5
	if s == 'PoorFull' and a == 'Play' and s_prime == 'RichFull':
		return 0.5
	if s == 'RichHungry' and a == 'Eat' and s_prime == 'RichHungry':
		return 0.2
	if s == 'RichHungry' and a == 'Eat' and s_prime == 'RichFull':
		return 0.8
	return 0

def reward_function(s_prime):
	if s_prime == 'PoorHungry':
		return -1
	if s_prime == 'PoorFull':
		return 1
	if s_prime == 'RichHungry':
		return 0
	if s_prime == 'RichFull':
		return 5

def q_star(s, a, prev):
	sum = 0
	for state_prime in states:
		sum += transition_model(s, a, state_prime) * (reward_function(state_prime) + prev)
	return sum

def v_star(s, prev):
	return max(q_star(s, actions[0], prev), q_star(s, actions[1], prev))

print '*********** i = 1 ***********'
print v_star('PoorHungry', 0)
print v_star('PoorFull', 0)
print v_star('RichHungry', 0)

print '*********** i = 2 ***********'
print v_star('PoorHungry', -0.6)
print v_star('PoorFull', 3.0)
print v_star('RichHungry', 4.0)

print '*********** i = 3 ***********'
print v_star('PoorHungry', -1.2)
print v_star('PoorFull', 6.0)
print v_star('RichHungry', 8.0)

