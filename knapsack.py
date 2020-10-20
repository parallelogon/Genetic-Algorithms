"""
Evolutionary Algorithm to solve the binary knapsack problem.

given an object numbered 1,...,k.  This will be done as a binary string
each has a value, v, v[i] will be the value of object i corresponding to the binary list of objects
and a weight, w, w[i] is the weight of object i

The goal is to maxmize the value subject to a weight constraint, total weight shouldbe less than
W
"""
import numpy as np
import matplotlib.pyplot as plt

"""
The knapsack function will take a list of values and a list of weights,
each floating point numbers and
1) Initialize a population of size lambda
2) test fitness using k-tournament selection
3) reproduce using a crossover operator
4) mutate randomly with a fixed percentage probability
5) select a new generation
"""

#Initialize a knapsack class which keeps a list of values, weights, and a capacity
class Knapsack:
	def __init__(self,k):
		self.values = 2*np.random.rand(k,1)
		self.weights = 2*np.random.rand(k,1)
		self.capacity = 0.25 * (self.weights.T @ np.ones((k,1)))
		self.best = 0
		self.heurBest = np.zeros((k,1))
		for i,k in enumerate(self.values/self.weights):
			newval = self.best + k
			if newval <= self.capacity:
				self.best = newval
				self.heurBest[i] = 1
				
#fitness function to minimize, increases in value when items are high in weight decreases in value when they are high value
#Returns infinite if the packing is over capacity

def fitness(items, knp):
	weight = items @ knp.weights
	value = items @ knp.values
	caps = knp.capacity * np.ones(weight.shape)
	return(- value + 100000*knp.capacity*4*(weight >= caps))
	


#selection, which picks out 2*lambda individuals by performing k tournament selection
def selection(population, knp, lmbda, k):
	L = population.shape[1]
	selected = np.zeros((2*lmbda, L))
	for ii in range(2*lmbda):
		ri = np.random.choice(lmbda,k)
		mi = np.argmin( fitness(population[ri, :], knp) )
		selected[ii] = population[ri[mi],:]
		
	return(selected)


#Randomly mutate each bit in each string with probability alpha
def mutation(offspring, alpha):
	L,W = offspring.shape
	rolls = np.random.rand(L,W)
	offspring = (offspring + 1*(rolls <= alpha))%2
	return offspring

#Randomly combines two members of the population as a 50/50 mix of each parent
def crossover(selectedPopulation):
	L,W = selectedPopulation.shape
	p = np.random.rand(L,W)
	return(selectedPopulation[np.random.choice(L,L),:]*(p >= 0.5) + selectedPopulation*(p < 0.5))
	
	
#Keeps the most fit of the joined population of offspring + parents
def elimination(joinedPopulation, knp, keep):
	fvals = fitness(joinedPopulation, knp)
	perm = np.argsort(fvals,0)
	survivors = joinedPopulation[np.reshape(perm[0:keep], keep),:]
	return(survivors)

def knapsackProblem(k):
	alpha = 0.1
	lmbda = 10*k
	knap = Knapsack(k)
	
	pop = 1.0*(np.random.randn(lmbda, k) >= 0.5)
	bestFOld = 100
	bestF = 10000
	count = 0
	
	while(abs(bestFOld - bestF) >= 0.001):
	#for i in range(20):
		selected = selection(pop, knap, lmbda, 5)
		offspring = crossover(selected)
		joinedPopulation = np.concatenate((mutation(offspring, alpha),pop))
		pop = elimination(joinedPopulation, knap, lmbda)
		best = pop[0,:]
		bestFOld = bestF
		bestF = fitness(best,knap)
		count += 1
		print(count, bestF)
	
	print(np.hstack((pop, pop @ knap.weights, pop @ knap.values / (knap.values.T @ np.ones((k,1))))), "\n\nCapacity: ", knap.capacity)
	print("HeurBest: ", knap.heurBest.T @ knap.values, "\nFoundBest: ", pop[0,:] @ knap.values )
	
knapsackProblem(1000)
