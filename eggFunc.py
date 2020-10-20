import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

#Initialize global variables/constants

np.random.seed(1337) # Set random seed

alpha = 0.05;     # Mutation probability
lbda = 100;     # Population and offspring size
k = 3;            # Tournament selection
intMax = 500;     # Boundary of the domain, not intended to be changed.

def plotPopulation(population):
	x = x = np.outer(np.linspace(0, intMax, 500), np.ones(500))
	y = x.copy().T
	F = -y*np.sin(np.sqrt(abs(x + y))) -x*np.sin(np.sqrt(abs(x - y)))
	(X,Y) = np.unravel_index(np.argmin(F, axis=None), F.shape)

	fig = plt.figure()
	ax = plt.axes(projection = '3d')

	ax.plot_surface(x,y,F, cmap = 'viridis', edgecolor = 'none', alpha = 0.5)

	ax.scatter(population[:,0],population[:,1], list(map(objf,population)), marker = '.', c = 'red')

	ax.scatter(X,Y, np.min(F), marker = 'v', c = 'green')
	ax.set_title('Surface Plot')
	plt.show()
	

#Compute the objective function at an (x,y) value.
def objf(x):
	sas = np.sqrt(abs(x[0]+x[1]))
	sad = np.sqrt(abs(x[0]-x[1]))
	f = -x[1]*np.sin(sas) - x[0]*np.sin(sad)
	return(f)


#Perform k-tournament selection to select pairs of parents.
#To do this we take k unique entries from 1-lmda, find the minimimum of those k
#and then select that minimum, we do this 2*lmda times
def selection(population, k):
	selected = np.zeros((2*lbda, 2))
	for ii in range(2*lbda):
		ri = np.random.choice(lbda,k)
		mi = np.argmin( objf(population[ri, :]) )
		selected[ii,:] = population[ri[mi],:]

	return(selected)

#Crossover operator to combine two 
def crossover(selected):
	weights = 3*np.random.rand(lbda,2) - 1	#Randomly distributed weights on interval U[-1,2]
	offspring = np.zeros((lbda, 2)) 	#Initialized list of offspring (empty)

	for ii in range(np.size(offspring,0)):
		offspring[ii,0] = min(intMax,max(0,selected[2*ii-1, 0] + weights[ii,0]*(selected[2*ii, 0]-selected[2*ii-1, 0])))
		offspring[ii,1] = min(intMax,max(0,selected[2*ii-1, 1] + weights[ii,1]*(selected[2*ii, 1]-selected[2*ii-1, 1])));

	return(offspring)

#Perform mutation, adding a random Gaussian perturbation.
def mutation(offspring):
	ii = np.asarray( np.random.rand(np.size(offspring,0),1) <= alpha ).nonzero()[0]
	offspring[ii,:] = offspring[ii,:] + 10*np.random.normal(len(ii),2)
	offspring[ii,0] = np.minimum(intMax, np.maximum(0, offspring[ii,0]))
	offspring[ii,1] = np.minimum(intMax, np.maximum(0, offspring[ii,1]))
	return(offspring)




#Eliminate the unfit candidate solutions.
def elimination(joinedPopulation, keep):
	fvals = list(map(objf,joinedPopulation))
	perm = np.argsort(fvals,0)
	survivors = joinedPopulation[perm[0:keep],:]
	return(survivors)

def eggholderEA():
  	#Initialize population this is a lbda x 2 vector
	population = intMax * np.random.rand(lbda, 2)
	selected = selection(population, k)

	#plot initial population
	plt.ion()
	x = x = np.outer(np.linspace(0, intMax, 500), np.ones(500))
	y = x.copy().T
	F = -y*np.sin(np.sqrt(abs(x + y))) -x*np.sin(np.sqrt(abs(x - y)))
	(X,Y) = np.unravel_index(np.argmin(F, axis=None), F.shape)

	fig = plt.figure()
	ax = plt.axes(projection = '3d')
	ax.set_title("Finding the Minima of The Eggholder Function")

	ax.plot_surface(x,y,F, cmap = 'viridis', edgecolor = 'none', alpha = 0.5)

	sc=ax.scatter(population[:,0],population[:,1], list(map(objf,population)), marker = 'o', c = 'red')
	ax.scatter(X,Y, np.min(F), marker = 'v', c = 'blue')

	plt.draw()

	for i in range(20):
    		# The evolutionary algorithm
		selected = selection(population, k);
		offspring = crossover(selected);
		joinedPopulation = np.concatenate((mutation(offspring),population))

		population = elimination(joinedPopulation, lbda)

		funcVals = list(map(objf,population))

		sc._offsets3d = (population[:,0], population[:,1], funcVals)
		fig.canvas.draw_idle()
		plt.pause(0.1)

    		# Show progress
		print('Mean fitness: ', np.mean(funcVals))
    
	#Show the result
	plt.waitforbuttonpress()

	print("POP0: ", population[0,:])
	print("minimum", objf(population[0,:]))

eggholderEA()
