 
import numpy as np

import matplotlib.pyplot as plt



def getAcc( position, mass, G, softening ):
	
	# positions r = [x,y,z] for all particles
	x_par = position[:,0:1]
	y_par = position[:,1:2]
	z_par = position[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x_par.T - x_par
	dy = y_par.T - y_par
	dz = z_par.T - z_par

	# matrix that stores 1/r^3 for all particle pairwise particle separations 
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
	inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))

	return a
	
def getEnergy( position, velocity, mass, G ):
	
	# Kinetic Energy:
	KE = 0.5 * np.sum(np.sum( mass * velocity**2 ))


	# Potential Energy:

	# positions r = [x,y,z] for all particles
	x_par = position[:,0:1]
	y_par = position[:,1:2]
	z_par = position[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x_par.T - x_par
	dy = y_par.T - y_par
	dz = z_par.T - z_par

	# matrix that stores 1/r for all particle pairwise particle separations 
	inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
	inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

	# sum over upper triangle, to count each interaction only once
	PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
	
	return KE, PE;


def main():

	
	# Simulation parameters
	N         = 10  # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 100.0   # time at which simulation ends
	dt        = 0.01   # timesteps
	softening = 0.1    # trail length
	G         = 1.0    # Grav constant in some system
	plotRealTime = True 
	
	
	np.random.seed(17)            # set the random number generator seed
	
	mass = 20.0*np.ones((N,1))/N  
	position  = np.random.randn(N,3)   # random selection of positions and velocities for the particles
	velocity  = np.random.randn(N,3)
	
	# Convert to Center-of-Mass frame
	velocity -= np.mean(mass * velocity,0) / np.mean(mass)
	
	# calculate initial gravitational accelerations
	acc = getAcc( position, mass, G, softening )
	
	# calculate initial energy of system
	KE, PE  = getEnergy( position, velocity, mass, G )
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# save energies, particle orbits for plotting trails
	position_save = np.zeros((N,3,Nt+1))
	position_save[:,:,0] = position
	KE_save = np.zeros(Nt+1)
	KE_save[0] = KE
	PE_save = np.zeros(Nt+1)
	PE_save[0] = PE
	t_all = np.arange(Nt+1)*dt
	
	# prep figure
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])
	
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		velocity += acc * dt/2.0
		
		# drift
		position += velocity * dt
		
		# update accelerations
		acc = getAcc( position, mass, G, softening )
		
		# (1/2) kick
		velocity += acc * dt/2.0
		
		# update time
		t += dt
		
		# get energy of system
		KE, PE  = getEnergy( position, velocity, mass, G )
		
		# save energies, positions for plotting trail
		position_save[:,:,i+1] = position
		KE_save[i+1] = KE
		PE_save[i+1] = PE
		
		# plot in real time
		if plotRealTime or (i == Nt-1):
			plt.sca(ax1)
			plt.cla()
			xx = position_save[:,0,max(i-50,0):i+1]
			yy = position_save[:,1,max(i-50,0):i+1]
			plt.scatter(xx,yy,s=1,color=[.7,.7,1])
			plt.scatter(position[:,0],position[:,1],s=10,color='blue', edgecolors='black')
			ax1.set(xlim=(-2, 2), ylim=(-2, 2))
			ax1.set_aspect('equal', 'box')
			ax1.set_xticks([-2,-1,0,1,2])
			ax1.set_yticks([-2,-1,0,1,2])
			
			plt.sca(ax2)
			plt.cla()
			plt.scatter(t_all,KE_save,color='red',s=1,label='KE' if i == Nt-1 else "")
			plt.scatter(t_all,PE_save,color='blue',s=1,label='PE' if i == Nt-1 else "")
			plt.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot' if i == Nt-1 else "")
			ax2.set(xlim=(0, tEnd), ylim=(-300, 300))
			ax2.set_aspect(0.007)
			
			plt.pause(0.001)
	    
	
	
	# add labels/legend
	plt.sca(ax2)
	plt.xlabel('time')
	plt.ylabel('energy')
	ax2.legend(loc='upper right')
	
	# Save figure
	plt.savefig('nbody.png',dpi=240)
	plt.show()
	    
	return 0
	


  
main()
  