"""
The code solves the one-dimensional advection diffusion equation u_t + v * u_s = D u_{ss} for time t in [0,T] on the spatial domain [0, 1 + Vel*t]
where Vel is the velocity of the deforming interval.
"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt


N = 1000          # Number of spatial (finite) elements
M = 50         # Number of time steps
Vel = 3        # Velocity of interval boundary
v = 0.5            # Advection coefficient
D = 0.01            # Diffusion coefficient
T = 2.5          # Total simulation time
dt = T / M       # Time step
dx = 1 / N       # Spatial step
x_nodes = np.linspace(0, 1, N + 1)  # Spatial nodes
fine = 2*N+1
x_fine = np.linspace(0,1, fine) # Fine mesh for hat functions



hat_function = np.zeros(( N+1, fine))
for i in range (1,N):                                          
        hat_function[[i], :] = np.piecewise(x_fine,
                        [x_fine==x_nodes[i], (x_fine<x_nodes[i])&(x_fine>x_nodes[i-1]), (x_fine<x_nodes[i+1])&(x_fine>x_nodes[i]), (x_fine<=x_nodes[i-1])&(x_fine>=x_nodes[i+1])], 
                        [1, lambda x_fine: 1-(x_nodes[i]-x_fine)/dx, lambda x_fine: 1-(x_fine-x_nodes[i])/dx, 0])
hat_function = np.delete(hat_function,0,0)   

        
# Initialize the solution matrix
u = np.zeros((N + 1, M + 1))

# Set initial condition
u[:, 0] = - np.sin(2*np.pi * x_nodes)

# Set the boundary conditions
u[0, :] = 0
u[-1, :] = 0

# Solution u at the inner nodes of the interval
u_inner = u[1:-1]



# Time-stepping loop with implicit Euler
for t_step in range(1,M+1):
    t = t_step * dt

    # Create a list of the entries of the tridiagonal stiffness, mass and derivative matrices
    stiffness_diagonals =  [2*D / (dx*(1+Vel*t)**2), - D / (dx*(1+Vel*t)**2) , -  D / (dx*(1+Vel*t)**2)]
    mass_diagonals = [2*dx/3, dx/6, dx/6]
    derivative_diagonals = [0, -v / (2*(1+ Vel * t)), v / (2*(1+ Vel * t))]

    # Create sparse stiffness, mass, derivative matrices
    Stiffness_matrix = sp.diags(stiffness_diagonals, [0, 1, -1], shape=(N-1, N-1), format='csr')
    Mass_matrix = sp.diags(mass_diagonals, [0, 1, -1], shape=(N-1, N-1), format='csr')
    Derivative_matrix = sp.diags(derivative_diagonals, [0, 1, -1], shape=(N-1, N-1), format='csr')

    # Solve the linear system using implicit Euler
    u_inner[:, t_step ] = splinalg.spsolve(Mass_matrix + dt * (Derivative_matrix + Stiffness_matrix) , Mass_matrix @ u_inner[:, t_step-1])
    
    
# Substitute the values at the inner nodes into solution matrix
u[1:-1] = u_inner 


# Plot the hat functions
#for j in range(0,N):
#    plt.plot(x_fine, hat_function[j])
#    plt.legend(['$\chi_1$','$\chi_2$','$\chi_3$','$\chi_4$' ])
#plt.show()    

# Create a grid of (x, t) values for plotting
X, T = np.meshgrid(x_nodes, np.linspace(0, T, M + 1))

# Plot the solution
plt.figure(figsize=(8, 6))
plt.contourf(X, T, u.T, cmap='viridis')
plt.colorbar()
plt.xlabel('s')
plt.ylabel('t')
plt.title('Numerical Solution u')
plt.show()


# Plot the solution for each time step separately
for t_step in range(M + 1):
    plt.figure(figsize=(8, 6))
    plt.plot(x_nodes, u[:, t_step], label=f't = {t_step * dt:.2f}')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f'Numerical Solution at t = {t_step * dt:.2f}')
    plt.legend()
    plt.grid(True)
    plt.show()   

 
plt.figure(figsize=(8, 6))
plt.plot(x_nodes, u[:, 0], label=f't = {0 * dt:.2f}')
plt.plot(x_nodes, u[:, 10], label=f't = {10 * dt:.2f}')
plt.plot(x_nodes, u[:, 20], label=f't = {20 * dt:.2f}')
plt.plot(x_nodes, u[:, 30], label=f't = {30 * dt:.2f}')
plt.plot(x_nodes, u[:, 40], label=f't = {40 * dt:.2f}')
plt.plot(x_nodes, u[:, 50], label=f't = {50 * dt:.2f}')
plt.xlabel('s')
plt.ylabel('$\overline{U}_h$')
plt.legend()
plt.grid(True)
plt.show() 
