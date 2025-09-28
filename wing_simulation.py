#!/usr/bin/env python3
"""
Fluid Simulation of a Wing

This module implements a simplified 2D fluid simulation around a wing profile
using computational fluid dynamics principles. The simulation uses finite
difference methods to solve a simplified version of the Navier-Stokes equations
for incompressible flow around a NACA airfoil.

Author: Generated for git_test repository
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')


class WingFluidSimulation:
    """
    A 2D fluid simulation around a wing profile using finite difference methods.
    
    This class implements a simplified CFD solver for incompressible flow around
    a NACA airfoil, including velocity field calculation, pressure distribution,
    and visualization capabilities.
    """
    
    def __init__(self, nx=200, ny=100, length=4.0, height=2.0):
        """
        Initialize the fluid simulation grid and parameters.
        
        Args:
            nx (int): Number of grid points in x-direction
            ny (int): Number of grid points in y-direction
            length (float): Domain length
            height (float): Domain height
        """
        self.nx = nx
        self.ny = ny
        self.length = length
        self.height = height
        
        # Grid spacing
        self.dx = length / (nx - 1)
        self.dy = height / (ny - 1)
        
        # Create coordinate arrays
        self.x = np.linspace(0, length, nx)
        self.y = np.linspace(0, height, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize velocity and pressure fields
        self.u = np.zeros((ny, nx))  # x-velocity component
        self.v = np.zeros((ny, nx))  # y-velocity component
        self.p = np.zeros((ny, nx))  # pressure field
        
        # Flow parameters
        self.Re = 100.0  # Reynolds number
        self.dt = 0.001  # Time step
        self.free_stream_velocity = 1.0  # Free stream velocity
        self.angle_of_attack = 5.0  # Angle of attack in degrees
        
        # Wing parameters
        self.chord_length = 1.0
        self.wing_x_pos = 1.0  # Wing position along x-axis
        self.wing_y_pos = height / 2  # Wing position along y-axis
        
        # Initialize simulation
        self._setup_initial_conditions()
        self._generate_wing_profile()
        self._apply_wing_boundary()
    
    def _setup_initial_conditions(self):
        """Set up initial flow conditions."""
        # Set uniform flow in x-direction
        angle_rad = np.deg2rad(self.angle_of_attack)
        self.u.fill(self.free_stream_velocity * np.cos(angle_rad))
        self.v.fill(self.free_stream_velocity * np.sin(angle_rad))
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
    
    def _generate_wing_profile(self):
        """Generate NACA 0012 airfoil profile."""
        # NACA 0012 airfoil profile
        chord_points = np.linspace(0, 1, 100)
        thickness = 0.12  # 12% thickness
        
        # Calculate airfoil shape using NACA equation
        y_upper = 5 * thickness * (
            0.2969 * np.sqrt(chord_points) - 
            0.1260 * chord_points - 
            0.3516 * chord_points**2 + 
            0.2843 * chord_points**3 - 
            0.1015 * chord_points**4
        )
        y_lower = -y_upper
        
        # Scale and position the wing
        self.wing_x = self.wing_x_pos + chord_points * self.chord_length
        self.wing_y_upper = self.wing_y_pos + y_upper * self.chord_length
        self.wing_y_lower = self.wing_y_pos + y_lower * self.chord_length
        
        # Create wing mask for boundary conditions
        self.wing_mask = np.zeros((self.ny, self.nx), dtype=bool)
        
        # Mark grid points inside the wing
        for i in range(self.ny):
            for j in range(self.nx):
                x_point = self.x[j]
                y_point = self.y[i]
                
                if (self.wing_x_pos <= x_point <= self.wing_x_pos + self.chord_length):
                    x_norm = (x_point - self.wing_x_pos) / self.chord_length
                    if 0 <= x_norm <= 1:
                        # Interpolate wing profile at this x position
                        y_u = np.interp(x_norm, chord_points, self.wing_y_upper)
                        y_l = np.interp(x_norm, chord_points, self.wing_y_lower)
                        
                        if y_l <= y_point <= y_u:
                            self.wing_mask[i, j] = True
    
    def _apply_wing_boundary(self):
        """Apply no-slip boundary condition at wing surface."""
        self.u[self.wing_mask] = 0.0
        self.v[self.wing_mask] = 0.0
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions to the domain."""
        # Inlet boundary (left side): fixed velocity
        angle_rad = np.deg2rad(self.angle_of_attack)
        self.u[:, 0] = self.free_stream_velocity * np.cos(angle_rad)
        self.v[:, 0] = self.free_stream_velocity * np.sin(angle_rad)
        
        # Outlet boundary (right side): zero gradient
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        
        # Top and bottom boundaries: slip condition
        self.v[0, :] = 0
        self.v[-1, :] = 0
        self.u[0, :] = self.u[1, :]
        self.u[-1, :] = self.u[-2, :]
    
    def _solve_pressure(self):
        """Solve pressure Poisson equation using Jacobi iteration."""
        p_new = self.p.copy()
        
        # Calculate pressure source term from velocity divergence
        dudx = np.zeros_like(self.u)
        dvdy = np.zeros_like(self.v)
        
        dudx[:, 1:-1] = (self.u[:, 2:] - self.u[:, :-2]) / (2 * self.dx)
        dvdy[1:-1, :] = (self.v[2:, :] - self.v[:-2, :]) / (2 * self.dy)
        
        source = -(dudx + dvdy) / self.dt
        
        # Jacobi iteration for pressure
        for _ in range(50):  # Limited iterations for performance
            p_old = p_new.copy()
            
            p_new[1:-1, 1:-1] = (
                (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * self.dy**2 +
                (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * self.dx**2 -
                source[1:-1, 1:-1] * self.dx**2 * self.dy**2
            ) / (2 * (self.dx**2 + self.dy**2))
            
            # Pressure boundary conditions
            p_new[:, 0] = p_new[:, 1]    # Inlet
            p_new[:, -1] = 0             # Outlet (reference pressure)
            p_new[0, :] = p_new[1, :]    # Top
            p_new[-1, :] = p_new[-2, :]  # Bottom
        
        self.p = p_new
    
    def step(self):
        """Perform one time step of the simulation."""
        u_new = self.u.copy()
        v_new = self.v.copy()
        
        # Calculate velocity derivatives
        dudx = np.zeros_like(self.u)
        dudy = np.zeros_like(self.u)
        dvdx = np.zeros_like(self.v)
        dvdy = np.zeros_like(self.v)
        
        # Central differences for interior points
        dudx[:, 1:-1] = (self.u[:, 2:] - self.u[:, :-2]) / (2 * self.dx)
        dudy[1:-1, :] = (self.u[2:, :] - self.u[:-2, :]) / (2 * self.dy)
        dvdx[:, 1:-1] = (self.v[:, 2:] - self.v[:, :-2]) / (2 * self.dx)
        dvdy[1:-1, :] = (self.v[2:, :] - self.v[:-2, :]) / (2 * self.dy)
        
        # Second derivatives for viscous terms
        d2udx2 = np.zeros_like(self.u)
        d2udy2 = np.zeros_like(self.u)
        d2vdx2 = np.zeros_like(self.v)
        d2vdy2 = np.zeros_like(self.v)
        
        d2udx2[:, 1:-1] = (self.u[:, 2:] - 2*self.u[:, 1:-1] + self.u[:, :-2]) / self.dx**2
        d2udy2[1:-1, :] = (self.u[2:, :] - 2*self.u[1:-1, :] + self.u[:-2, :]) / self.dy**2
        d2vdx2[:, 1:-1] = (self.v[:, 2:] - 2*self.v[:, 1:-1] + self.v[:, :-2]) / self.dx**2
        d2vdy2[1:-1, :] = (self.v[2:, :] - 2*self.v[1:-1, :] + self.v[:-2, :]) / self.dy**2
        
        # Pressure gradients
        dpdx = np.zeros_like(self.p)
        dpdy = np.zeros_like(self.p)
        dpdx[:, 1:-1] = (self.p[:, 2:] - self.p[:, :-2]) / (2 * self.dx)
        dpdy[1:-1, :] = (self.p[2:, :] - self.p[:-2, :]) / (2 * self.dy)
        
        # Update velocities using simplified Navier-Stokes
        nu = 1.0 / self.Re  # Kinematic viscosity
        
        # u-momentum equation
        u_new = self.u + self.dt * (
            -self.u * dudx - self.v * dudy +  # Convection
            nu * (d2udx2 + d2udy2) -          # Viscous diffusion
            dpdx                               # Pressure gradient
        )
        
        # v-momentum equation
        v_new = self.v + self.dt * (
            -self.u * dvdx - self.v * dvdy +  # Convection
            nu * (d2vdx2 + d2vdy2) -          # Viscous diffusion
            dpdy                               # Pressure gradient
        )
        
        # Update velocity fields with numerical stability checks
        # Clip extreme values to prevent numerical instability
        u_new = np.clip(u_new, -10.0, 10.0)
        v_new = np.clip(v_new, -10.0, 10.0)
        
        self.u = u_new
        self.v = v_new
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
        self._apply_wing_boundary()
        
        # Solve pressure equation
        self._solve_pressure()
    
    def run_simulation(self, n_steps=1000):
        """Run the simulation for a specified number of steps."""
        print(f"Running simulation for {n_steps} steps...")
        for step in range(n_steps):
            if step % 100 == 0:
                print(f"Step {step}/{n_steps}")
            self.step()
        print("Simulation completed!")
    
    def visualize_flow(self, save_filename=None):
        """Visualize the flow field around the wing."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Velocity field with streamlines
        speed = np.sqrt(self.u**2 + self.v**2)
        
        # Create streamlines
        start_points_x = np.full(15, 0.1)
        start_points_y = np.linspace(0.1, self.height - 0.1, 15)
        start_points = np.column_stack([start_points_x, start_points_y])
        
        im1 = ax1.contourf(self.X, self.Y, speed, levels=50, cmap='viridis')
        ax1.streamplot(self.X, self.Y, self.u, self.v, 
                      start_points=start_points, density=2, color='white', linewidth=0.8)
        
        # Draw wing
        wing_coords = np.column_stack([
            np.concatenate([self.wing_x, self.wing_x[::-1]]),
            np.concatenate([self.wing_y_upper, self.wing_y_lower[::-1]])
        ])
        wing_patch = Polygon(wing_coords, closed=True, facecolor='red', edgecolor='black')
        ax1.add_patch(wing_patch)
        
        ax1.set_xlim(0, self.length)
        ax1.set_ylim(0, self.height)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Velocity Field and Streamlines')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='Speed (m/s)')
        
        # Plot 2: Pressure field
        im2 = ax2.contourf(self.X, self.Y, self.p, levels=50, cmap='RdBu_r')
        
        # Draw wing
        wing_patch2 = Polygon(wing_coords, closed=True, facecolor='black', edgecolor='white')
        ax2.add_patch(wing_patch2)
        
        ax2.set_xlim(0, self.length)
        ax2.set_ylim(0, self.height)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Pressure Field')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='Pressure (Pa)')
        
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {save_filename}")
        
        plt.show()
    
    def calculate_forces(self):
        """Calculate lift and drag forces on the wing."""
        # This is a simplified force calculation using surface pressure integration
        
        # Find wing surface points
        wing_surface_points = []
        for i in range(1, self.ny-1):  # Avoid boundary points
            for j in range(1, self.nx-1):
                if self.wing_mask[i, j]:
                    # Check if it's a surface point (has fluid neighbor)
                    is_surface = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (0 <= ni < self.ny and 0 <= nj < self.nx and 
                                not self.wing_mask[ni, nj]):
                                is_surface = True
                                break
                        if is_surface:
                            break
                    
                    if is_surface:
                        wing_surface_points.append((i, j))
        
        if not wing_surface_points:
            return 0.0, 0.0
        
        # Calculate pressure and viscous forces
        total_lift = 0.0
        total_drag = 0.0
        
        for i, j in wing_surface_points:
            # Pressure force (simplified)
            if not np.isnan(self.p[i, j]) and not np.isinf(self.p[i, j]):
                pressure_force = self.p[i, j] * self.dx * self.dy
                
                # Simplified force calculation based on position relative to wing center
                y_center = self.wing_y_pos
                if self.y[i] > y_center:  # Upper surface
                    total_lift += pressure_force * 0.5
                else:  # Lower surface
                    total_lift -= pressure_force * 0.5
                
                # Drag from pressure (simplified)
                total_drag += abs(pressure_force) * 0.1
        
        return total_lift, total_drag


def main():
    """Main function to run the wing fluid simulation."""
    print("Initializing Wing Fluid Simulation...")
    
    # Create simulation instance
    sim = WingFluidSimulation(nx=150, ny=75, length=3.0, height=1.5)
    
    print(f"Grid size: {sim.nx} x {sim.ny}")
    print(f"Reynolds number: {sim.Re}")
    print(f"Angle of attack: {sim.angle_of_attack}°")
    print(f"Free stream velocity: {sim.free_stream_velocity} m/s")
    
    # Run simulation
    sim.run_simulation(n_steps=500)
    
    # Calculate aerodynamic forces
    lift, drag = sim.calculate_forces()
    print(f"\nAerodynamic Forces:")
    print(f"Lift: {lift:.4f} N")
    print(f"Drag: {drag:.4f} N")
    
    # Visualize results
    print("\nGenerating visualization...")
    sim.visualize_flow(save_filename="wing_simulation_results.png")
    
    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()