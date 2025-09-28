#!/usr/bin/env python3
"""
Simple example demonstrating wing fluid simulation usage.

This script shows how to create and run different simulation scenarios
with varying parameters.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from wing_simulation import WingFluidSimulation


def run_angle_comparison():
    """Compare flow at different angles of attack."""
    print("Running angle of attack comparison...")
    
    angles = [0, 5, 10, 15]  # degrees
    results = []
    
    for angle in angles:
        print(f"  Simulating at {angle}° angle of attack...")
        
        # Create simulation
        sim = WingFluidSimulation(nx=100, ny=50, length=2.5, height=1.2)
        sim.angle_of_attack = angle
        sim.Re = 150.0
        
        # Run simulation
        sim.run_simulation(n_steps=200)
        
        # Calculate forces
        lift, drag = sim.calculate_forces()
        results.append((angle, lift, drag))
        
        # Save visualization
        filename = f"wing_flow_angle_{angle}deg.png"
        sim.visualize_flow(save_filename=filename)
        print(f"    Saved visualization: {filename}")
    
    # Print results summary
    print("\nResults Summary:")
    print("Angle (°) | Lift (N) | Drag (N) | L/D Ratio")
    print("-" * 45)
    for angle, lift, drag in results:
        ld_ratio = lift / drag if drag != 0 else 0
        print(f"{angle:8} | {lift:8.4f} | {drag:8.4f} | {ld_ratio:8.4f}")


def run_reynolds_comparison():
    """Compare flow at different Reynolds numbers."""
    print("\nRunning Reynolds number comparison...")
    
    reynolds_numbers = [50, 100, 200, 400]
    
    for re in reynolds_numbers:
        print(f"  Simulating at Re = {re}...")
        
        # Create simulation
        sim = WingFluidSimulation(nx=100, ny=50, length=2.5, height=1.2)
        sim.angle_of_attack = 8.0  # Fixed angle
        sim.Re = re
        
        # Run simulation
        sim.run_simulation(n_steps=300)
        
        # Calculate forces
        lift, drag = sim.calculate_forces()
        
        # Save visualization
        filename = f"wing_flow_Re_{re}.png"
        sim.visualize_flow(save_filename=filename)
        
        print(f"    Re={re}: Lift={lift:.4f} N, Drag={drag:.4f} N")
        print(f"    Saved visualization: {filename}")


def main():
    """Run example simulations."""
    print("Wing Fluid Simulation Examples")
    print("=" * 40)
    
    try:
        # Run different simulation scenarios
        run_angle_comparison()
        run_reynolds_comparison()
        
        print("\n🎉 All example simulations completed!")
        print("\nGenerated files:")
        print("- wing_flow_angle_*.png: Flow visualization at different angles")
        print("- wing_flow_Re_*.png: Flow visualization at different Reynolds numbers")
        
    except Exception as e:
        print(f"❌ Example failed with error: {e}")
        raise


if __name__ == "__main__":
    main()