#!/usr/bin/env python3
"""
Display the generated wing simulation results.

This script helps visualize the generated PNG files from the simulation.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_simulation_results():
    """Display available simulation result images."""
    
    # Find all PNG files in current directory
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    
    if not png_files:
        print("No PNG result files found. Run the simulation first:")
        print("  python wing_simulation.py")
        print("  python test_simulation.py") 
        return
    
    print(f"Found {len(png_files)} result files:")
    for i, filename in enumerate(png_files, 1):
        print(f"  {i}. {filename}")
    
    # Show information about each file
    print("\nSimulation Results Summary:")
    print("=" * 50)
    
    for filename in png_files:
        try:
            img = mpimg.imread(filename)
            height, width = img.shape[:2]
            file_size = os.path.getsize(filename)
            
            print(f"\n📊 {filename}")
            print(f"   Image size: {width} x {height} pixels")
            print(f"   File size: {file_size:,} bytes")
            
            if 'wing_simulation_results' in filename:
                print("   Content: Main simulation results with velocity streamlines and pressure contours")
            elif 'test_simulation_result' in filename:
                print("   Content: Quick test simulation results")
            elif 'angle' in filename:
                print("   Content: Angle of attack comparison study")
            elif 'Re' in filename:
                print("   Content: Reynolds number comparison study")
            else:
                print("   Content: Simulation visualization")
                
        except Exception as e:
            print(f"   Error reading {filename}: {e}")
    
    print(f"\n✅ All visualization files are ready!")
    print(f"💡 Open the PNG files in an image viewer to see the fluid flow simulation results.")
    print(f"   The visualizations show:")
    print(f"   - Left panel: Velocity field with streamlines around the wing")
    print(f"   - Right panel: Pressure distribution with wing geometry")


def main():
    """Main function."""
    print("Wing Fluid Simulation - Results Viewer")
    print("=" * 40)
    
    show_simulation_results()


if __name__ == "__main__":
    main()