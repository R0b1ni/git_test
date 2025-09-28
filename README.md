# Wing Fluid Simulation

This repository contains a Python implementation of a 2D fluid simulation around a wing profile using computational fluid dynamics (CFD) principles.

## Features

- **NACA Airfoil Geometry**: Implements NACA 0012 airfoil profile for realistic wing shape
- **Finite Difference CFD Solver**: Uses finite difference methods to solve simplified Navier-Stokes equations
- **Flow Visualization**: Generates velocity field streamlines and pressure contour plots
- **Force Calculation**: Computes lift and drag forces on the wing surface
- **Configurable Parameters**: Adjustable Reynolds number, angle of attack, and grid resolution

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Main Simulation

```bash
python wing_simulation.py
```

This will run a full simulation with default parameters and generate a visualization file `wing_simulation_results.png`.

### Run Tests

```bash
python test_simulation.py
```

This runs a quick test suite to verify the simulation functionality.

### Custom Simulation

```python
from wing_simulation import WingFluidSimulation

# Create custom simulation
sim = WingFluidSimulation(
    nx=150,           # Grid points in x-direction
    ny=75,            # Grid points in y-direction
    length=3.0,       # Domain length
    height=1.5        # Domain height
)

# Set parameters
sim.angle_of_attack = 10.0  # degrees
sim.Re = 200.0              # Reynolds number

# Run simulation
sim.run_simulation(n_steps=1000)

# Calculate forces
lift, drag = sim.calculate_forces()
print(f"Lift: {lift:.4f} N, Drag: {drag:.4f} N")

# Visualize
sim.visualize_flow(save_filename="my_simulation.png")
```

## Simulation Physics

The simulation implements:

- **Incompressible Flow**: Assumes incompressible fluid (air) flow
- **Navier-Stokes Equations**: Solves momentum equations with viscous and pressure terms
- **Boundary Conditions**: 
  - No-slip condition at wing surface
  - Inlet boundary with prescribed velocity
  - Outlet with zero-gradient condition
  - Slip conditions at top and bottom boundaries
- **Pressure-Velocity Coupling**: Uses pressure Poisson equation to maintain incompressibility

## Output

The simulation generates:
1. Velocity field visualization with streamlines
2. Pressure field contour plots
3. Calculated lift and drag forces
4. Wing geometry overlay

## Parameters

- `Reynolds number (Re)`: Controls viscous effects (default: 100)
- `Angle of attack`: Wing orientation in degrees (default: 5°)
- `Grid resolution`: Number of computational grid points
- `Time step`: Simulation time increment for stability

## Limitations

This is a simplified CFD implementation suitable for educational purposes and basic analysis. For production aerodynamic analysis, consider commercial CFD software.

## Files

- `wing_simulation.py`: Main simulation code with WingFluidSimulation class
- `test_simulation.py`: Test suite for verification
- `requirements.txt`: Python package dependencies
- Generated output: `*.png` visualization files