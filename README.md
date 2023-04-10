# Eulerian-and-Lagrangian-methods

This repo contains code used to run simulations for the manuscript "A comparison of Eulerian and Lagrangian methods for vertical particle
transport in the water column" by Nordam, Kristiansen, Nepstad, van Sebille & Booth (submitted to GMD). Some examples of plotting are also shown.

## Simulation code

The essence of the repo is two equivalent implementations of a water-column model for rising and settling particles, one Lagrangian and one Eulerian. The two implementations are used to study three example cases, described in detail in the manuscript. Scripts to run each of the three cases with either implementation is found in ```./scripts```. The scripts take different command line arguments to change numerical and other parameters. Run with ```-h``` or ```--help``` to see options, or run without arguments to accept defaults, e.g., ```python lagrangian_case2.py```.

To run for a large set of different numerical parameters, for investigation of numerical convergence, two convenience scripts have been created that will generate a (large) set of arguments to scan through different numerical parameters. To use these, it is recommended to also use, e.g., ```xargs``` to run the simulations in parallel (change ```-P8``` to reflect number of parallel simulations to run):

```
python generate_args_eulerian | xargs -n8 -P8 python eulerian_case2.py
```

## Example plots

The notebook ```notebooks/Example-plotting.ipynb``` contains some examples for plotting the results of the simulations. Note that you have to run the simulation first, and that you may have to update the notebook to match the simulations you ran, and the location of the results files.

## Terminal velocity distribution of microplastics

One of the three cases makes use of a terminal velocity distribution for microplastics, derived by combining the results of two earlier papers. The code used to obtain the velocity distribution is demonstrated in some detail in the notebookn ```notebooks/Microplastics-Speed-Distribution.ipynb```.
