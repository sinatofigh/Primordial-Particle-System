# Primordial-Particle-System
##### A Python implementation of primordial particle system

This is my first take on implementing Primordial Particle System(PPS) by Schmickl, Stefanec, and Crailsheim (https://www.nature.com/articles/srep37969) in Python. In a nutshell, PPS is  an example of an emergent self-sustaining self-organizing system, made up of self-propelled particles driven by a simple motion law. 

As an example of an Agent Based Modelling (ABM), one is tempted to write it up as objects and classes, but for self-education and speeding purposes, I tried to stick with matrices and functions (probably another reason could be the unfriendly blend of Numba library with OOP).

One file (PPS_simulator.py) only renders the simulation and saves all frames data as an npz file, then another file (PPS_animator.py) creates the animation file.
