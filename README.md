# NAS Parallel Benchmarks in Rust - NPB-Rust

*Authors: Gabriel Leite Bessa, [Luan Mark Da Silva Borela](https://github.com/Lord-Mark) and [João A. Soares](https://github.com/JoaoNevesSoares).*

We developed the NAS Parallel Benchmarks (NPB) in Rust as a project for the subject *Tópicos Especiais em Computação - Programação paralela* at the Federal University of Pelotas (*Universidade Federal de Pelotas - UFPel*).

The NPB consists into 5 kernels (short programs):
- IS - Integer Sort, random memory access
- EP - Embarrassingly Parallel
- CG - Conjugate Gradient, irregular memory access and communication
- MG - Multi-Grid on a sequence of meshes, long- and short-distance communication, memory intensive
- FT - discrete 3D fast Fourier Transform, all-to-all communication

And 3 pseudo applications (large programs): 

- BT - Block Tri-diagonal solver
- SP - Scalar Penta-diagonal solver
- LU - Lower-Upper Gauss-Seidel solver

You can find out more about at [NPB website](https://www.nas.nasa.gov/software/npb.html).

This repo contains all NPB benchmarks that we did til today. We pretend to expand and publish here all kernels and applications with their respective serial and parallel versions.

## Dependencies

There are a few dependecies we use in the project, but don't worry, all dependencies are specified at the Cargo.toml file and will be installed automatically once you run the project.

## How to run this project

We've made a Makefile to help you with this task.

You can just compile it with:

`make compile KERNEL=<kernel_you_want> CLASS=<class_you_want>`

To run you can execute this command:

`make run KERNEL=<kernel_you_want> CLASS=<class_you_want> NUM_THREADS=<number_of_threads_you_want>`

The number of threads is specified just with the `make run` and it will only be considered with the parallel versions of each program.

## How it is structured

We made an implementation very close to the original and with the CPP version of it. So we have all common source files and setparams exactly the same as those others, but since Rust have a predefined project structure we decided to put common libraries at a sub module inside our project, the setparams file inside the bin folder and all source files inside src/templates.
When you run `make compile ...` it will modify the template of the kernel you specified and create a version with the correct parameters at src/bin, and then you can easily run `make run ...` to execute the benchmark.

There are others files at the root folder that we developed to run a benchmark multiple times and extract its result. Feel free to use it to compare with your results.
