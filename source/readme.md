# Source
In this directory the source for the main program can be found. The main portion of the project is essentially just one program but its source is split in two files: **ex3.cuh** and **ex3.cu**.

While these files are just a header-source code pair, the header file (**ex3.cuh**) contains auxiliary functions, macros, structs and CUDA kernels while the source file (**ex3.cu**) contains the important CUDA kernels as well as the entry point for the program. While the way the code is organized isn't the standard *(correct)* way, this project is relatively small so this deviation from the norm was considered harmless. 

---
# ex3.cuh
In the header file the following can be found.
## CUDA Runtime Functions

## Parameter Calculation Kernels
One of the features of this program is its dynamic selection of CUDA Kernel parameters, that is how big a block 
## File Loading Functions
The program uses two functions to load input files and export result files, the latter being used for debugging. The **input_cant_be_used** function is used in an if-condition in **main()** but implements a very rudimentary check regarding input matrix dimensions.
