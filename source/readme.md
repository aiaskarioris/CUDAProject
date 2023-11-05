# Source
In this directory the source for the main program can be found. The main portion of the project is essentially just one program but its source is split in two files: **ex3.cuh** and **ex3.cu**.

While these files are just a header-source code pair, the header file (**ex3.cuh**) contains auxiliary functions, macros, structs and CUDA kernels while the source file (**ex3.cu**) contains the important CUDA kernels as well as the entry point for the program. While the way the code is organized isn't the standard *(correct)* way, this project is relatively small so this deviation from the norm was considered harmless. 

---
# ex3.cuh
In the header file the following can be found.
## CUDA Runtime Functions
These functions are used by the GPU for various tasks. **checkErrors** is a macro that calls **checkErrors__line** so that the CUDA C++ function **cudaGetLastError** can be called and terminate execution of the program if an error occurred. **allocateAndLoad** is a convenience function used to quickly allocate memory on the GPU and load it with data.

## Parameter Calculation Kernels
One of the features of this program is its dynamic selection of CUDA Kernel parameters, that is how a big block of input numbers will be split to multiple smaller blocks. Since many limitations exist on the values these parameters can get, due to the properties of the GPU and the algorithms used, the kernels defined in this section poll information from the GPU and compute these parameters.

## File Loading Functions
The program uses two functions to load input files and export result files, the latter being used for time-measurement and debugging. The **input_cant_be_used** function is used in an if-condition in **main()** but implements a very rudimentary check regarding input matrix dimensions.

# ex3.cu
This is the file that contains the program's entry point as well as the main CUDA kernels.