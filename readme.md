# CUDA Project
This is a project for the Parallel Systems course provided by the Department of Informatics and Computer Engineering, in University of W. Attica. It is written in CUDA C++, an expansion of C++ for low-level programming of CUDA-enabled NVIDIA graphics cards. This repository contains
- the CUDA C++ source 
- standard C++ source code for other parts of the project
- documentation and notes about the project.

## Abstract
Given a matrix (MxN) and a vector (1xN) of single-precision floats, the program computes a new vector (1xM) in the following way:

First, the input matrix A is multiplied by the input vector V, producing the vector B.

Then, the transpose of A is multiplied with B, giving the resulting vector C.

While this operation is arbitrary, the goal of the project was to write an efficient hyper-parallel program showcasing entry-level as well as more intermediate practices and optimizations for working with graphics cards and thousands of computing units.

## Development Environment
This project was developed for Linux and was tested on a remote server utilizing an NVIDIA Titan RTX graphics card (GPU). As such, it should be stated that:
- Building the project has not been tested on non-Linux Operating Systems
- The code has not been tested on other NVIDIA GPUs

---

# The Project itself
In the following paragraphs the way the program works is explained in detail, as well as the thought that went behind the code. A PDF version of the following is also provided, albeit written in Greek.

## Summary
As explained above, this project is essentially an implementation of a simple, two-operation calculation regarding matrices and vectors, algebraic structures with innate parallelism. 

The program is written with parameterization in mind as well as using as little memory as possible. Input data is divided into "tiles", allowing for more efficient control as well as better control of memory.

## Source Code
The source code of the project is split in two files that can be found in the [**source**](https://github.com/aiaskarioris/CUDAProject/tree/main/source) directory. More details about these files can be found there. Also, in the [**tools-source**](https://github.com/aiaskarioris/CUDAProject/tree/main/tools-source) directory, the source code of the tools that were developed for assistance can be found, as well as short descriptions on what they do.

## Input Files
The executable program receives as input files containing a matrix and a vector each, stored in a binary format. These files are called 'Test Files' in the documentation and are generated by the **gen** tool (see [tools-source](https://github.com/aiaskarioris/CUDAProject/tree/main/tools-source)). The test file format is fairly simple, consisting of just a 16 byte header and then the raw data of the matrix and the vector.

<p align="center">
 <image src="https://github.com/aiaskarioris/CUDAProject/blob/main/pictures/test_format.png" alt="Test file format"></image>
</p>

# The program's outline
In short, the main program loads a matrix and a vector of appropriate dimensions and computes their product. Then, to multiply said product by the transpose of the input matrix, the input matrix can be used again as is from the GPU's memory but with a different indexing scheme, a scheme that matches a transposed matrix. This way unnecessary computations and transfers from/to memory are avoided, simplifying the code and accelerating execution.


By now it should be clear that the GPU must execute two different functions (kernels): One for multiplying matrices by vectors and one for multiplying the transpose of matrices by vectors.

## Selecting an input file
The serial portion of the program is controlled via the *main()* function. When calling [**ex3**](https://github.com/aiaskarioris/CUDAProject/tree/main/source#ex3cu), the input test's number is provided as an argument through the CLI.

---

# CUDA Kernels
As mentioned above, the program uses two CUDA kernels to complete the desired tasks. These two kernels work in the same way: A matrix and a vector of matching dimensions are used to produce a new vector.

## mult_MatByVec
Below is a brief summary of how the first kernel to be executed works. 

### Parts & Blocks
When tiling the input data, two main variables are used: **parts** and **blocks**. Blocks in this context are closely related to CUDA blocks, that is groups of processing units (PU) in the GPU. For this program, CUDA blocks are hard-coded to be 128 threads. Each line of the input matrix, as well as the input vector in its entirety, are split into multiple parts. Each part is then split into blocks, so that each CUDA block can work in parallel on a portion of the currently loaded part. Unlike matrices that aren't loaded into shared memory (but are read directly from the GPU's global memory), input vectors are split into parts and loaded into shared memory, having one part loaded at each given execution time. The size of the vector part loaded is defined by the *IN_VEC_BLOCK_COUNT* (counted in blocks, e.g. 2 blocks of 128 numbers) which in turn specifies the value of the *IN_VEC_TILE_SIZE* constant.

### Parallel Loading
During execution, each CUDA block loads into shared memory a part of the vector in parallel, stored in the *Vds* array. *Vds* size is defined by the *IN_VEC_TILE_SIZE* constant. This way copying data from global memory to shared memory becomes a very quick operation.

### Parallel Calculation
To calculate the n-th element of the product vector, the elements of the n-th line of the matrix must each be multiplied with the respective elements of the input vector and then have all products summed. This algorithm has two useful properties when implemented with parallelism:
 - First, each element of the product vector is a sum. This means that calculating it can be split into calculating independent partial sums and then have all these parts summed up together.
 - Second, all product vector elements require the input vector in its entirety for computations.

Each CUDA block is responsible for computing a few consecutive elements of the product vector. If for example it was decided that each CUDA block would have to calculate 2 elements, the first block would calculate elements 0 and 1, the second block elements 2 and 3, and so on (See [findKernelParameters1D](https://github.com/aiaskarioris/CUDAProject#findkernelparameters1d) for details).
 
The two aforementioned properties can be utilized so that data I/O in global memory is minimized. Simply put, since all product vector elements are sums, using an array with one element for each vector element/matrix line is a reasonable choice, hence the *Rds* array. The following steps are followed to calculate the final vector:
 1. For each part of the input vector, said part is loaded from global memory to shared memory. Let's say that a part has *P* elements.
 2. For each input-matrix line a CUDA block is responsible for, *P* elements are read from the line.
 3. Using the buffered input vector part and the freshly-read matrix elements, the partial sum of one product vector element corresponding to the matrix line is calculated and stored into *Rds*. The summation portion of the algorithm is done using a reduction algorithm.
 4. Steps 2 and 3 are repeated by each CUDA block for each matrix line/product vector element the block is responsible for. 
 5. By now, each *Rds* element has been filled with data but the sums it stores are not yet complete. Step 1 is repeated but for the next vector part. The CUDA blocks will read the next *P* elements of their input matrix lines.
 6. After all input vector parts have passed through the shared memory, *Rds* stores the product vector. *Rds* is copied to global memory.

<p align="center">
 <image src="https://github.com/aiaskarioris/CUDAProject/blob/main/pictures/parts_and_blocks.png" alt="Parts and blocks illustration"></image>
</p>

By prioritizing using the loaded input vector elements to their fullest instead of prioritizing calculating each product vector element to its fullest, writes to shared memory are minimized to their bare minimum, thus avoiding memory access penalties.

## mult_TransByVec
The second main kernel of the program is almost identical to *mult_MatByVector* with the only difference being that instead of each CUDA block accessing input matrix lines, matrix columns are accessed instead. This way at no given point is there in memory a copy of the input matrix's transpose.

It should be noted that because by the end of *mult_MatByVec* the GPU's global memory has the input matrix and the intermediate vector (B) loaded, the CPU doesn't need to move data to the GPU as these are the input data needed by *Mult_TransByVec*.

## findKernelParameters1D
This is an auxiliary kernel that calculates the optimal number of blocks required by the program. The kernel uses the GPU to find the largest number that divides the number of lines of the matrix and is also a viable option for number of blocks. The same kernel finds the optimal number that divides the number of columns as well. To represent the kernel parameters the *kernelParameters1D* structure is used. In this structure, the fields **blockCount_for_lines** and **blockCount_for_columns** are used to store the number of blocks the two kernels will use respectively. See [Choosing Kernel Parameters](https://github.com/aiaskarioris/CUDAProject/tree/main#choosing-kernel-parameters) for more.

# Features and Limitations
## Memory Limitations
Because the kernels above split matrices and vectors to parts and blocks, a few limitation exist. First and foremost, the kernels use static memory allocation. This means that *Rds, Vds* as well as some other arrays used must have their sizes set at compile time. In practice these arrays get a maximum size and each CUDA block uses as much space as it needs, leaving some elements unused. While the kernels are written in such a way as to work with limited memory, if the input data dimensions do fit entirely in shared memory the program will behave accordingly. However, because the input vector and the matrix is split into equal portions of *BLOCK_SIZE* elements, input dimensions must be multiples of *BLOCK_SIZE*. A theoretically possible but not implemented solution to this limitation is presented in [Possible Improvements](https://github.com/aiaskarioris/CUDAProject/tree/main#possible-improvements).

## Choosing Kernel Parameters
In [Parts & Blocks](https://github.com/aiaskarioris/CUDAProject/tree/main#parts--blocks), the *IN_VEC_BLOCK_COUNT* and *IN_VEC_TILE_SIZE* constants are mentioned, along with some other parameters such as the number of lines each CUDA block is responsible for. In this section constants and parameters such as these are explained.

### Block Size
The block size, wether talking about CUDA blocks or number blocks, is the most important and basic parameter in the kernels. Essentially, the block parameter specifies how many threads are working on the same instruction at any given moment. The value of a block is hardcoded to be **128** but this number itself is derived by experimentation and is specific to the Titan RTX GPU. The specifics of the number are CUDA-related but, essentially, choosing a block size of 128 threads allows for 576 independent blocks, with each Streaming Multiprocessor controlling 8 blocks, providing each block with 8KiB of block-shared memory. More powerful GPUs would provide more benefits in larger block counts.

### Tile Sizes
The constants *IN_VEC_BLOCK_COUNT* and *IN_VEC_TILE_SIZE* can be collectively called as tile sizes as they define the size of a tile in CUDA-talk. Once again, these values are determined by the GPU's memory capabilities, setting the number of blocks per part (*IN_VEC_BLOCK_COUNT*) to 14, thus setting *IN_VEC_TILE_SIZE* to 14×BLOCK_SIZE single-precision floats.

### Lines/Columns/Elements per Block
While CUDA kernels might seem a little restrictive with their lack of dynamic allocation, they do have a lot of flexibility when called by the CPU since the number of blocks that will execute the kernel and the sizes of these blocks are configurable. The number of blocks that will be used for the two kernels are calculated by **FindKernelParameters1D**.

## Input Dimension Limitations due to Tile Size
As discussed above, with the use of tiling, the CUDA kernels are able to process data that wouldn't otherwise fit in shared memory. However, due to a coding quirk, input dimensions must be integer multiples of the tile size otherwise the kernels fail to process all parts of the input. This is because the number of parts each CUDA block must compute is calculated with the division below. If the result isn't an integer it gets rounded down to the nearest integer. 
<p align="center">
 <image src="https://github.com/aiaskarioris/CUDAProject/blob/main/pictures/partcount_division.png" alt="Parts and blocks illustration"></image>
</p>

So, for example, if part size was set to 256 numbers but input was set to 384 (= 128*3), *partCount* would be 1.5 and then rounded down to 1, skipping some blocks (one in this case).

# Possible Improvements
The program's code, while efficient, has room for improvements. In this section, some of these possible changes are discussed.

## Removing the Input Size Limitations due to Tiles
To mitigate the limitations imposed by tile sizes, an extra check could be added to catch the problematic case above and re-organize calculations accordingly. Caution should then be taken to stop a CUDA block from processing another CUDA block's data. However, such an addition would increase the code's complexity.

## Removing the Input Dimensions Limitations due to Block Size
Programming the kernels so that input dimensions are independent from  block size is theoretically possible but would greatly increase the code's complexity and execution time. Design-wise, the use of CUDA blocks and number blocks is used so that no individual thread has to check if it should or shouldn't participate in a calculation. However, it should be possible to have each thread check if it should stop working on a block calculation. For example, if an input line of 100 numbers was provided, 28 threads wouldn't have to work on each CUDA block. This kind of thread-level control flow, however, greatly decreases efficiency and is better off avoided.

## Dynamic Kernel Selection
In the [CUDA Kernels](https://github.com/aiaskarioris/CUDAProject#cuda-kernels) section, a few constants are described that because of CUDA limitations cannot change at runtime. If the kernels were to be executed on multiple different GPUs, such as ones with less memory and cores, or on stronger GPUs, it could be beneficial to have these constants have different values. 

This dynamic selection of constants could be achieved by defining different kernel programs that execute the same code but with different constants. Then, when *FindKernelParameters1D* would be executed, the program would determine which kernel would be the optimal to use.

## 2D Kernel Parameters
As discussed in [findKernelParameters1D](https://github.com/aiaskarioris/CUDAProject#findkernelparameters1d), kernels are executed with 1D CUDA Blocks. In cases where the GPU could support it in terms of thread count, it could be beneficial to use 2D CUDA blocks. Then, the first dimension could control how many lines/columns each CUDA block processes and the second dimension could control what parts of the line/column each CUDA block would process.
