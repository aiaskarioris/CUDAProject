#include <stdio.h>
#include <stdlib.h>

/// CUDA Runtime Functions ///////////////////////////////////////

/* Checks if an error occurred and exits if it did */
#define checkErrors() checkErrors__line(__LINE__)
inline void checkErrors__line(unsigned int __line){
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("[ERROR] (Line %d) %s: %s\n", __line ,cudaGetErrorName(err), cudaGetErrorString(err));
		exit(1);
	}
}


/* Calls cudaMalloc and cudaMemcpy (optionally) */
inline void allocateAndLoad(void **dev_mem, size_t _size, void *cpy_buffer = NULL){
	cudaMalloc(dev_mem, _size);
	checkErrors();
	if(cpy_buffer == NULL)
		return;
	cudaMemcpy(*dev_mem, cpy_buffer, _size, cudaMemcpyHostToDevice);
	checkErrors();
}


/// Parameter Calculation Kernels //////////////////////////////

/* Returns the largest number out of the two arguments.
 * Defined only for readability */

__device__ unsigned maxInt(unsigned _a, unsigned _b){
	return _b > _a ? _b : _a;
}


/* Finds and returns in ret the largest number equal or smaller to _maxBlockCount that divides _input. 
 * To do this, all threads who's ID is smaller than _maxBlockCount check if their ID +1 divides
 * _input and if it does write to local memory their ID, 0 if not.
 * Then, in parallel, the threads compute the maximum number in local memory. */

__global__ void findBlockCount(unsigned int _input, unsigned int _maxBlockCount, unsigned int *ret){
	__shared__ unsigned int Ld[1024];

	// Initialize shared memory
	Ld[threadIdx.x] = 0;
	
	if(threadIdx.x < _maxBlockCount){ // Enter only if threadId allows it
		if (_input % (threadIdx.x + 1) == 0) // Check modulo
			Ld[threadIdx.x] = threadIdx.x + 1;
	}
	__syncthreads();
	
	// Find how many steps a reduction tree for _maxBlockCount numbers needs (use larger power of 2 closer to _maxBlockCount)
	unsigned int steps = 1;
	while((1<<steps) < _maxBlockCount){ // Loop breaks once 1<<steps is larger than _maxBlockCount
		steps++;
	}

	// Use reduction to find max in Ld
	unsigned int stride;
	int i = steps-1;
	do{
		stride = 1 << i;
		if (threadIdx.x < stride) // Thread should not participate; Go to barrier
			Ld[threadIdx.x] = maxInt(Ld[threadIdx.x], Ld[threadIdx.x + stride]);
		__syncthreads();
		i--;
	} while (i >= 0);

	if(threadIdx.x == 0) // One thread writes the result to global memory
		ret[0] = Ld[0];
}


/* Structure containing parameters for a kernel.
 * Only allows 1D block dimensions. */
struct kernelParameters1D{
	unsigned blockSize;
	unsigned blockCount_for_lines; 		// Matches number of lines (Y)
	unsigned blockCount_for_columns; 	// Matches number of columns (X)
};


/* Given the size of the input matrix determines the
 * optional number of blocks to use. Grids are 1D.
 * Calculated parameters are returned in ret. */
void findKernelParameters1D(unsigned int _lines, unsigned int _columns, struct kernelParameters1D *ret){
	// Block size is standard
	unsigned int blockSize = BLOCK_SIZE; 
	ret->blockSize = blockSize;

	// Find device capabilities
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);	checkErrors();
	unsigned int smCount = deviceProp.multiProcessorCount;
	//unsigned int blocksPerSM = deviceProp.maxBlocksPerMultiProcessor; 
	unsigned int threadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
	//unsigned int maxBlockSize = deviceProp.maxThreadsPerBlock;
	//unsigned int globalMemorySize = deviceProp.totalGlobalMem;
	//unsigned int blockSharedMem = deviceProp.sharedMemPerBlock;

	// Find block count based on _lines

	// Given a predefined number of threads per block, each block will
	// receive the same amount of lines to compute.

	// Maximum number of blocks the device can properly operate with
	unsigned int maxBlockCount = threadsPerSM / blockSize * smCount; 

	// Ideally, there are as many blocks as there are lines, giving just one line to each block
	unsigned int blockCount = _lines;

	// If there aren't enough blocks to accommodate this, decrease the number of blocks so that all blocks
	// still share an equal amount of lines.
	if(blockCount > maxBlockCount){
		// Call a kernel that will determine the largest blockCount that divides _lines and store the result in dev_blockCount
		unsigned int *dev_blockCount;
		cudaMalloc((void**)&dev_blockCount, sizeof(unsigned int));
		checkErrors();
		findBlockCount<<<1, maxBlockCount>>>(_lines, maxBlockCount, dev_blockCount);
		checkErrors();
		cudaMemcpy(&blockCount, (void*)dev_blockCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		checkErrors();
		cudaFree(dev_blockCount);
		checkErrors();
	}

    printf("* %u blocks (out of %u), %zu lines per block.\n", blockCount, maxBlockCount, _lines/(size_t)blockCount);

    ret->blockCount_for_lines = blockCount;

	// Find block for count based on _columns

	// The operation is identical with above, just switch _lines with _columns
	blockCount = _columns;

	if(blockCount > maxBlockCount){
		unsigned int *dev_blockCount;
		cudaMalloc((void**)&dev_blockCount, sizeof(unsigned int));
		checkErrors();
		findBlockCount<<<1, maxBlockCount>>>(_columns, maxBlockCount, dev_blockCount);
		checkErrors();
		cudaMemcpy(&blockCount, (void*)dev_blockCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		checkErrors();
		cudaFree(dev_blockCount);
		checkErrors();
	}

    printf("* %u blocks (out of %u), %zu lines per block.\n", blockCount, maxBlockCount, _columns/(size_t)blockCount);


	ret->blockCount_for_columns = blockCount;	
}




/// File Loading functions ///////////////////////////////

/* Loads a test file.
 * Returns the dimensions of the matrix in X, Y and
 * the matrix itself in _matrix. The vector is stored in _vector. */
int loadTest(unsigned int _test_no, unsigned int *X, unsigned int *Y, float **_matrix, float **_vector){
	FILE *fp;
	char filename[13];
	char fid_str[4];
	if(_test_no > 999)
		return -1;
		
	sprintf(fid_str, "%d", _test_no);	
	strcpy(filename, "test/");
	strcat(filename, fid_str);
	
	fp = fopen((const char*)filename, "rb");
	if(fp!=NULL){
		// Read dimensions
		fread(X, sizeof(unsigned int), 1, fp);
        fread(Y, sizeof(unsigned int), 1, fp);
		// Skip padding
		fseek(fp, 8, SEEK_CUR); 
		// Allocate memory
		*_matrix = (float*)malloc(sizeof(float)*(*X)*(*Y));
		*_vector = (float*)malloc(sizeof(float)*(*X));
		// Load values
		fread(*_matrix, sizeof(float), (*X)*(*Y), fp);
		fread(*_vector, sizeof(float), (*X), fp);
	}
	else{
		printf("Error opening output file\n");
		return 1000;
	}
	fclose(fp);
	
	return 0;
}

/* Exports the content of _array to a file named "_test_no-bin.out" */
void export_results(const char *_test_no, float *_array, unsigned int _size){
    char binout[12]; // _test_no will be 3 char long under normal operation
    strcpy(binout, _test_no);
    strcat(binout, "-out.bin");
	printf("Done. Writing to %s.\n", binout);
	FILE *o = fopen(binout, "wb");
	if(o == NULL){
		printf("Failed to write output.\n");
	}
	else{
		fwrite(_array, sizeof(float), _size, o);
		fclose(o);
	}
}

/* Checks if input dimensions can be used.
 * Returns 0 if X and Y can be used and 1 otherwise */
inline int input_cant_be_used(unsigned int X, unsigned int Y, unsigned int A){
	if(X % A != 0){
        printf("error: line width of %u is not divided by %u. This file cannot be used.\n", X, BLOCK_SIZE);
		return 1;
	}
    if(Y % A != 0){
        printf("error: number of lines (%u) are not divided by %u. This file cannot be used.\n", Y, BLOCK_SIZE);
		return 1;
	}
	return 0;
}
