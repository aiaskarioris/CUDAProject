#define BLOCK_SIZE 128
#define BLOCK_SIZE_LOG2 7

#include "ex3.cuh" // ex3.cuh uses BLOCK_SIZE; Must be included after BLOCK_SIZE definition

#define IN_VEC_BLOCK_COUNT 14
#define IN_VEC_TILE_SIZE (BLOCK_SIZE * IN_VEC_BLOCK_COUNT) // <-- removing the ( ) will cause the program to work wrong and will be really hard to find out why
#define MAX_LINE_COUNT  128

/// CUDA Kernels ///////////////////////////////////////

/* Multiplies global_matrix with global_vector. Result is stored in global_out */
__global__ void mult_MatByVec(unsigned _width, unsigned _lines, float *global_matrix, float *global_vector, float *global_out){
	// Tile for locally storing parts of the vector: 14 x 128 Floats -> 7KB
    __shared__ float Vds[IN_VEC_TILE_SIZE];

    // Accumulator for each line. Dynamic allocation is not possible in kernels so a safe static value is chosen (MAX_LINE_COUNT).
    // Used only by thread 0
    // 128 floats -> 512B
    __shared__ float Rds[MAX_LINE_COUNT];
    for (unsigned i = threadIdx.x; i < MAX_LINE_COUNT; i+= blockDim.x) 
        Rds[i] = 0.0;

    // Float accumulator for each thread. Visible to other threads as well
    // 128 Floats -> 512B
    __shared__ float a[BLOCK_SIZE];

    // Lines the block must compute
    unsigned linesPerBlock = _lines / gridDim.x;
    // First line the block gets
    unsigned startLine = blockIdx.x * linesPerBlock;

    // Number of parts a line must be broken into
    unsigned partCount;
    if(_width > IN_VEC_TILE_SIZE)
        partCount = _width / IN_VEC_TILE_SIZE;
    else
        partCount = 1;

    // Counter of blocks processed
    unsigned int j = 0;
    // General purpose variable
    unsigned int i;
    // Indexes for addressing global and local (shared) memory respectively
    size_t globalIndex, localIndex;

    
    // _width is split in parts. Move through parts and calculate lines accordingly
    unsigned currentPart = 0;
    do{
        // Copy part of vector
        for (j = 0; j < IN_VEC_BLOCK_COUNT; j++){ // Copy until shared memory is full (or vector is copied whole)
            globalIndex = (currentPart * IN_VEC_TILE_SIZE) + (j * blockDim.x) + threadIdx.x; 
            localIndex = (j * blockDim.x) + threadIdx.x;

            // Break for-loop if the whole vector has been copied (can be true if _width < IN_VEC_TILE_SIZE)
            if(globalIndex >= _width){
                break;
            }

            // Copy
            Vds[localIndex] = global_vector[globalIndex];

        }
        __syncthreads();

        // Loop through lines. Note each element from global_matrix is read only once so there's no need to copy it to shared memory
        unsigned currentLine;
        for (currentLine = startLine; currentLine < startLine + linesPerBlock; currentLine++){

            for (j = 0; j < IN_VEC_BLOCK_COUNT; j++){ // For every 128-number block in the copied part... 
                // Break if the whole line was processed
                if (j * blockDim.x >= _width) // Works for _width < VEC_TILE_SIZE, won't work if _width > IN_VEC_TILE_SIZE but _width isn't a multiple of tile size
                    break;

                // Get index for accessing an element in the matrix
                globalIndex = currentLine * _width; // Select line
                globalIndex += (j * blockDim.x) + (currentPart * IN_VEC_TILE_SIZE); // Select current part and block
                globalIndex += threadIdx.x; 
                // Get index for accessing an element from the (locally stored) vector
                localIndex = (j * blockDim.x) + threadIdx.x;

                a[threadIdx.x] = Vds[localIndex] * global_matrix[globalIndex];
                __syncthreads(); // Wait for every thread

                
                // Sum-up all elements of a[] using reduction
                for (i = BLOCK_SIZE / 2; i > 0; i /= 2){
                    if(threadIdx.x < i){
                        a[threadIdx.x] += a[threadIdx.x + i];
                    }
                    __syncthreads();
                }

                // Keep line's result in Rds; It will be used again once the current line is processed again
                if(threadIdx.x == 0){
                    i = currentLine - startLine;
                    Rds[i] += a[0];
                }
                __syncthreads();
            }
        }
        currentPart++;
    } while (currentPart < partCount);


    // Copy Rds (results from each line) to global_result
    for (i = threadIdx.x; i < linesPerBlock; i += blockDim.x){ 
        j = (linesPerBlock * blockIdx.x) + i;
        global_out[j] = Rds[i];
    }
}

/* Variation of mult_MatByVec that returns the product of the transposed matrix with a vector.
 * To do this, the elements of the vector are multiplied with the columns of the matrix, instead of the lines. */
__global__ void mult_TransByVec(unsigned int _column_count, unsigned int _column_height, float *global_matrix, float *global_vector, float *global_out){
    // Tile for locally storing parts of the vector
    __shared__ float Vds[IN_VEC_TILE_SIZE];

    // Accumulator for each line. Dynamic allocation is not possible in kernels so a safe static value is chosen (MAX_LINE_COUNT).
    // Used only by thread 0
    __shared__ float Rds[MAX_LINE_COUNT];
    for (unsigned i = threadIdx.x; i < MAX_LINE_COUNT; i+= blockDim.x) // Init. Rds with 0
        Rds[i] = 0.0;

    // Float accumulator for each thread. Visible to other threads as well
    __shared__ float a[BLOCK_SIZE];

    // Columns the block must compute
    unsigned columnsPerBlock = _column_count / gridDim.x;
    // First column the block gets
    unsigned startColumn = blockIdx.x * columnsPerBlock;

    // Number of parts a column must be broken into
    unsigned partCount;
    if(_column_height > IN_VEC_TILE_SIZE)
        partCount = _column_height / IN_VEC_TILE_SIZE;
    else
        partCount = 1;

    // Counter of blocks processed
    unsigned int j = 0;
    // General purpose variable
    unsigned int i;
    // Indexes for addressing global and local (shared) memory respectively
    size_t globalIndex, localIndex;

    
    // _column_height is split to parts. Move through parts and calculate columns accordingly
    unsigned currentPart = 0;
    do{
        // Copy part of vector
        for (j = 0; j < IN_VEC_BLOCK_COUNT; j++){ 
            globalIndex = (currentPart * IN_VEC_TILE_SIZE) + (j * blockDim.x) + threadIdx.x; 
            localIndex = (j * blockDim.x) + threadIdx.x;

            if(globalIndex >= _column_height){
                break;
            }

            Vds[localIndex] = global_vector[globalIndex];

        }
        __syncthreads();

        // Loop through columns. Note each element from global_matrix is read only once so there's no need to copy it to shared memory
        unsigned currentColumn;
        for (currentColumn = startColumn; currentColumn < startColumn + columnsPerBlock; currentColumn++){

            for (j = 0; j < IN_VEC_BLOCK_COUNT; j++){
                if (j * blockDim.x >= _column_height)
                    break;

                // Get index for accessing an element from the matrix
                globalIndex = currentColumn; // Select column
                globalIndex += ((j * blockDim.x) + (currentPart * IN_VEC_TILE_SIZE)) * _column_count; // Select part and block (i.e. select a line)
                globalIndex += threadIdx.x * _column_count; // Select a specific element in column (i.e. select a line)
                
                // Get index for accessing an element from the (locally stored) vector
                localIndex = (j * blockDim.x) + threadIdx.x;

                a[threadIdx.x] = Vds[localIndex] * global_matrix[globalIndex];
                __syncthreads();

                // Sum-up all elements of a[] using reduction
                for (i = BLOCK_SIZE / 2; i > 0; i /= 2){
                    if(threadIdx.x < i){
                        a[threadIdx.x] += a[threadIdx.x + i];
                    }
                    __syncthreads();
                }

                // Keep line's result in Rds; It will be used again once the current line is processed again
                if(threadIdx.x == 0){
                    i = currentColumn - startColumn;
                    Rds[i] += a[0];
                }
                __syncthreads();
            }
        }

        currentPart++;
    } while (currentPart < partCount);


    // Copy Rds (results from each line) to global_result
    for (i = threadIdx.x; i < columnsPerBlock; i += blockDim.x){ 
        j = (columnsPerBlock * blockIdx.x) + i;
        global_out[j] = Rds[i];
    }

}

int main(int argc, char **argv){
    unsigned int test_no;
    // Input dimensions
	unsigned X, Y;
    // Sizes of inputs in numbers
    size_t matrixSize, vectorSize, mat_vec_productSize;
    // Sizes of inputs in bytes
    size_t matrixBytes, vectorBytes,  mat_vec_productBytes;
    // Pointers for storing the inputs
    float *matrix, *vector;
    // CUDA Event objects
    cudaEvent_t event[6];

    printf("CUDA Exc. II\n");
	if(argc != 2){
        printf("usage: ex3_2 [test_number]\n\n");
        return 0;
    }

    // Get events ready
    for (unsigned int e = 0; e < 6; e++)
        cudaEventCreate(&(event[e]));

    // Open and load file
    test_no = atoi(argv[1]);
    if(loadTest(test_no, &X, &Y, &matrix, &vector)){
        printf("error: %d is not a valid test number\n\n", test_no);
        return -1;
    }

    // Check X & Y are multiples of BLOCK_SIZE
    if(input_cant_be_used(X, Y, BLOCK_SIZE)){
        free(matrix);
        free(vector);
        return 1;
    }

    // Calculate basic variables
    matrixSize = X * Y;
    matrixBytes = matrixSize * sizeof(float);
    vectorSize = X;
    vectorBytes = vectorSize * sizeof(float);
    mat_vec_productSize = Y;
    mat_vec_productBytes = mat_vec_productSize * sizeof(float);
    printf("Loaded test %3d:\n\tMatrix: %zu numbers (%ux%u)\n\tVector: %zu numbers (1x%u)\n",
             test_no, matrixSize, X, Y, vectorSize, X);

    // Start timer
    cudaEventRecord(event[0]);

    // Prepare device buffers
    float *dev_matrix, *dev_vector, *dev_mat_vecProduct;
    allocateAndLoad((void**)&dev_matrix, matrixBytes, matrix);
    allocateAndLoad((void**)&dev_vector, vectorBytes, vector);
    allocateAndLoad((void**)&dev_mat_vecProduct, mat_vec_productBytes, NULL);
    
    // Calculate kernel parameters
    struct kernelParameters1D params;
    findKernelParameters1D(Y, X, &params);

    // Calculate A*x; The result will be stored in dev_mat_vecProduct but will never be copied to the host as it is not needed
    cudaEventRecord(event[1]);
    mult_MatByVec<<<params.blockCount_for_lines, params.blockSize>>>(X, Y, dev_matrix, dev_vector, dev_mat_vecProduct);
    checkErrors();
    cudaEventRecord(event[2]);


    // Multiply A's transpose with the vector produced by mult_MatByVec
    cudaEventRecord(event[3]);
    mult_TransByVec<<<params.blockCount_for_columns, params.blockSize>>>(X, Y, dev_matrix, dev_mat_vecProduct, dev_vector); // dev_vector's content will be overwritten with the result vector
    checkErrors();
    cudaEventRecord(event[4]);

    // Get results
    cudaMemcpy(vector, dev_vector, vectorBytes, cudaMemcpyDeviceToHost);
    checkErrors();

    // End timer
    cudaEventRecord(event[5]);

    // Print time information
    float time[3];
    cudaEventElapsedTime(&(time[0]), event[1], event[2]);
    cudaEventElapsedTime(&(time[1]), event[3], event[4]);
    cudaEventElapsedTime(&(time[2]), event[0], event[5]);
    printf("\n* Kernel #1 time:  \t%.4f ms\n* Kernel #2 time:  \t%.4f ms\n", time[0], time[1]);

    // Destroy events
    for (unsigned int e = 0; e < 6; e++)
        cudaEventDestroy(event[e]);

    // Output debug results
    export_results(argv[1], vector, X);
    
    // De-allocate
    free(vector);
    free(matrix);
    
	return 0;
}
