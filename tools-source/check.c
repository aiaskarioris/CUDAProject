#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Output validation tool ===============================================
 * Loads an input test file and the output of the main program
 * to check that the program's results are correct
 ================================================================ */

// Loads input test file
int loadTest(unsigned int _test_no, unsigned int *X, unsigned int *Y, float **matrix, float **vector){
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
		fseek(fp, 8, SEEK_CUR); // skip padding
		// Allocate memory
		*matrix = (float*)malloc(sizeof(float)*(*X)*(*Y));
		*vector = (float*)malloc(sizeof(float)*(*X));
		// Read A
		fread(*matrix, sizeof(float), (*X)*(*Y), fp);
		fread(*vector, sizeof(float), (*X), fp);
	}
	else{
		printf("Error opening output file\n");
		return 1000;
	}
	fclose(fp);
	
	return 0;
}

// Loads ex3's output
int loadOutput(const char *_test_no, size_t _N, float **A){
    char binout[12];
    strcpy(binout, _test_no);
    strcat(binout, "-out.bin");

    FILE *fp;
    fp = fopen(binout, "rb");
	if(fp!=NULL){
        *A = (float *)malloc(_N * sizeof(float));
        size_t bytes_read;
        bytes_read = fread(*A, sizeof(float), _N, fp);
        if(bytes_read != _N){
            free(A);
            printf("Expected %zu numbers but read %zu. No check will be performed.\n", _N, bytes_read);
            fclose(fp);
            return -2;
        }
    }
    else{
        printf("Could not open %s\n", binout);
        return 1000;
    }
    fclose(fp);

    return 0;
}


int main(int argc, char **argv){
    if(argc != 2){
        printf("usage: check [test number] \n");
        return 1;
    }

    unsigned int X, Y;
    size_t N;
    size_t matrixBytes, vectorBytes, innerVectorBytes;
    float *inmat, *invec, *cudaout, *correct, *innerVec;
    clock_t time[6];

    time[0] = clock();

    // Load Test
    unsigned int test_no = atoi(argv[1]);
    if(loadTest(test_no, &X, &Y, &inmat, &invec))
        return 1;
    N = X * Y;
    matrixBytes = N * sizeof(float);
    vectorBytes = X * sizeof(float);
    innerVectorBytes = Y * sizeof(float);

    // Load CUDA Out
    int cuda_loaded = 0;
    if(loadOutput(argv[1], X, &cudaout)){
        printf("Only correct output will be generated.\n");
    }
    else
        cuda_loaded = 1;

    printf("Input matrix:\t\tOK [%ux%u (%zu)]\n", X, Y, N);
    printf("Input vector:\t\tOK [%ux%u (%u)]\n", X, 1, X);
    if(cuda_loaded)
        printf("Output from cuda:\tOK [%ux%u (%u)]\n", X, 1, X);
    else
        printf("Output from cuda:\tNONE [%ux%u (%u)]\n", X, 1, X);
    printf("(Input: %zuKB\tOutput: %zuKB)\n", (matrixBytes+vectorBytes) / 1024, vectorBytes / 1024);

    printf("\nCalculating Matrix by vector results...");
    innerVec = (float*)malloc(innerVectorBytes);
    unsigned int line, pos;
    time[1] = clock();
    for (line = 0; line < Y; line++){
        innerVec[line] = 0;
        for (pos = 0; pos < X; pos++){
            innerVec[line] += invec[pos] * inmat[line * X + pos];
            //correct[line] += invec[pos];
        }
    }
    time[2] = clock();
    printf("OK\n");

    free(invec);

    printf("Calculating final vector...");
    correct = (float*)malloc(vectorBytes);
    time[3] = clock();
    for (pos = 0; pos < X; pos++) {
        correct[pos] = 0;
    }

    for (line = 0; line < Y; line++){
        for (pos = 0; pos < X; pos++){
            correct[pos] += innerVec[line] * inmat[line * X + pos];
            //correct[pos] += innerVec[line];
        }
    }
    time[4] = clock();
    printf("OK\n");

    if(cuda_loaded){
        printf("Checking...");
        size_t errors = 0;
        float diff;
        for (size_t i = 0; i < X; i++){
            //printf("%.2f -- %.2f\n", cudaout[i], correct[i]);
            diff = correct[i] - cudaout[i];
            if(diff < 0)
                diff *= -1;
            if(diff > 0.5)
                errors++;
        }

        if(errors){
            printf("FAILED\n%zu errors where encountered.\n", errors);
        }
        else{
            printf("OK\n");
        }
    }

    time[5] = clock();

    // Display times
    double time_ell[4];
    time_ell[0] = (time[2] - time[1]) / ((double)CLOCKS_PER_SEC) * 1000.0;
    time_ell[1] = (time[4] - time[3]) / ((double)CLOCKS_PER_SEC) * 1000.0;
    time_ell[2] = time_ell[0] + time_ell[1];
    time_ell[3] = (time[5] - time[0]) / ((double)CLOCKS_PER_SEC) * 1000.0;
    printf("\nTime for step 1:\t\t%.3f ms\nTime for step 2:\t\t%.3f ms\nTime for both steps:\t%.3f ms\nTotal time ellapsed:\t%.3f ms\n\n", time_ell[0], time_ell[1], time_ell[2], time_ell[3]);


    FILE *o = fopen("checkout.bin", "wb");
    if(o == NULL){
        printf("Failed to write output.\n");
    }
    else{
        printf("Writing to checkout.bin: %zuKB\n", X * sizeof(float) / 1024);
        fwrite(correct, sizeof(float), X, o);
        fclose(o);
    }

    free(inmat);
    if(cuda_loaded)
        free(cudaout);
    free(correct);
    free(innerVec);

    return 0;
}
