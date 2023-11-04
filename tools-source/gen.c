#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>

/* Input Generation Tool ===========================================
 * Tool used to create and browse test files.
 * New files are outputed to test/ and the program fails
 * if the directory isn't found.
 ================================================================ */


#define MAX_NUM 65536
#define true 1
#define false 0

void usage(){
	printf("Usage: gen X Y\n");
	printf("       gen -l [test]: Show metadata and list numbers\n");
	printf("       gen -d [test]: Same as -l but doesn't list numbers\n\n");
}

void genBuffer(size_t N, float **_buffer, size_t *_buffersize){
	// Allocate memory
	size_t size = N * sizeof(float);
	float *buffer = (float*)malloc(size);
		
	// Generate numbers
	size_t i;
	for(i = 0; i < N; i++){
		// Add fraction
		buffer[i] = (rand() % 9 == 0) ? 1 : ((unsigned int)rand() % 1024) / 10000.0;

		// Add sign
		buffer[i] *= (rand() % 3 == 0) ? -1 : 1; 

		// Zero
		buffer[i] *= (rand() % 16 == 0) ? 0 : 1; 
		
	}
	
	// Pass the buffer to main
	*_buffersize = size;
	*_buffer = buffer;
	
}

int writeFile(float *_matrix, float *_vector, unsigned _X, unsigned _Y){
	FILE *fp;
	char filename[13];
	char pid_c[5];
	int fileid = getpid()%1000;
	
	strcpy(filename, "test/");
	sprintf(pid_c, "%d", fileid);
	
	strcat(filename, pid_c);
	
	size_t zero = 0;
	fp = fopen((const char*)filename, "wb");
	if(fp!=NULL){
		fwrite(&_X, sizeof(unsigned), 1, fp);
		fwrite(&_Y, sizeof(unsigned), 1, fp);
		fseek(fp, 16-2*sizeof(unsigned int), SEEK_CUR); // Padding
		fwrite(_matrix, sizeof(float), _X*_Y, fp); 
		fwrite(_vector, sizeof(float), _X, fp);

	}
	else{
		printf("Error opening output file\n");
		return 1000;
	}
	fclose(fp);
	
	return fileid;
}

float sum(float *A, size_t N){
	float ret = 0;
	for (unsigned int i = 0; i < N; i++)
		ret += A[i];
	return ret;
}

void readFile(const char *test_no, int _list){
	FILE *fp;
	char filename[13];
	
	strcpy(filename, "test/");
	strcat(filename, test_no);

	size_t N;
	unsigned int X, Y;
	float *A, *B;
	size_t check;
	fp = fopen((const char *)filename, "rb");
	if(fp != NULL){
		fread(&X, sizeof(unsigned int), 1, fp);
		fread(&Y, sizeof(unsigned int), 1, fp);
		fseek(fp, 16-2*sizeof(unsigned int), SEEK_CUR);
		N = X * Y;
		A = (float*)malloc(sizeof(float) * N);
		B = (float *)malloc(sizeof(float) * X);
		check = fread(A, sizeof(float), N, fp);
		check += fread(B, sizeof(float), X, fp);
		fclose(fp);

		if(check != N + X){
			printf("[ERROR] This file is corrupted (read %zu/%zu)\n", check, (size_t)N + X);
		}

		printf("Test %.3s:\n\tMatrix: %ux%u (%zu numbers, %zuKB)\n", test_no, X, Y, N, N * sizeof(float) / 1024);
		printf("\tVector: %u (%zuKB) (Sum: %.6f)\n", X, X*sizeof(float)/1024, sum(B, X));
		printf("\tOutput vector: %u (%zuKB)\n", Y, Y*sizeof(float)/1024);

		if(_list){
			size_t i;
			/*
			for (i = 0; i < Y; i++){
				printf("%zd: %.3f\n", i, sum(&(A[i * X]), X));
			}
			*/
			printf("Matrix:\n");
			for (i = 0; i < N; i+= 4)
			{
				printf("%2.4f\t%2.4f\t%2.4f\t%2.4f", A[i], A[i+1], A[i+2], A[i+3]);
				printf("\n");
			}
			printf("\n");

			printf("Vector:\n");
			for (i = 0; i < X; i++)
			{
				printf("%2.4f\t", B[i]);
			}
			
			printf("\n");
		}
		free(A);
		free(B);
	}
	else{
		printf("No such file was found\n\n");
	}
}


int main(int argc, char **argv){
	srand((size_t)(getpid() ^ 0x7188F29E));

	size_t N;
	unsigned int X, Y;
	size_t matrixSize, vectorSize;
	float *matrix, *vector;
	

	if(argc == 2 && !strcmp(argv[1], "--help")){
		usage();
		return 0;
	}
	else if(argc == 3 && !strcmp(argv[1], "-l")){
		readFile(argv[2], 1);
		return 0;
	}
	else if(argc == 3 && !strcmp(argv[1], "-d")){
		readFile(argv[2], 0);
		return 0;
	}
	else if(argc == 3){ // ===================================================================
		X = strtoul(argv[1], NULL, 0); // const char* to size_t
		Y = strtoul(argv[2], NULL, 0); // const char* to size_t 		
		N = X * Y;
		if (N > MAX_NUM)
		{
			printf("gen: N=%zu out of limits; Must be smaller than %u (0x%x)\n\n", N, MAX_NUM, MAX_NUM);
			return 1;
		}
		printf("Generating matrix (%zu numbers)\n", N);
		genBuffer(N, &matrix, &matrixSize);
		printf("Generating vector (%u numbers)\n", X);
		genBuffer(X, &vector, &vectorSize);
	
	} // ====================================================================================
	else{
		usage();
		return 0;
	}
	
	int fileid = writeFile(matrix, vector, X, Y);
	printf("Created %d\n\n", fileid);
	free(matrix);
	free(vector);

	return 0;
}
