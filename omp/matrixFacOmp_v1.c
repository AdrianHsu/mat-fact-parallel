#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define MAXCHAR 1000
char str[MAXCHAR];
FILE *fp;
char fileName[128];
char * pch;

int iterations;
double a;
int N;
int M;
int K;
int Z;


void dot2(double mat1[N][K], double mat2[K][M], double res[N][M]){
	int i, j, k;
    	for (i = 0; i < N; i++) {
        	for (j = 0; j < M; j++) {
            	res[i][j] = 0;
            	for (k = 0; k < K; k++)
                	res[i][j] += mat1[i][k] * mat2[k][j];
        	}
    	}
}

void matrix_factorization2(double** R,
              double prediction[Z], double errors[Z],
              int nonZeroRow[Z],
              int nonZeroColumn[Z],
              double** P, double** Q,
              double** tempP, double** tempQ,
              int N, int M, int K,
              int Z,
              double alpha){
        	int n,m,k,z;

        omp_set_num_threads(2);
    
        #pragma omp parallel for private(k, n, m)
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                tempP[n][k] = P[n][k];
            }
            for (int m = 0; m < M; m++) {
                tempQ[k][m] = Q[k][m];
            }
            
            
            /* for only main thread */
    
        }
    
        #pragma omp parallel for private(z, k) schedule(static)
        for (int z = 0; z < Z; z++) {
            //int tid = omp_get_thread_num();
            //printf("Thread = %d\n", tid);
            prediction[z] = 0;
            errors[z] = 0;
            for (int k = 0; k < K; k++) {
                prediction[z] += P[nonZeroRow[z]][k] * Q[k][nonZeroColumn[z]];
            }
            errors[z] = R[nonZeroRow[z]][nonZeroColumn[z]] - prediction[z];
        }
    
        #pragma omp parallel for private(z, k) collapse(2) schedule(static)
        for (int z = 0; z < Z; z++) {
            for (int k = 0; k < K; k++) {
                #pragma omp atomic
                P[nonZeroRow[z]][k] += alpha * (2 * errors[z] * tempQ[k][nonZeroColumn[z]]);
                #pragma omp atomic
                Q[k][nonZeroColumn[z]] += alpha * (2 * errors[z] * tempP[nonZeroRow[z]][k]);
            }
        }
    
    
    
}


int main(){
	// read input
	printf("enter file name\n");
	scanf("%123s", fileName);
	fp = fopen(fileName, "r");
	if (fp == NULL){
        printf("Could not open file %s",fileName);
        return 1;
    }
    // iterations
    if (fgets(str, MAXCHAR, fp) != NULL) {
    	iterations = atoi(str);
    }
    // alpha
    if (fgets(str, MAXCHAR, fp) != NULL) {
    	a = atof(str);
    }
    // latent features
    if (fgets(str, MAXCHAR, fp) != NULL) {
    	K = atoi(str);
    }
    // users, items and non-zero elements
    if (fgets(str, MAXCHAR, fp) != NULL) {
    	pch = strtok (str," ,.-");
    	N = atoi(str);
    	pch = strtok (NULL," ,.-");
    	M = atoi(pch);
    	pch = strtok (NULL," ,.-");
    	Z = atoi(pch);
    }
    printf("Users: %d, Items: %d, Features: %d, Non-zero elements: %d\n", N, M, K, Z);
    int nonZeroRow[Z];
    int nonZeroColumn[Z];
    // initialize the matrix R
    // double R[N][M];
    // acquire memory for 2D array - float R[N][M];
    double **R = malloc(N * sizeof * R);
    for (size_t i = 0; i < N; i++) {
      R[i] = malloc(M * sizeof * R[i]);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            R[i][j] = 0;
        }
    }
    // non-zero elements
    int nonZeroIndex = 0;
    while (fgets(str, MAXCHAR, fp) != NULL) {
		pch = strtok (str," ,.-");
    	int u = atoi(str);
    	pch = strtok (NULL," ,.-");
    	int i = atoi(pch);
    	pch = strtok (NULL," ,.-");
    	int v = atoi(pch);
        nonZeroRow[nonZeroIndex] = u;
        nonZeroColumn[nonZeroIndex++] = i;
    	R[u][i] = v;
    }
	fclose(fp);

    double prediction[Z];
    double errors[Z];
    // double tempP[N][K];
    // double tempQ[K][M];
	// double P[N][K];
	// double Q[K][M];
	// float Q_Result[M][K];

    double **P = malloc(N * sizeof * P);
    double **Q = malloc(M * sizeof * Q);
    double **Q_Result = malloc(M * sizeof * Q_Result);
    double **tempP = malloc(N * sizeof * P);
    double **tempQ = malloc(M * sizeof * Q);

    // acquire memory for 2D array - P, Q and Q_Result;
    for (size_t i = 0; i < N; i++) {
      P[i] = malloc(K * sizeof * P[i]);
      tempP[i] = malloc(K * sizeof * tempP[i]);
    }

    for (size_t i = 0; i < M; i++) {
      Q_Result[i] = malloc(K * sizeof * Q_Result[i]);
      Q[i] = malloc(K * sizeof * Q[i] );
      tempQ[i] = malloc(K * sizeof * tempQ[i]);
    }
	double time_spent = 0.0;
	for(int i = 0; i < N;i++) {
		for(int j = 0; j < K;j++) {
			P[i][j] = ((float)rand()) / ((float)RAND_MAX)*1.0;
		}
	}
	for(int i = 0; i < M;i++) {
        for(int j = 0; j < K;j++) {
                Q[i][j] = ((float)rand()) / ((float)RAND_MAX)*1.0;
        }
    }
	
    double begin = omp_get_wtime();
	for (int iteration = 0; iteration < iterations; iteration++) {
        matrix_factorization2(R, prediction, errors, nonZeroRow, nonZeroColumn, P,Q,tempP, tempQ, N,M,K,Z, a);
	}
	double end = omp_get_wtime();

	time_spent += (double)(end - begin);

    printf("Number of threads uses: 2.\n");
    printf("Users: %-5dItems: %-5dFeatures: %-5dIterations: %-8dalpha: %-5lf\n", N, M, K, iterations, a);
    printf("OMP: Time elpased is %lf seconds\n", time_spent);
 //    printf("Predict R:\n");
	
	// double result[N][M];
	// dot2(P, Q, result);

	// for (int i = 0; i < N; i++){
	// 	for (int j = 0; j < M; j++){
	// 		printf("%.6f", result[i][j]);
 //            printf("    ");
	// 	}
	// 	printf("\n");
	// }
	return 0;
}
