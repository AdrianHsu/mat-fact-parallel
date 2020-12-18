#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAXCHAR 1000
char str[MAXCHAR];
FILE *fp;
char fileName[128];
char * pch;

int iterations;
float a;
int N;
int M;
int K;
int Z;


void transpose(float** A, float** B) 
{  
    for (int i = 0; i < K; i++){ 
        for (int j = 0; j < M; j++){ 
            B[i][j] = A[j][i]; 
        }
    }
} 


void transpose2(float** A, float** B)
{  
    for (int i = 0; i < K; i++){ 
        for (int j = 0; j < M; j++){ 
            B[j][i] = A[i][j];
        }
    }
} 

float dot(float** A, float **B, int rowA, int colB){
	float result = 0;
	for (int i = 0; i < K; i++){
		result += A[rowA][i] * B[i][colB];
	}
	return result;
}

void dot2(float** mat1, float **mat2, float res[N][M]){
	int i, j, k;
    	for (i = 0; i < N; i++) {
        	for (j = 0; j < M; j++) {
            	res[i][j] = 0;
            	for (k = 0; k < K; k++)
                	res[i][j] += mat1[i][k] * mat2[k][j];
        	}
    	}
}

void matrix_factorization(float** R, float** P, float** Q, int K, int steps, float alpha, float beta,  float** Q_Result){
	// float Q_T[K][M];
	float **Q_T = malloc(K * sizeof * Q_T);
	for (size_t i = 0; i < K; i++) {
	  Q_T[i] = malloc(M * sizeof * Q_T[i]);
	}
	transpose(Q, Q_T);

	for (int step = 0; step < steps; step++){
		for (int i = 0; i < N; i++){
			for (int j = 0; j < M; j++){
				if (R[i][j] > 0){
					float eij = R[i][j] - dot(P, Q_T, i, j); 
					for (int k = 0; k < K; k++){
						P[i][k] = P[i][k] + alpha * (2 * eij * Q_T[k][j]);
						Q_T[k][j] = Q_T[k][j] + alpha * (2 * eij * P[i][k]);

					}
				}
			}
		}
		float eR[N][M];
		dot2(P,Q_T,eR);
		float e = 0;
		for (int i = 0; i < N; i++){
			for (int j = 0; j < M; j++){
				if (R[i][j] > 0){
					e = e + (R[i][j] - dot(P, Q_T, i, j)) * (R[i][j] - dot(P, Q_T, i, j));
				}
			}
		}
	}
	transpose2(Q_T, Q_Result);
}



int main() {
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

    // acquire memory for 2D array - float R[N][M];
    float **R = malloc(N * sizeof * R);
    for (size_t i = 0; i < N; i++) {
	  R[i] = malloc(M * sizeof * R[i]);
	}

    // initialize the matrix R
    for (int i = 0; i < N; i++) {
    	for (int j = 0; j < M; j++) {
    		R[i][j] = 0;    	}
    }
    // non-zero elements
    while (fgets(str, MAXCHAR, fp) != NULL) {
		pch = strtok (str," ,.-");
    	int u = atoi(str);
    	pch = strtok (NULL," ,.-");
    	int i = atoi(pch);
    	pch = strtok (NULL," ,.-");
    	int v = atoi(pch);
    	R[u][i] = v;
    }
	fclose(fp);

	float **P = malloc(N * sizeof * P);
	float **Q = malloc(M * sizeof * Q);
	float **Q_Result = malloc(M * sizeof *Q_Result);
	// acquire memory for 2D array - P, Q and Q_Result;
	for (size_t i = 0; i < N; i++) {
	  P[i] = malloc(K * sizeof * P[i]);
	}
	for (size_t i = 0; i < M; i++) {
	  Q_Result[i] = malloc(K * sizeof * Q_Result[i]);
	  Q[i] = malloc(K * sizeof * Q[i] );
	}
	
	double time_spent = 0.0;
	
	// random value assign to P and Q
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
	clock_t begin = clock();
	// matrix factorization
	matrix_factorization(R, P, Q, K, iterations, a, 0.02, Q_Result);
	clock_t end = clock();

    
	time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 	printf("Users: %-5dItems: %-5dFeatures: %-5dIterations: %-8dalpha: %-5lf\n", N, M, K, iterations, a);
    printf("Serial: Time elpased is %lf seconds\n", time_spent);
	// float result[N][M];
	// float Q_T[K][M];
	// transpose(Q_Result, Q_T);
	// dot2(P, Q_T, result);

    // printf("Predict R:\n");
	// for (int i = 0; i < N; i++){
	// 	for (int j = 0; j < M; j++){
	// 		printf("%.5lf  ", result[i][j]);
	// 	}
	// 	printf("\n");
	// }
	return 0;
}