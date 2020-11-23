#include "updateLR.h"

__global__ void updateLR(double *&A,
              double *&prediction, double *&delta,
              int *&nonZeroUserIndexes,
              int *&nonZeroItemIndexes,
              double *&L, double *&R,
              int &numberOfUsers, int &numberOfItems, int &numberOfFeatures,
              int &numberOfNonZeroElements,
              double &convergenceCoefficient) {

    int i, l, k;

    // I don't see the need to do this.
    // Copy L -> StoreL & R -> StoreR
    // for (int i = 0; i < numberOfFeatures; i++) {
    //     for (int k = 0; k < numberOfUsers; k++) {
    //         StoreL[k * numberOfFeatures + i] = L[k * numberOfFeatures + i];
    //     }
    //     for (int k = 0; k < numberOfItems; k++) {
    //         StoreR[i * numberOfItems + k] = R[i * numberOfItems + k];
    //     }
    // }

    // Transfromed into code below. Code kept for reference.
    // Compute difference between prediction and real value
    // for (int l = 0; l < numberOfNonZeroElements; l++) {
    //     prediction[l] = 0;
    //     delta[l] = 0;
    //     // L * R
    //     for (int k = 0; k < numberOfFeatures; k++) {
    //         prediction[l] += L[nonZeroUserIndexes[l] * numberOfFeatures + k] * R[k * numberOfItems + nonZeroItemIndexes[l]];
    //     }
    //     delta[l] = A[nonZeroUserIndexes[l] * numberOfItems + nonZeroItemIndexes[l]] - prediction[l];
    // }

    l = blockIdx.x;
    k = threadIdx.x;
    // Init by the first thread
    if (k == 0) {
        prediction[l] = 0;
        delta[l] = 0;
    }
    
    // Synchronize (ensure all the data is available) 
    __syncthreads();

    // Inner for loop
    double predict = L[nonZeroUserIndexes[l] * numberOfFeatures + k] * R[k * numberOfItems + nonZeroItemIndexes[l]];
    atomicAdd(&prediction[l], predict);

    __syncthreads();

    if (k==0) {
        delta[l] = A[nonZeroUserIndexes[l] * numberOfItems + nonZeroItemIndexes[l]] - prediction[l];
    }

    __syncthreads();


    // Transfromed into code below. Code kept for reference.
    // for (int l = 0; l < numberOfNonZeroElements; l++) {
    //     for (int k = 0; k < numberOfFeatures; k++) {
    //         L[nonZeroUserIndexes[l] * numberOfFeatures + k] += convergenceCoefficient * (2 * delta[l] * StoreR[k * numberOfItems + nonZeroItemIndexes[l]]);
    //         R[k * numberOfItems + nonZeroItemIndexes[l]] += convergenceCoefficient * (2 * delta[l] * StoreL[nonZeroUserIndexes[l] * numberOfFeatures + k]);
    //     }
    // }
    
    L[nonZeroUserIndexes[l] * numberOfFeatures + k] += convergenceCoefficient * (2 * delta[l] * StoreR[k * numberOfItems + nonZeroItemIndexes[l]]);
    R[k * numberOfItems + nonZeroItemIndexes[l]] += convergenceCoefficient * (2 * delta[l] * StoreL[nonZeroUserIndexes[l] * numberOfFeatures + k]);
};