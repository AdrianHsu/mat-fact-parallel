#ifndef CUDA_UPDATELR_H
#define CUDA_UPDATELR_H

__global__ void updateLR(double *A,
              double *prediction, double *delta,
              int *nonZeroUserIndexes,
              int *nonZeroItemIndexes,
              double *L, double *R,
              int *numberOfUsers, int *numberOfItems, int *numberOfFeatures,
              int *numberOfNonZeroElements,
              double *convergenceCoefficient);

#endif //CUDA_UPDATELR_H
