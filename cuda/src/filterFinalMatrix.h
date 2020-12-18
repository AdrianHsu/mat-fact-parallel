#ifndef CUDA_FILTERFINALMATRIX_H
#define CUDA_FILTERFINALMATRIX_H

void filterFinalMatrix(double *&A, double *&B,
                       int *&nonZeroUserIndexes,
                       int *&nonZeroItemIndexes,
                       double *&nonZeroElements,
                       double *&L,
                       double *&R,
                       int &numberOfUsers, int &numberOfItems, int &numberOfFeatures,
                       int &numberOfNonZeroElements,
                       int *&BV);


#endif //CUDA_FILTERFINALMATRIX_H
