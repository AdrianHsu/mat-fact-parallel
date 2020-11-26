#include <omp.h>
// #include "/usr/local/opt/libomp/include/omp.h"
#include "readInput.h"
#include "initialLR.h"
#include "updateLR.cuh"
#include "filterFinalMatrix.h"
#include "verifyResult.h"
#include "printMatrix.h"

void matFact(std::string inputFileName) {
    time_t start_time;
    time_t read_input;
    time_t initial_lr;
    time_t total_time;

    start_time = omp_get_wtime();

    double *A = new double[0];
    int *nonZeroUserIndexes = new int[0];
    int *nonZeroItemIndexes = new int[0];
    double *nonZeroElements = new double[0];

    int numberOfIterations, numberOfFeatures, numberOfUsers, numberOfItems, numberOfNonZeroElements;
    double convergenceCoefficient;
    

    int *nonZeroUserIndexes_, *nonZeroItemIndexes_;
    int *numberOfFeatures_, *numberOfUsers_, *numberOfItems_, *numberOfNonZeroElements_;
    double *convergenceCoefficient_; 
    double *A_, *L_, *R_, *prediction_, *delta_;

    readInput(inputFileName, A, nonZeroUserIndexes, nonZeroItemIndexes, nonZeroElements,
              numberOfIterations, numberOfFeatures, convergenceCoefficient,
              numberOfUsers, numberOfItems,
              numberOfNonZeroElements);

//    printMatrix("A", A, numberOfUsers, numberOfItems);

    read_input = omp_get_wtime();

    auto *L = new double[numberOfUsers * numberOfFeatures];
    auto *R = new double[numberOfFeatures * numberOfItems];

    initialLR(L, R, numberOfUsers, numberOfItems, numberOfFeatures);

    initial_lr = omp_get_wtime();

//    printMatrix("L", L, numberOfUsers, numberOfFeatures);
//    printMatrix("R", R, numberOfFeatures, numberOfItems);

    // auto *StoreL = new double[numberOfUsers * numberOfFeatures];
    // auto *StoreR = new double[numberOfFeatures * numberOfItems];
    // auto *prediction = new double[numberOfNonZeroElements];
    // auto *delta = new double[numberOfNonZeroElements];

    // GPU Allocation
    cudaMalloc((void **)&nonZeroUserIndexes_, numberOfNonZeroElements * sizeof(int));
    cudaMalloc((void **)&nonZeroItemIndexes_, numberOfNonZeroElements * sizeof(int));
    cudaMalloc((void **)&numberOfUsers_, sizeof(int));
    cudaMalloc((void **)&numberOfItems_, sizeof(int));
    cudaMalloc((void **)&numberOfFeatures_, sizeof(int));
    cudaMalloc((void **)&numberOfNonZeroElements_, sizeof(int));
    cudaMalloc((void **)&convergenceCoefficient_, sizeof(double));

    cudaMalloc((void **)&A_, numberOfUsers * numberOfItems * sizeof(double));
    cudaMalloc((void **)&L_, numberOfUsers * numberOfFeatures * sizeof(double));
    cudaMalloc((void **)&R_, numberOfFeatures * numberOfItems * sizeof(double));
    cudaMalloc((void **)&prediction_, numberOfNonZeroElements * sizeof(double));
    cudaMalloc((void **)&delta_, numberOfNonZeroElements * sizeof(double));

    // Copy data to GPU
    cudaMemcpy(nonZeroUserIndexes_, nonZeroUserIndexes, numberOfNonZeroElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nonZeroItemIndexes_, nonZeroItemIndexes, numberOfNonZeroElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(numberOfUsers_, &numberOfUsers, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(numberOfItems_, &numberOfItems, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(numberOfFeatures_, &numberOfFeatures, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(numberOfNonZeroElements_, &numberOfNonZeroElements, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(convergenceCoefficient_, &convergenceCoefficient, sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(A_, A, numberOfUsers * numberOfItems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(L_, L, numberOfUsers * numberOfFeatures * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(R_, R, numberOfFeatures * numberOfItems * sizeof(double), cudaMemcpyHostToDevice); 

    for (int iteration = 0; iteration < numberOfIterations; iteration++) {
        // Parallel to numberOfNonZeroElements blocks, and numberOfFeatures threads.
        updateLR<<<numberOfNonZeroElements, numberOfFeatures>>>(A_,
                 prediction_, delta_,
                 nonZeroUserIndexes_,
                 nonZeroItemIndexes_,
                 L_, R_,
                 numberOfUsers_, numberOfItems_, numberOfFeatures_,
                 numberOfNonZeroElements_,
                 convergenceCoefficient_);
    }


    // Copy data back to host.
    cudaMemcpy(L, L_, numberOfUsers * numberOfFeatures * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(R, R_, numberOfFeatures * numberOfItems * sizeof(double), cudaMemcpyDeviceToHost);

    time_t final_filtering = omp_get_wtime();

    delete[] prediction_;
    delete[] delta_;
    //delete[] StoreL;
    //delete[] StoreR;

    auto *B = new double[numberOfUsers * numberOfItems];
    auto *BV = new int[numberOfUsers];

    filterFinalMatrix(A, B, nonZeroUserIndexes,
                      nonZeroItemIndexes,
                      nonZeroElements,
                      L, R,
                      numberOfUsers, numberOfItems, numberOfFeatures,
                      numberOfNonZeroElements,
                      BV);

    delete[] L;
    delete[] R;
    delete[] A;
    delete[] nonZeroUserIndexes;
    delete[] nonZeroItemIndexes;
    delete[] nonZeroElements;
    delete[] B;

    total_time = omp_get_wtime();

    if (std::getenv("LOG_RESULTS")) {
        std::ofstream logResults("../compare/comparison.cuda.csv", std::ios::app);
        logResults << inputFileName << ", ";
        logResults << 1 << ", ";
        std::string outputFileName = inputFileName.substr(0, inputFileName.length() - 2).append("out");
        int numberOfErrors = verifyResult(outputFileName, BV);

        std::cout << numberOfErrors << std::endl;

        logResults << numberOfErrors << ", ";
        logResults << numberOfUsers << ", ";
        logResults << numberOfItems << ", ";
        logResults << numberOfFeatures << ", ";
        logResults << numberOfNonZeroElements << ", ";
        logResults << numberOfIterations << ", ";
        logResults << double(read_input - start_time) << ", ";
        logResults << double(initial_lr - read_input) << ", ";
        logResults << double(final_filtering - initial_lr) << ", ";
        logResults << double(total_time - final_filtering) << ", ";
        logResults << double(total_time - start_time);
        logResults << std::endl;
        logResults.close();
    }

    delete[] BV;
}
