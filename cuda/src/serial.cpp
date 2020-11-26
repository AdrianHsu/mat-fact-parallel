#include "matFact.cuh"

int main(int argc, char *argv[]) {
    std::string inputFileName = argv[1];
    matFact(inputFileName);
    return 0;
}
