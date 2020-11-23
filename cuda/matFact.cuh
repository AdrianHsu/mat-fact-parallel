#ifndef SERIAL_MATFACT_H
#define SERIAL_MATFACT_H

#include <iostream>
#include <fstream>
#include <sstream>

__global__ void matFact(std::string inputFileName);

#endif //SERIAL_MATFACT_H