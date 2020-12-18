# matrix-factorization-parallel

## TODO

- [x] Serial
- [x] OpenMP
- [x] CUDA

## Required Packages

* CMake
* [GoogleTest](https://github.com/google/googletest/blob/master/googletest/README.md)
* OpenMP (already installed on OS X)

## How-to

### Cuda
```sh
cd /cuda/build
cmake ..
make
./src/cuda ../../instances/inst0.in
```

## References

[pedrorio/parallel_and_distributed_computing](https://github.com/pedrorio/parallel_and_distributed_computing)
