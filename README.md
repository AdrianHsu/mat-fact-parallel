# matrix-factorization-parallel

## TODO

- [x] Serial
- [x] OpenMP
- [ ] MPI
- [ ] CUDA

## Required Packages

* CMake
* [GoogleTest](https://github.com/google/googletest/blob/master/googletest/README.md)
* OpenMP (already installed on OS X)

## How-to

### Serial
```sh
cd /serial/build
cmake ..
make
./src/serial ../../instances/inst0.in
```

### OpenMP
```sh
cd /omp/build
cmake ..
make
./src/omp ../../instances/inst0.in
```

## References

[pedrorio/parallel_and_distributed_computing](https://github.com/pedrorio/parallel_and_distributed_computing)
