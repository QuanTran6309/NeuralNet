

#include "algebra/tensor.hpp"
#include "algebra/matrix.hpp"
#include <benchmark/benchmark.h>

using namespace IdioticML;

int MatrixOp(Matrix<int> matrix1, Matrix<int> matrix2){
    Matrix<int> matrix = matrix1 * matrix2;
    return 1;
}

// Benchmark wrapper
static void BM_MyFunction(benchmark::State& state) {
    int src1[] = {1, 2, 3, 4, 5, 6};
    int src2[] = {3, 4, 5, 6, 7, 8};

    Matrix<int> matrix1(1000, 1000);    
    Matrix<int> matrix2(1000, 1000);
    for (auto _ : state) {
        auto result = MatrixOp(matrix1, matrix2);
        benchmark::DoNotOptimize(result); // prevent compiler optimization
    }
}

BENCHMARK(BM_MyFunction)->Arg(1)->Arg(100)->Arg(10000);

BENCHMARK_MAIN();

