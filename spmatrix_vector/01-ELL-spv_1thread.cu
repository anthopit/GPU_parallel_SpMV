#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

extern "C" {
#include "mmio.h"
#include "matrix_utils.h"
}

#define BD 256

const dim3 BLOCK_DIM(BD);

void CpuMatrixVector(int m, const int* maxnz, const int* ja, const double* as, const double* x, double* y) {
    int i,j;
    for (i = 0; i < m; ++i) {
        double t=0.0;
        for (j = 0; j < *maxnz; ++j) {
            t = t + as[i*(*maxnz)+j]*x[ja[i*(*maxnz)+j]];
        }
        y[i] = t;
    }
}

__global__ void gpuMatrixVectorELL(int rows, const int* maxnz, const int* ja, const double* as,
                                const double* x, double* y) {
    int tr     = threadIdx.x;
    int row    = blockIdx.x*blockDim.x + tr;
    if (row < rows) {
        double t  = 0.0;
        for (int col = 0; col < *maxnz; ++col) {
            t = t + as[col + row * (*maxnz)]*x[ja[col + row * (*maxnz)]];
        }
        y[row] = t;
    }
}

int main(int argc, char** argv) {

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int m, n, nz;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }
    else
    {
        if ((f = fopen(argv[1], "r")) == NULL){
            printf("Could not open file %s", argv[1]);
            exit(1);
        }
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode) && mm_is_integer(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nz)) !=0)
        exit(1);
  
  
  // ----------------------- Host memory initialisation ----------------------- //

    int h_maxnz;
    int **ja;
    double **as;
    double* h_x = new double[n];
    double* h_y = new double[m];
    double* h_y_d = new double[m];

    srand(123456);
    for (int col = 0; col < m; ++col) {
        h_x[col] = 100.0f * static_cast<double>(rand()) / RAND_MAX;
    }

    read_mtx_coo_ellpack(f, m, n, &nz, &h_maxnz, &ja, &as, mm_is_symmetric(matcode), mm_is_pattern(matcode));
//   print_ellpack_mtx_ellpack(m, n, &maxnz, ja, as);
//   print_ellpack_mtx_2D(m, n, &maxnz, ja, as);

    int* h_ja = new int[m*h_maxnz];
    double* h_as = new double[m*h_maxnz];
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < h_maxnz; ++col) {
            h_ja[col + row * h_maxnz] = ja[row][col];
            h_as[col + row * h_maxnz] = as[row][col];
        }
    }

    // ---------------------- Device memory initialisation ---------------------- //

    int *d_maxnz, *d_ja;
    double *d_as, *d_x, *d_y;

    checkCudaErrors(cudaMalloc((void**) &d_maxnz, sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_ja, m * h_maxnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_as, m * h_maxnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_x, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_y, m * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_maxnz, &h_maxnz, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ja, h_ja, m * h_maxnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_as, h_as, m * h_maxnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x,  n * sizeof(double), cudaMemcpyHostToDevice));

  // ------------------------ Calculations on the CPU ------------------------- //
    double flopcnt=2.e-6*nz;

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();
    CpuMatrixVector(m, &h_maxnz,h_ja,h_as,h_x,h_y);

    timer->stop();
    double cpuflops=flopcnt/ timer->getTime();
    std::cout << "  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;

// ------------------------ Calculations on the GPU ------------------------- //

    const dim3 GRID_DIM((m - 1 + BLOCK_DIM.x)/ BLOCK_DIM.x  ,1);

    timer->reset();
    timer->start();
    gpuMatrixVectorELL<<<GRID_DIM, BLOCK_DIM >>>(m, d_maxnz, d_ja, d_as, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    double gpuflops=flopcnt/ timer->getTime();
    std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

    checkCudaErrors(cudaMemcpy(h_y_d, d_y, m*sizeof(double),cudaMemcpyDeviceToHost));

    double reldiff = 0.0f;
    double diff = 0.0f;

    for (int row = 0; row < m; ++row) {
        double maxabs = std::max(std::abs(h_y[row]),std::abs(h_y_d[row]));
        if (maxabs == 0.0) maxabs=1.0;
        reldiff = std::max(reldiff, std::abs(h_y[row] - h_y_d[row])/maxabs);
        diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
    }
    std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;
// ------------------------------- Cleaning up ------------------------------ //

    delete timer;

    checkCudaErrors(cudaFree(d_maxnz));
    checkCudaErrors(cudaFree(d_ja));
    checkCudaErrors(cudaFree(d_as));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));


    delete[] h_ja;
    delete[] h_as;
    delete[] h_x;
    delete[] h_y;
    delete[] h_y_d;
    return 0;
}
