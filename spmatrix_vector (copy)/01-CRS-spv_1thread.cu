#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

extern "C" {
#include "mmio.h"
#include "matrix_utils.h"
}

// Block dimension
#define BD 256

const dim3 BLOCK_DIM(BD);

void CpuMatrixVector(int m, int n, const int* irp, const int* ja, const double* as, const float* x, float* y) {
  for (int row = 0; row < m; ++row) {
    float t=0.0;
    for (int col = irp[row]; col < irp[row+1]; ++col) {
        t = t + as[col]*x[ja[col]];
    }
    y[row] = t;
  }
}

__global__ void gpuMatrixVectorCSR(int rows, int cols, const int* irp, const int* ja, const double* as,
                                const float* x, float* y) {
    int tr     = threadIdx.x;
    int row    = blockIdx.x*blockDim.x + tr;
    if (row < rows) {
        float t  = 0.0;
        for (int col = irp[row]; col < irp[row+1]; ++col) {
            t = t + as[col]*x[ja[col]];
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

    int *h_irp, *h_ja;
    double *h_as;
    float* h_x = new float[n];
    float* h_y = new float[m];
    float* h_y_d = new float[m];

    srand(123456);
    for (int col = 0; col < m; ++col) {
        h_x[col] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
    }

    read_mtx_coo_csr(f, m, n, &nz, &h_irp, &h_ja, &h_as, mm_is_symmetric(matcode), mm_is_pattern(matcode));
    //print_csr_mtx_csr(m, n, nz, h_irp, h_ja, h_as);
    //print_csr_mtx_2D(m, n, nz, h_irp, h_ja, h_as);

    // ---------------------- Device memory initialisation ---------------------- //
    int *d_irp, *d_ja;
    double *d_as;
    float *d_x, *d_y;

    checkCudaErrors(cudaMalloc((void**) &d_irp, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_ja, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_as, nz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_x, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**) &d_y, m * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_irp, h_irp, (m+1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ja, h_ja, nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_as, h_as, nz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x,  n * sizeof(float), cudaMemcpyHostToDevice));

  // ------------------------ Calculations on the CPU ------------------------- //
    float flopcnt=2.e-6*nz;

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();
    CpuMatrixVector(m, n,h_irp,h_ja,h_as,h_x,h_y);

    timer->stop();
    float cpuflops=flopcnt/ timer->getTime();
    std::cout << "  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;

//// ------------------------ Calculations on the GPU ------------------------- //
    // Set the grid size to have one thread per row of the matrix.
    const dim3 GRID_DIM((m - 1 + BLOCK_DIM.x)/ BLOCK_DIM.x  ,1);

    timer->reset();
    timer->start();
    gpuMatrixVectorCSR<<<GRID_DIM, BLOCK_DIM >>>(m, n, d_irp, d_ja, d_as, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    float gpuflops=flopcnt/ timer->getTime();
    std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;


    checkCudaErrors(cudaMemcpy(h_y_d, d_y, m*sizeof(float),cudaMemcpyDeviceToHost));

    float reldiff = 0.0f;
    float diff = 0.0f;

    for (int row = 0; row < m; ++row) {
        float maxabs = std::max(std::abs(h_y[row]),std::abs(h_y_d[row]));
        if (maxabs == 0.0) maxabs=1.0;
        reldiff = std::max(reldiff, std::abs(h_y[row] - h_y_d[row])/maxabs);
        diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
    }
    std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;

// ------------------------------- Cleaning up ------------------------------ //

    delete timer;

    checkCudaErrors(cudaFree(d_irp));
    checkCudaErrors(cudaFree(d_ja));
    checkCudaErrors(cudaFree(d_as));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    delete[] h_irp;
    delete[] h_ja;
    delete[] h_as;
    delete[] h_x;
    delete[] h_y;
    delete[] h_y_d;
    return 0;
}
