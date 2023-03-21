cuda_add_executable( deviceQuery  00-deviceQuery.cpp )

#CRS
cuda_add_executable( 01-CRS-spv_1thread  01-CRS-spv_1thread.cu mmio.h mmio.c matrix_utils.h matrix_utils.c)
cuda_add_executable( 02-CRS-spv_vector  02-CRS-spv_vector.cu mmio.h mmio.c matrix_utils.h matrix_utils.c)
cuda_add_executable( 03-CSR-spv_2d_block 03-CSR-spv_2d_block.cu mmio.h mmio.c matrix_utils.h matrix_utils.c)

#ELL
cuda_add_executable( 01-ELL-spv_1thread  01-ELL-spv_1thread.cu mmio.h mmio.c matrix_utils.h matrix_utils.c)
cuda_add_executable( 01-ELL-spv_1thread_co  01-ELL-spv_1thread_co.cu mmio.h mmio.c matrix_utils.h matrix_utils.c)
cuda_add_executable( 02-ELL-spv_vector 02-ELL-spv_vector.cu mmio.h mmio.c matrix_utils.h matrix_utils.c)
cuda_add_executable( 03-ELL-spv_2d_block 03-ELL-spv_2d_block.cu mmio.h mmio.c matrix_utils.h matrix_utils.c)

