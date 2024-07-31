#include<cblas.h>



void transpose_blas(float * in, float * out, size_t n, size_t ld)
{

  cblas_somatcopy(CblasRowMajor, CblasTrans, n, n, 1.0f, in, ld, out, ld);

}
