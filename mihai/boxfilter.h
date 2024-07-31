

void boxfilter1D(const float *x_in, float *x_out, size_t r, size_t n, size_t ld);

void boxfilter1D_norm(const float *x_in, float *x_out, size_t r, size_t n, size_t ld, const float * a_norm, const float * b_norm);

void transpose_8x8(float * a, float * b, size_t n);

void transpose(float * in, float * out, size_t n, size_t ld);


int boxfilter(const float *x_in, float *x_out, size_t r, size_t n, size_t ld, float * buf);


void matmul(const float *x1, const float *x2, float *y, size_t n, size_t ld);


void diffmatmul(const float *x1, const float *x2, const float * x3, float *y, size_t n, size_t ld);

void addmatmul(const float *x1, const float *x2, const float * x3, float *y, size_t n, size_t ld);

void matdivconst(const float *x1, const float *x2, float *y, size_t n, size_t ld, float e);

int guidedfilter(const float *I, const float *p, float *q, size_t r, size_t n, size_t ld, float eps);
