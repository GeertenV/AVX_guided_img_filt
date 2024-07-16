# AVX_guided_img_filt
AVX compilation
gcc -mavx -o fmeanr.exe fmeanr.c

OpenMP + AVX compiltion
gcc -fopenmp -mavx -o fmeanr_multi_vector_moving_sum.exe fmeanr_multi_vector_moving_sum.c

Cuda compilation
nvcc -o fmeanr_cuda.exe fmeanr_cuda.cu
( On QCE Cuda server you also need to do    )
(    module load cuda/11.0                  )
(    module load devtoolset/9               )

memusage -T ./fmeanr.exe