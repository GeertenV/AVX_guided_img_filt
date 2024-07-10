# AVX_guided_img_filt
compiled with
gcc -mavx -o fmeanr.exe fmeanr.c

and with this for openmp
gcc -fopenmp -mavx -o fmeanr.exe fmeanr.c

memusage -T ./fmeanr.exe