#include <immintrin.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda.h>

static void HandleError( cudaError_t err ) {
    if (err != cudaSuccess) {
        printf( "Cuda Error: %s\n", cudaGetErrorString( err ));
        exit( EXIT_FAILURE );
    }
}

void fill_image_data(float *image, int Nx, int Ny){
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
           image[y*Nx+x] = (float)((float)y*Nx)+(float)x;
        }
    }
}

void print_image_data(float *image, int Nx, int Ny){
    for(int y=0;y<Ny;y++){
        for(int x=0;x<Nx;x++){
            printf("%f\t\t",image[y*Nx+x]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void fmeanr_gpu(float *image,float *output, int Nx, int Ny, int block_size, int thread_blocks, int r)
{
    int ix = blockIdx.x*block_size + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float accumulator[128];

    if (ix<Nx){

        int window_h = 1+r;
        int window_w = 1+r;

        if(ix<r){
            window_w += ix;
        }
        else if(ix+r >= Nx){
            window_w += Nx-1-ix;
        }
        else{
            window_w += r;
        }

        accumulator[tid] = image[ix];

        for(int y = 1; y <= r; y++){
            accumulator[tid] += image[y*Nx+ix];
        }

        __syncthreads();

        float sum = 0.0;
        for(int x = tid-r<0 ? 0 : tid-r; x <= tid+r && x < Nx ; x++){
            sum += accumulator[x];
        }
        printf("sum = %f\n",sum);
        output[ix] = sum;///(window_h*window_w);
        __syncthreads();

        for(int y = 1; y<r; y++){
            window_h++;
            accumulator[ix] += image[(y+r)*Nx+ix];
            float sum = 0.0;
            for(int x = ix-r<0 ? 0 : ix-r; x <= ix+r && x < Nx ; x++){
                sum += accumulator[x];
            }
            output[y*Nx+ix] = sum;///(window_h*window_w);
        }
        window_h++;
        for(int y = r; y < Ny - r; y++){
            accumulator[ix] += image[(y+r)*Nx+ix];
            accumulator[ix] -= image[(y-r-1)*Nx+ix];
            float sum = 0.0;
            for(int x = ix-r<0 ? 0 : ix-r; x <= ix+r && x < Nx ; x++){
                sum += accumulator[x];
            }
            output[y*Nx+ix] = sum;///(window_h*window_w);
        }

        for(int y = Ny - r; y<Ny; y++){
            window_h--;
            accumulator[ix] -= image[(y-r-1)*Nx+ix];
            float sum = 0.0;
            for(int x = ix-r<0 ? 0 : ix-r; x <= ix+r && x < Nx ; x++){
                sum += accumulator[x];
            }
            output[y*Nx+ix] = sum;///(window_h*window_w);
        }
    }
}

int main() {
    cudaSetDevice(0);
    int r = 10;
    int Nx = 128;
    int Ny = 128;
    int block_size = 128;
    int thread_blocks = (Nx+block_size-1)/block_size;
    float *image = (float*)malloc(Nx*Ny*sizeof(float));
    float *output = (float*)malloc(Nx*Ny*sizeof(float));
    float *d_image, *d_output;
    HandleError(cudaMalloc(&d_image,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc(&d_output,Nx*Ny*sizeof(float)));

    fill_image_data(image, Nx, Ny);

    struct timeval  tv1, tv2, tv3, tv4;
    gettimeofday(&tv1, NULL);

    HandleError(cudaMemcpy(d_image, image, Nx*Ny*sizeof(float),cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    gettimeofday(&tv2, NULL);

    fmeanr_gpu<<<thread_blocks, block_size>>>(d_image, d_output, Nx, Ny, block_size, thread_blocks, r);
    cudaDeviceSynchronize();

    gettimeofday(&tv3, NULL);

    cudaMemcpy(output, d_output, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    gettimeofday(&tv4, NULL);
    
    printf ("Copy to device = %f ms\n",
        (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
        (double) (tv2.tv_sec - tv1.tv_sec) * 1000);
    printf ("Kernel time = %f ms\n",
        (double) (tv3.tv_usec - tv2.tv_usec) / 1000 +
        (double) (tv3.tv_sec - tv2.tv_sec) * 1000);
    printf ("Copy to host = %f ms\n",
        (double) (tv4.tv_usec - tv3.tv_usec) / 1000 +
        (double) (tv4.tv_sec - tv3.tv_sec) * 1000);
    printf ("Total = %f ms\n",
        (double) (tv4.tv_usec - tv1.tv_usec) / 1000 +
        (double) (tv4.tv_sec - tv1.tv_sec) * 1000);

    print_image_data(output, Nx, Ny);

    free(image);
    free(output);

    cudaFree(d_image);
    cudaFree(d_output);

    return 0;
}
