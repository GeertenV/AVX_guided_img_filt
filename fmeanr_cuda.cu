#include <immintrin.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda.h>

#define SIZE 1850
#define RADIUS 10
#define VLEN 8
#define VLEN5 40

static void HandleError( cudaError_t err ) {
    if (err != cudaSuccess) {
        printf( "henk: %s\n", cudaGetErrorString( err ));
        exit( EXIT_FAILURE );
    }
}

void fill_image_data(float *image){
    for(int y=0;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
           image[y*SIZE+x] = (float)((float)y*SIZE)+(float)x;
        }
    }
}

void print_image_data(float *image){
    for(int y=0;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
            printf("%f\t\t",image[y*SIZE+x]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void fmeanr_gpu(float *image,float *output, int Nx, int Ny, int r)
{
    //each block is assigned to a row of an image, iy index of y value                  
    int iy = blockIdx.x;  

    //each thread is assigned to a pixel of a row, ix index of x value
    int ix = threadIdx.x; 

    //idx global index (all blocks) of the image pixel 
    int idx = iy*Nx +ix;                        

    int window_h = 1+r;
    int window_w = 1+r;
    int y_start = iy-r;
    int x_start = ix-r;

    if(iy<r){
        window_h += iy;
        y_start = 0;
    }
    else if(iy+r >= SIZE){
        window_h += SIZE-1-iy;
    }
    else{
        window_h += r;
    }

    if(ix<r){
        window_w += ix;
        x_start = 0;
    }
    else if(ix+r >= SIZE){
        window_w += SIZE-1-ix;
    }
    else{
        window_w += r;
    }		

    int ii, jj;
    float sum = 0.0;

    __syncthreads();			  

    if (idx<Nx*Ny){
        for (int ki = 0; ki<window_h; ki++){
            for (int kj = 0; kj<window_w; kj++){
                ii = kj + x_start;
                jj = ki + y_start;
                sum+=image[jj*Nx+ii];
            }
        }
        output[idx] = sum/(window_h*window_w);
    }
}

int main() {
    cudaSetDevice(0);
    int r = RADIUS;
    int Nx = SIZE;
    int Ny = SIZE;
    float *image = (float*)malloc(Nx*Ny*sizeof(float));
    float *output = (float*)malloc(Nx*Ny*sizeof(float));
    float *d_image, *d_output;
    HandleError(cudaMalloc(&d_image,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc(&d_output,Nx*Ny*sizeof(float)));

    fill_image_data(image);

    HandleError(cudaMemcpy(d_image, image, Nx*Ny*sizeof(float),cudaMemcpyHostToDevice));

    int Nblocks = Ny;
    int Nthreads = Nx;

    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);

    fmeanr_gpu<<<Nblocks, Nthreads>>>(d_image, d_output, Nx, Ny, r);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&tv2, NULL);
    printf ("Total time = %f ms\n",
        (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
        (double) (tv2.tv_sec - tv1.tv_sec) * 1000);

    //print_image_data(output);

    free(image);
    free(output);

    cudaFree(d_image);
    cudaFree(d_output);

    return 0;
}
