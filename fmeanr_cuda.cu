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

__global__ void fmeanr_gpu(float *image,float *output, int Nx, int Ny, int threadblock_dim, int x_blocks, int y_blocks, int r)
{
    int block_x = blockIdx.x%x_blocks;
    int block_y = blockIdx.x/x_blocks;  

    int ix = block_x*threadblock_dim + threadIdx.x%threadblock_dim;
    int iy = block_y*threadblock_dim + threadIdx.x/threadblock_dim;

    //idx global index
    int idx = iy*Nx +ix;                        

    if (ix<Nx && iy<Ny){

        int window_h = 1+r;
        int window_w = 1+r;
        int y_start = iy-r;
        int x_start = ix-r;

        if(iy<r){
            window_h += iy;
            y_start = 0;
        }
        else if(iy+r >= Ny){
            window_h += Ny-1-iy;
        }
        else{
            window_h += r;
        }

        if(ix<r){
            window_w += ix;
            x_start = 0;
        }
        else if(ix+r >= Nx){
            window_w += Nx-1-ix;
        }
        else{
            window_w += r;
        }		

        int ii, jj;
        float sum = 0.0;		  


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
    int r = 7;
    int Nx = 2268;
    int Ny = 1512;
    int threadblock_dim = 32;
    int x_blocks = (Nx+threadblock_dim-1)/threadblock_dim;
    int y_blocks = (Ny+threadblock_dim-1)/threadblock_dim;
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

    fmeanr_gpu<<<x_blocks*y_blocks, threadblock_dim*threadblock_dim>>>(d_image, d_output, Nx, Ny, threadblock_dim, x_blocks, y_blocks, r);
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

    //print_image_data(output, Nx, Ny);

    free(image);
    free(output);

    cudaFree(d_image);
    cudaFree(d_output);

    return 0;
}
