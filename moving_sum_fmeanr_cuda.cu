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

__global__ void ver_fmeanr_gpu(float *image,float *temp, int Nx, int Ny, int threadblock_dim, int r)
{
    int ix = blockIdx.x*threadblock_dim + threadIdx.x;

    if (ix<Nx){
        float sum = 0;

        for(int y=0;y<=r;y++){
            sum += image[y*Nx+ix];
        }
        temp[ix] = sum;

        for(int y = 1; y<r; y++){
            sum += image[(y+r)*Nx+ix];
            temp[y*Nx+ix] = sum;
        }

        for(int y = r; y < Ny - r; y++){
            sum += image[(y+r)*Nx+ix];
            sum -= image[(y-r-1)*Nx+ix];
            temp[y*Nx+ix] = sum;
        }

        for(int y = Ny - r; y<Ny; y++){
            sum -= image[(y-r-1)*Nx+ix];
            temp[y*Nx+ix] = sum;
        }
    }
}

__global__ void hor_fmeanr_gpu(float *temp,float *output, int Nx, int Ny, int threadblock_dim, int r)
{
    int iy = blockIdx.x*threadblock_dim + threadIdx.x;

    if (iy<Ny){
        int window_h = 1+r;
        int window_w = 1+r;

        if(iy<r){
            window_h += iy;
        }
        else if(iy+r >= Ny){
            window_h += Ny-1-iy;
        }
        else{
            window_h += r;
        }

        float sum = 0;

        for(int x=0;x<=r;x++){
            sum += temp[iy*Nx+x];
        }
        output[iy*Nx] = sum/(window_h*window_w);

        for(int x = 1; x<=r; x++){
            window_w++;
            sum += temp[iy*Nx+x+r];
            output[iy*Nx+x] = sum/(window_h*window_w);
        }

        for(int x = r+1; x < Nx - r; x++){
            sum += temp[iy*Nx+x+r];
            sum -= temp[iy*Nx+x-r-1];
            output[iy*Nx+x] = sum/(window_h*window_w);
        }

        for(int x = Nx - r; x<Nx; x++){
            window_w--;
            sum -= temp[iy*Nx+x-r-1];
            output[iy*Nx+x] = sum/(window_h*window_w);
        }
    }
}

int main() {
    cudaSetDevice(0);
    int r = 19;
    int Nx = 2268;
    int Ny = 1512;
    int threadblock_dim = 128;
    int x_blocks = (Nx+threadblock_dim-1)/threadblock_dim;
    int y_blocks = (Ny+threadblock_dim-1)/threadblock_dim;
    float *image = (float*)malloc(Nx*Ny*sizeof(float));
    float *output = (float*)malloc(Nx*Ny*sizeof(float));
    float *d_image, *d_output, *d_temp;
    HandleError(cudaMalloc(&d_image,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc(&d_output,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc(&d_temp,Nx*Ny*sizeof(float)));

    fill_image_data(image, Nx, Ny);

    struct timeval  tv1, tv2, tv3, tv4;
    gettimeofday(&tv1, NULL);

    HandleError(cudaMemcpy(d_image, image, Nx*Ny*sizeof(float),cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    gettimeofday(&tv2, NULL);

    ver_fmeanr_gpu<<<x_blocks, threadblock_dim>>>(d_image, d_temp, Nx, Ny, threadblock_dim, r);
    cudaDeviceSynchronize();

    hor_fmeanr_gpu<<<y_blocks, threadblock_dim>>>(d_temp, d_output, Nx, Ny, threadblock_dim, r);
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
