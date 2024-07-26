#include <immintrin.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda.h>

#define SIZE 512
#define RADIUS 8

static void HandleError( cudaError_t err ) {
    if (err != cudaSuccess) {
        printf( "Cuda Error: %s\n", cudaGetErrorString( err ));
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

__global__ void point_multiply_gpu(float *a,float *b,float *output, int Nx, int Ny){
    int iy = blockIdx.x;
    int ix = threadIdx.x;
    int idx = iy*Nx +ix;  
    __syncthreads();

    if (idx<Nx*Ny){
        output[idx] = a[idx]*b[idx];
    }
}

__global__ void point_minus_and_multiply_gpu(float *a,float *b,float* c, float *output, int Nx, int Ny){
    int iy = blockIdx.x;
    int ix = threadIdx.x;
    int idx = iy*Nx +ix;  
    __syncthreads();

    if (idx<Nx*Ny){
        output[idx] = a[idx]-(b[idx]*c[idx]);
    }
}

__global__ void calc_a_gpu(float *a,float *b,float eps, float *output, int Nx, int Ny){
    int iy = blockIdx.x;
    int ix = threadIdx.x;
    int idx = iy*Nx +ix;  
    __syncthreads();

    if (idx<Nx*Ny){
        output[idx] = a[idx]/(b[idx]+eps);
    }
}

__global__ void calc_output_gpu(float *a, float *b, float *c, int *output, int Nx, int Ny){
    int iy = blockIdx.x;
    int ix = threadIdx.x;
    int idx = iy*Nx +ix;  
    __syncthreads();

    if (idx<Nx*Ny){
        int val = (a[idx]*b[idx]+c[idx])*255;
        if(val<0)val = 0;
        output[idx] = val;
    }
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
    int Nblocks = Ny;
    int Nthreads = Nx;
    float eps = 0.0004;
    float *image = (float*)malloc(Nx*Ny*sizeof(float));
    float *guide = (float*)malloc(Nx*Ny*sizeof(float));
    int *output = (int*)malloc(Nx*Ny*sizeof(int));
    float *d_i;
    float *d_g;
    float *d_gg;
    float *d_ig;
    float *d_i_mean;
    float *d_g_mean;
    float *d_g_corr;
    float *d_ig_corr;
    float *d_g_var;
    float *d_ig_cov;
    float *d_a;
    float *d_b;
    float *d_a_mean;
    float *d_b_mean;
    int *d_output;

    HandleError(cudaMalloc( &d_i         ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_g         ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_gg        ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_ig        ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_i_mean    ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_g_mean    ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_g_corr    ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_ig_corr   ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_g_var     ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_ig_cov    ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_a         ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_b         ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_a_mean    ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_b_mean    ,Nx*Ny*sizeof(float)));
    HandleError(cudaMalloc( &d_output    ,Nx*Ny*sizeof(float)));

    struct timeval  tva, tvb;
    gettimeofday(&tva, NULL);

    for(int channel = 0; channel < 3 ; channel++){

        FILE *i_file;
        FILE *g_file;
        FILE *out_file;
        if(channel == 0){
            i_file = fopen("imagedata/cave-noflash_red.txt", "r");
            g_file = fopen("imagedata/cave-flash_red.txt", "r");
            out_file = fopen("imagedata/red_output.txt", "w");
        }
        else if(channel == 1){
            i_file = fopen("imagedata/cave-noflash_green.txt", "r");
            g_file = fopen("imagedata/cave-flash_green.txt", "r");
            out_file = fopen("imagedata/green_output.txt", "w");
        }
        else if(channel == 2){
            i_file = fopen("imagedata/cave-noflash_blue.txt", "r");
            g_file = fopen("imagedata/cave-flash_blue.txt", "r");
            out_file = fopen("imagedata/blue_output.txt", "w");
        }

        int int_1;
        int int_2;
        for(int y=0;y<Ny;y++){
            for(int x=0;x<Nx;x++){
                fscanf(i_file, "%d", &int_1);
                fscanf(g_file, "%d", &int_2);
                image[y*Nx+x] = (float)int_1/255.0;
                guide[y*Nx+x] = (float)int_2/255.0;
            }
        }
        fclose(i_file);
        fclose(g_file);

        struct timeval  tv1, tv2, tv3, tv4, tv5;
        gettimeofday(&tv1, NULL);

        HandleError(cudaMemcpy(d_i, image, Nx*Ny*sizeof(float),cudaMemcpyHostToDevice));
        HandleError(cudaMemcpy(d_g, guide, Nx*Ny*sizeof(float),cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();

        gettimeofday(&tv2, NULL);

        point_multiply_gpu<<<Nblocks, Nthreads>>>(d_i, d_g, d_ig, Nx, Ny);
        point_multiply_gpu<<<Nblocks, Nthreads>>>(d_g, d_g, d_gg, Nx, Ny);

        cudaDeviceSynchronize();

        fmeanr_gpu<<<Nblocks, Nthreads>>>(d_i, d_i_mean, Nx, Ny, r);
        fmeanr_gpu<<<Nblocks, Nthreads>>>(d_g, d_g_mean, Nx, Ny, r);
        fmeanr_gpu<<<Nblocks, Nthreads>>>(d_gg, d_g_corr, Nx, Ny, r);
        fmeanr_gpu<<<Nblocks, Nthreads>>>(d_ig, d_ig_corr, Nx, Ny, r);

        cudaDeviceSynchronize();

        point_minus_and_multiply_gpu<<<Nblocks, Nthreads>>>(d_g_corr, d_g_mean, d_g_mean, d_g_var, Nx, Ny);
        point_minus_and_multiply_gpu<<<Nblocks, Nthreads>>>(d_ig_corr, d_i_mean, d_g_mean, d_ig_cov, Nx, Ny);

        cudaDeviceSynchronize();

        calc_a_gpu<<<Nblocks, Nthreads>>>(d_ig_cov, d_g_var, eps, d_a, Nx, Ny);

        cudaDeviceSynchronize();

        point_minus_and_multiply_gpu<<<Nblocks, Nthreads>>>(d_i_mean, d_a, d_g_mean, d_b, Nx, Ny);

        cudaDeviceSynchronize();

        fmeanr_gpu<<<Nblocks, Nthreads>>>(d_a, d_a_mean, Nx, Ny, r);
        fmeanr_gpu<<<Nblocks, Nthreads>>>(d_b, d_b_mean, Nx, Ny, r);

        cudaDeviceSynchronize();

        calc_output_gpu<<<Nblocks, Nthreads>>>(d_a_mean, d_g, d_b_mean, d_output, Nx, Ny);

        cudaDeviceSynchronize();

        gettimeofday(&tv3, NULL);

        cudaMemcpy(output, d_output, Nx*Ny*sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        gettimeofday(&tv4, NULL);

        for(int y=0;y<Ny;y++){
            for(int x=0;x<Nx;x++){
                fprintf(out_file, "%d ", output[y*Nx+x]);
            }
        }

        fclose(out_file);

        gettimeofday(&tv5, NULL);
    
        printf ("One channel copy to device = \t%f ms\n",
            (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
            (double) (tv2.tv_sec - tv1.tv_sec) * 1000);
        printf ("One channel kernel time = \t%f ms\n",
            (double) (tv3.tv_usec - tv2.tv_usec) / 1000 +
            (double) (tv3.tv_sec - tv2.tv_sec) * 1000);
        printf ("One channel copy to host = \t%f ms\n",
            (double) (tv4.tv_usec - tv3.tv_usec) / 1000 +
            (double) (tv4.tv_sec - tv3.tv_sec) * 1000);
        printf ("One channel write time = \t%f ms\n",
            (double) (tv5.tv_usec - tv4.tv_usec) / 1000 +
            (double) (tv5.tv_sec - tv4.tv_sec) * 1000);
        printf ("One channel total time = \t%f ms\n\n",
            (double) (tv5.tv_usec - tv1.tv_usec) / 1000 +
            (double) (tv5.tv_sec - tv1.tv_sec) * 1000);


    }

    gettimeofday(&tvb, NULL);

    printf ("Total total time = %f ms\n",
        (double) (tvb.tv_usec - tva.tv_usec) / 1000 +
        (double) (tvb.tv_sec - tva.tv_sec) * 1000);

    free(image);
    free(guide);
    free(output);

    cudaFree(d_i      );
    cudaFree(d_g      );
    cudaFree(d_gg     );
    cudaFree(d_ig     );
    cudaFree(d_i_mean );
    cudaFree(d_g_mean );
    cudaFree(d_g_corr );
    cudaFree(d_ig_corr);
    cudaFree(d_g_var  );
    cudaFree(d_ig_cov );
    cudaFree(d_a      );
    cudaFree(d_b      );
    cudaFree(d_a_mean );
    cudaFree(d_b_mean );
    cudaFree(d_output );

    return 0;
}
