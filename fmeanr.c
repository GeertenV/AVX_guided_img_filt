#include <immintrin.h>
#include <stdio.h>

#define SIZE 512
#define RADIUS 8
#define VLEN 8

void fill_image_data(float (*image)[SIZE]){
    for(int y=0;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
           image[y][x] = (float)((float)y*SIZE)+(float)x;
        }
    }
}

void print_image_data(float (*image)[SIZE]){
    for(int y=0;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
            printf("%f\t",image[y][x]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_m256(__m256 vector){
    float* f = (float*)&vector;
    printf("%f %f %f %f %f %f %f %f\n",
    f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
}

__m256i select_mask(int length){
    __m256i mask1 = _mm256_setr_epi32(-1, -0, -0, -0, -0, -0, -0, -0);
    __m256i mask2 = _mm256_setr_epi32(-1, -1, -0, -0, -0, -0, -0, -0);
    __m256i mask3 = _mm256_setr_epi32(-1, -1, -1, -0, -0, -0, -0, -0);
    __m256i mask4 = _mm256_setr_epi32(-1, -1, -1, -1, -0, -0, -0, -0);
    __m256i mask5 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -0, -0, -0);
    __m256i mask6 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -0, -0);
    __m256i mask7 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -0);
    __m256i mask8 = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);

    switch(length) {
    case 1:
        return mask1;
    case 2:
        return mask2;
    case 3:
        return mask3;
    case 4:
        return mask4;
    case 5:
        return mask5;
    case 6:
        return mask6;
    case 7:
        return mask7;
    case 8:
        return mask8;
    }
}

float scalar_window_sum(float (*image)[SIZE],int height, int width,int y_start, int x_start){
    float sum = 0.0;
    for(int j = 0; j < height; j++){
        for(int i = 0; i < width; i++){
            sum += image[y_start+j][x_start+i];
        }
    }
    return sum;
}

float vectorized_window_sum(float (*image)[SIZE],int height, int width,int y_start, int x_start){
    float sum = 0.0;
    for(int i = width; i > 0; i -= VLEN){
        __m256i mask = select_mask(i);
        __m256 accumulator = _mm256_maskload_ps(&image[y_start][x_start],mask);
        for(int j = 1; j < height; j++){
            __m256 vector = _mm256_maskload_ps(&image[y_start+j][x_start],mask);
            accumulator = _mm256_add_ps(vector, accumulator);
        }
        accumulator = _mm256_hadd_ps(accumulator, accumulator);
        accumulator = _mm256_hadd_ps(accumulator, accumulator);

        float* f = (float*)&accumulator;
        sum += f[3] + f[4];
        x_start += VLEN;
    }
    return sum;
}

int main() {
    int r = RADIUS;
    float (*image)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));
    float (*output)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));

    fill_image_data(image);

    //print_image_data(image);

    for(int y=0; y<SIZE; y++){
        for(int x=0; x<SIZE; x++){
            int h = 1+r;
            int w = 1+r;
            int y_start = y-r;
            int x_start = x-r;

            if(y<r){
                h += y;
                y_start = 0;
            }
            else if(y+r >= SIZE){
                h += SIZE-1-y;
            }
            else{
                h += r;
            }

            if(x<r){
                w += x;
                x_start = 0;
            }
            else if(x+r >= SIZE){
                w += SIZE-1-x;
            }
            else{
                w += r;
            }

            //float sum = scalar_window_sum(image,h,w,y_start,x_start);

            float sum = vectorized_window_sum(image,h,w,y_start,x_start);
            float avg = sum/(h*w);
            output[y][x] = avg;
        }
    }
    print_image_data(output);

    return 0;
}
