#include <stdio.h>
#include <immintrin.h>
#include <sys/time.h>

#define SIZE 4096
#define RADIUS 3
#define VLEN 8

void fill_image_data(float (*image)[SIZE]){
    for(int y=0;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
           image[y][x] = (float)((float)y*SIZE)+(float)x;
        }
    }
}

void normalize(float (*sum)[SIZE]){
    for(int y=0;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
            int h = 1+RADIUS;
            int w = 1+RADIUS;
            if(y<RADIUS) h+=y;
            else if(y+RADIUS>=SIZE) h+=SIZE-1-y;
            else h+=RADIUS;
            if(x<RADIUS) w+=x;
            else if(x+RADIUS>=SIZE) w+=SIZE-1-x;
            else w+=RADIUS;
            sum[y][x] = (float)sum[y][x]/(h*w);
        }
    }

}

void print_image_data(float (*image)[SIZE]){
    for(int y=0;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
            printf("%f\t\t",image[y][x]);
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

void setup_col(float (*image)[SIZE], float (*temp)[SIZE]){
    for(int y=0;y<SIZE;y++){
        float sum = 0.0;
        for(int x=0;x<=RADIUS;x++){
            sum += image[y][x];
        }
        temp[y][0] = sum;
    }
}

void moving_sum_hor(float (*image)[SIZE], float (*temp)[SIZE]){
    for(int y=0;y<SIZE;y++){
        for(int x=1;x<SIZE;x++){
            float sum = temp[y][x-1];
            if(RADIUS+x < SIZE){
                sum += image[y][x+RADIUS];
            }
            if(x>RADIUS){
                sum -= image[y][x-RADIUS-1];
            }
            temp[y][x] = sum;
        }
    }
}

void setup_row(float (*temp)[SIZE], float (*output)[SIZE]){
    for(int x=0;x<SIZE;x++){
        float sum = 0.0;
        for(int y=0;y<=RADIUS;y++){
            sum += temp[y][x];
        }
        output[0][x] = sum;
    }
}

void moving_sum_ver(float (*temp)[SIZE], float (*output)[SIZE]){
    int x=0;
    for(int y=1;y<SIZE;y++){
        for(x;x<=SIZE-VLEN;x+=VLEN){
            __m256 sum = _mm256_loadu_ps(&output[y-1][x]);
            if(RADIUS+y < SIZE){
                __m256 vector = _mm256_loadu_ps(&temp[y+RADIUS][x]);
                sum = _mm256_add_ps(vector, sum);
            }
            if(y>RADIUS){
                __m256 vector = _mm256_loadu_ps(&temp[y-RADIUS-1][x]);
                sum = _mm256_sub_ps(sum, vector);
            }
            _mm256_storeu_ps(&output[y][x],sum);
        }
    }

    // for(x;x<=SIZE-VLEN;x+=VLEN){
    //     __m256 sum = _mm256_loadu_ps(&output[0][x]);
    //     for(int y=1;y<SIZE;y++){
    //         if(RADIUS+y < SIZE){
    //             __m256 vector = _mm256_loadu_ps(&temp[y+RADIUS][x]);
    //             sum = _mm256_add_ps(vector, sum);
    //         }
    //         if(y>RADIUS){
    //             __m256 vector = _mm256_loadu_ps(&temp[y-RADIUS-1][x]);
    //             sum = _mm256_sub_ps(sum, vector);
    //         }
    //         _mm256_storeu_ps(&output[y][x],sum);
    //     }
    // }

    if(SIZE%VLEN>0){
        __m256i mask = select_mask(SIZE%VLEN);
        for(int y=1;y<SIZE;y++){
            __m256 sum = _mm256_maskload_ps(&output[y-1][x],mask);
            if(RADIUS+y < SIZE){
                __m256 vector = _mm256_maskload_ps(&temp[y+RADIUS][x],mask);
                sum = _mm256_add_ps(vector, sum);
            }
            if(y>RADIUS){
                __m256 vector = _mm256_maskload_ps(&temp[y-RADIUS-1][x],mask);
                sum = _mm256_sub_ps(sum, vector);
            }
            _mm256_maskstore_ps(&output[y][x],mask,sum);
        }
    }
}

int main() {
    int r = RADIUS;
    float (*image)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));
    float (*temp)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));
    float (*output)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));

    fill_image_data(image);

    struct timeval  tv1, tv2;

    gettimeofday(&tv1, NULL);
        setup_col(image,temp);
        moving_sum_hor(image,temp);     
        setup_row(temp,output);
        moving_sum_ver(temp,output);
        normalize(output);
    gettimeofday(&tv2, NULL);
    printf ("Total time = %f ms\n",
            (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
            (double) (tv2.tv_sec - tv1.tv_sec) * 1000);

    //print_image_data(output);

    return 0;
}
