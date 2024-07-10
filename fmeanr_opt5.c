#include <immintrin.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#define SIZE 1856
#define RADIUS 10
#define VLEN 8
#define VLEN5 40

void fill_image_data(float (*image)[SIZE]){
    for(int y=0;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
           image[y][x] = (float)((float)y*SIZE)+(float)x;
        }
    }
}

void normalize(float (*sum)[SIZE]){
    #pragma omp parallel for
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
    printf("%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t\t",
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

void extract_output_sum_left(float (*output)[SIZE], float (*accumulators)[8], int y){
    float moving_sum = 0.0;
    for(int i = 0; i <= RADIUS; i++){
        moving_sum += accumulators[i/VLEN][i%VLEN];
    }
    output[y][0] = moving_sum;

    for(int i = 1; i < VLEN5-RADIUS; i++){
        moving_sum += accumulators[(i+RADIUS)/VLEN][(i+RADIUS)%VLEN];
        if(i > RADIUS){
            moving_sum -= accumulators[(i-RADIUS-1)/VLEN][(i-RADIUS-1)%VLEN];
        }
        //printf("y = %d, x = %d, sum = %f\n",y,i+x_start,moving_sum);
        output[y][i] = moving_sum;
    }
}

void extract_output_sum_middle(float (*output)[SIZE], float (*accumulators)[8], int x_start, int y){
    float moving_sum = 0.0;
    for(int i = 0; i <= RADIUS; i++){
        moving_sum += accumulators[i/VLEN][i%VLEN];
    }

    for(int i = 1; i < VLEN5-RADIUS; i++){
        moving_sum += accumulators[(i+RADIUS)/VLEN][(i+RADIUS)%VLEN];
        if(i > RADIUS){
            moving_sum -= accumulators[(i-RADIUS-1)/VLEN][(i-RADIUS-1)%VLEN];
        }
        //printf("y = %d, x = %d, sum = %f\n",y,i+x_start,moving_sum);
        if(i >= RADIUS) output[y][i+x_start] = moving_sum;
    }
}

void extract_output_sum_right(float (*output)[SIZE], float (*accumulators)[8], int x_start, int y){
    int width = SIZE - x_start;
    width = width < VLEN5 ? width:VLEN5;

    float moving_sum = 0.0;
    for(int i = 0; i <= RADIUS && i < width; i++){
        moving_sum += accumulators[i/VLEN][i%VLEN];
    }

    for(int i = 1; i < width ; i++){
        if(i+RADIUS < width ){
            moving_sum += accumulators[(i+RADIUS)/VLEN][(i+RADIUS)%VLEN];
        }
        if(i > RADIUS){
            moving_sum -= accumulators[(i-RADIUS-1)/VLEN][(i-RADIUS-1)%VLEN];
        }
        //printf("y = %d, x = %d, sum = %f\n",y,i+x_start,moving_sum);
        if(i >= RADIUS) output[y][i+x_start] = moving_sum;
    }
}

void column_sum_masked(float (*image)[SIZE],float (*output)[SIZE], int x_start){
    int width = SIZE - x_start;
    int complete_vectors = width/VLEN;

    __m256i mask = select_mask(width%VLEN);
    __m256 accumulators[5];

    for(int i = 0; i < complete_vectors; i++){
        __m256 accumulator = _mm256_loadu_ps(&image[0][x_start+(i*VLEN)]);
        accumulators[i] = accumulator;

    }
    accumulators[complete_vectors] = _mm256_maskload_ps(&image[0][x_start+(complete_vectors*VLEN)],mask);


    for(int i = 1; i <= RADIUS; i++){
        for(int j = 0; j < complete_vectors; j++){
            __m256 vector = _mm256_loadu_ps(&image[i][x_start+(j*VLEN)]);
            accumulators[j] = _mm256_add_ps(vector, accumulators[j]);
        }
        __m256 vector = _mm256_maskload_ps(&image[i][x_start+(complete_vectors*VLEN)],mask);
        accumulators[complete_vectors] = _mm256_add_ps(vector, accumulators[complete_vectors]);
    }

    float* a1 = (float*)&accumulators[0];
    float* a2 = (float*)&accumulators[1];
    float* a3 = (float*)&accumulators[2];
    float* a4 = (float*)&accumulators[3];
    float* a5 = (float*)&accumulators[4];

    float (*float_accumulators)[8] = {a1,a2,a3,a4,a5};

    extract_output_sum_right(output,float_accumulators,x_start,0);

    for(int y = 1; y<SIZE; y++){
        if(RADIUS+y < SIZE){
            for(int j = 0; j < complete_vectors; j++){
                __m256 vector = _mm256_loadu_ps(&image[y+RADIUS][x_start+(j*VLEN)]);
                accumulators[j] = _mm256_add_ps(vector, accumulators[j]);
            }
            __m256 vector = _mm256_maskload_ps(&image[y+RADIUS][x_start+(complete_vectors*VLEN)],mask);
            accumulators[complete_vectors] = _mm256_add_ps(vector, accumulators[complete_vectors]);
        }
        if(y > RADIUS){
            for(int j = 0; j < complete_vectors; j++){
                __m256 vector = _mm256_loadu_ps(&image[y-RADIUS-1][x_start+(j*VLEN)]);
                accumulators[j] = _mm256_sub_ps(accumulators[j], vector);
            }
            __m256 vector = _mm256_maskload_ps(&image[y-RADIUS-1][x_start+(complete_vectors*VLEN)],mask);
            accumulators[complete_vectors] = _mm256_sub_ps(accumulators[complete_vectors], vector);
        }
        extract_output_sum_right(output,float_accumulators,x_start,y);
    }
}

void column_sum(float (*image)[SIZE],float (*output)[SIZE], int x_start){
    __m256 accumulator1 = _mm256_loadu_ps(&image[0][x_start]);
    __m256 accumulator2 = _mm256_loadu_ps(&image[0][x_start+VLEN]);
    __m256 accumulator3 = _mm256_loadu_ps(&image[0][x_start+(2*VLEN)]);
    __m256 accumulator4 = _mm256_loadu_ps(&image[0][x_start+(3*VLEN)]);
    __m256 accumulator5 = _mm256_loadu_ps(&image[0][x_start+(4*VLEN)]);

    for(int i = 1; i <= RADIUS; i++){
        __m256 vector1 = _mm256_loadu_ps(&image[i][x_start]);
        __m256 vector2 = _mm256_loadu_ps(&image[i][x_start+VLEN]);
        __m256 vector3 = _mm256_loadu_ps(&image[i][x_start+(2*VLEN)]);
        __m256 vector4 = _mm256_loadu_ps(&image[i][x_start+(3*VLEN)]);
        __m256 vector5 = _mm256_loadu_ps(&image[i][x_start+(4*VLEN)]);
        accumulator1 = _mm256_add_ps(vector1, accumulator1);
        accumulator2 = _mm256_add_ps(vector2, accumulator2);
        accumulator3 = _mm256_add_ps(vector3, accumulator3);
        accumulator4 = _mm256_add_ps(vector4, accumulator4);
        accumulator5 = _mm256_add_ps(vector5, accumulator5);
    }

    float* a1 = (float*)&accumulator1;
    float* a2 = (float*)&accumulator2;
    float* a3 = (float*)&accumulator3;
    float* a4 = (float*)&accumulator4;
    float* a5 = (float*)&accumulator5;

    float (*accumulators)[8] = {a1,a2,a3,a4,a5};

    if(x_start == 0)extract_output_sum_left(output,accumulators,0);
    else if(x_start+VLEN5 >= SIZE) extract_output_sum_right(output,accumulators,x_start,0);
    else extract_output_sum_middle(output,accumulators,x_start,0);

    for(int y = 1; y<SIZE; y++){
        if(RADIUS+y < SIZE){
            __m256 vector1 = _mm256_loadu_ps(&image[y+RADIUS][x_start]);
            __m256 vector2 = _mm256_loadu_ps(&image[y+RADIUS][x_start+VLEN]);
            __m256 vector3 = _mm256_loadu_ps(&image[y+RADIUS][x_start+(2*VLEN)]);
            __m256 vector4 = _mm256_loadu_ps(&image[y+RADIUS][x_start+(3*VLEN)]);
            __m256 vector5 = _mm256_loadu_ps(&image[y+RADIUS][x_start+(4*VLEN)]);
            accumulator1 = _mm256_add_ps(vector1, accumulator1);
            accumulator2 = _mm256_add_ps(vector2, accumulator2);
            accumulator3 = _mm256_add_ps(vector3, accumulator3);
            accumulator4 = _mm256_add_ps(vector4, accumulator4);
            accumulator5 = _mm256_add_ps(vector5, accumulator5);
        }
        if(y > RADIUS){
            __m256 vector1 = _mm256_loadu_ps(&image[y-RADIUS-1][x_start]);
            __m256 vector2 = _mm256_loadu_ps(&image[y-RADIUS-1][x_start+VLEN]);
            __m256 vector3 = _mm256_loadu_ps(&image[y-RADIUS-1][x_start+(2*VLEN)]);
            __m256 vector4 = _mm256_loadu_ps(&image[y-RADIUS-1][x_start+(3*VLEN)]);
            __m256 vector5 = _mm256_loadu_ps(&image[y-RADIUS-1][x_start+(4*VLEN)]);
            accumulator1 = _mm256_sub_ps(accumulator1, vector1);
            accumulator2 = _mm256_sub_ps(accumulator2, vector2);
            accumulator3 = _mm256_sub_ps(accumulator3, vector3);
            accumulator4 = _mm256_sub_ps(accumulator4, vector4);
            accumulator5 = _mm256_sub_ps(accumulator5, vector5);
        }
        if(x_start == 0)extract_output_sum_left(output,accumulators,y);
        else if(x_start+VLEN5 >= SIZE) extract_output_sum_right(output,accumulators,x_start,y);
        else extract_output_sum_middle(output,accumulators,x_start,y);
    }
}

int main() {
    int r = RADIUS;
    float (*image)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));
    float (*output)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));

    fill_image_data(image);

    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);
        #pragma omp parallel for
        for(int x=0; x <= SIZE-VLEN5; x+=VLEN5-(2*RADIUS)){
            column_sum(image,output,x);
        }
        int remainder = (SIZE-2*(VLEN5-RADIUS))%((VLEN5-2*RADIUS));
        if(remainder !=0){
            int x= SIZE-2*RADIUS-remainder;
            column_sum_masked(image,output,x);
        }
        normalize(output);
    gettimeofday(&tv2, NULL);
    printf ("Total time = %f ms\n",
        (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
        (double) (tv2.tv_sec - tv1.tv_sec) * 1000);

    //print_image_data(output);

    return 0;
}
