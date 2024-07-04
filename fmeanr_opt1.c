#include <immintrin.h>
#include <stdio.h>

#define SIZE 22
#define RADIUS 2
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
            printf("%.2f\t",image[y][x]);
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

void extract_output_sum(float (*output)[SIZE],float* accumulator, int x_start, int y){
    if(x_start == 0){
        for(int i = 0; i < RADIUS; i++){
            float sum = 0.0;
            for(int j = 0; j <= i+RADIUS; j++){
                sum += accumulator[j];
            }
            output[y][i+x_start] = sum;
        }
    }

    for(int i = RADIUS; i < VLEN-RADIUS && i < SIZE-x_start-RADIUS; i++){
        float sum = 0.0;
        for(int j = i-RADIUS; j <= i+RADIUS; j++){
            sum += accumulator[j];
        }
        output[y][i+x_start] = sum;
    }

    if(x_start+VLEN >= SIZE){
        for(int i = VLEN-RADIUS-(x_start+VLEN-SIZE); i < SIZE-x_start; i++){
            float sum = 0.0;
            for(int j = i-RADIUS; j < SIZE-x_start; j++){
                sum += accumulator[j];
            }
            output[y][i+x_start] = sum;
        }
    }
}

void column_sum_masked(float (*image)[SIZE],float (*output)[SIZE], int x_start){
    __m256i mask = select_mask((SIZE-x_start));
    __m256 accumulator = _mm256_maskload_ps(&image[0][x_start],mask);
    float* a = (float*)&accumulator;
    for(int i = 1; i <= RADIUS; i++){
        __m256 vector = _mm256_maskload_ps(&image[i][x_start],mask);
        accumulator = _mm256_add_ps(vector, accumulator);
    }

    extract_output_sum(output,a,x_start,0);

    for(int y = 1; y<SIZE; y++){
        if(RADIUS+y < SIZE){
            __m256 vector = _mm256_maskload_ps(&image[y+RADIUS][x_start],mask);
            accumulator = _mm256_add_ps(vector, accumulator);
        }
        if(y > RADIUS){
            __m256 vector = _mm256_maskload_ps(&image[y-RADIUS-1][x_start],mask);
            accumulator = _mm256_sub_ps(accumulator, vector);
        }
        extract_output_sum(output,a,x_start,y);
    }
}

void column_sum(float (*image)[SIZE],float (*output)[SIZE], int x_start){
    __m256 accumulator = _mm256_loadu_ps(&image[0][x_start]);
    float* a = (float*)&accumulator;
    for(int i = 1; i <= RADIUS; i++){
        __m256 vector = _mm256_loadu_ps(&image[i][x_start]);
        accumulator = _mm256_add_ps(vector, accumulator);
    }

    extract_output_sum(output,a,x_start,0);

    for(int y = 1; y<SIZE; y++){
        if(RADIUS+y < SIZE){
            __m256 vector = _mm256_loadu_ps(&image[y+RADIUS][x_start]);
            accumulator = _mm256_add_ps(vector, accumulator);
        }
        if(y > RADIUS){
            __m256 vector = _mm256_loadu_ps(&image[y-RADIUS-1][x_start]);
            accumulator = _mm256_sub_ps(accumulator, vector);
        }
        extract_output_sum(output,a,x_start,y);
    }
}

int main() {
    int r = RADIUS;
    float (*image)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));
    float (*output)[SIZE] = malloc(sizeof(float[SIZE][SIZE]));

    fill_image_data(image);

    //print_image_data(image);

    int x=0;
    for(x; x<=SIZE-VLEN; x+=(VLEN-(2*RADIUS))){
        column_sum(image,output,x);
    }
    if(x>SIZE-VLEN)column_sum_masked(image,output,x);
    normalize(output);
    print_image_data(output);

    return 0;
}
