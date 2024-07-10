#include <stdio.h>
#include <immintrin.h>
#include <sys/time.h>

#define SIZE 4096
#define RADIUS 10

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
        float sum = 0;
        for(int y=0;y<=RADIUS;y++){
            sum += temp[y][x];
        }
        output[0][x] = sum;
    }
}

void moving_sum_ver(float (*temp)[SIZE], float (*output)[SIZE]){
    for(int y=1;y<SIZE;y++){
        for(int x=0;x<SIZE;x++){
            float sum = output[y-1][x];
            if(RADIUS+y < SIZE){
                sum += temp[y+RADIUS][x];
            }
            if(y>RADIUS){
                sum -= temp[y-RADIUS-1][x];
            }
            output[y][x] = sum;
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
    printf ("normalize time = %f ms\n",
        (double) (tv2.tv_usec - tv1.tv_usec) / 1000 +
        (double) (tv2.tv_sec - tv1.tv_sec) * 1000);

    //print_image_data(output);

    return 0;
}
