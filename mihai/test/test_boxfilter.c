#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "minunit.h"

#include<immintrin.h>
#include <stdalign.h>

#include "i.h"
#include "boxfilter.h"


MU_TEST(test_boxfilter1D128) {

  int N = 128;

  int R = 2;
  
  float *a = aligned_alloc(32, N*N*sizeof(float));
  float *b = aligned_alloc(32, N*N*sizeof(float));

  int i;
  
  for (i=0; i<N*N; i++)
    {
      a[i] = 1.0;
    }

  boxfilter1D(a, b, R, N, N);
  
  
  for (i=0; i<N; i++)
    {
      mu_assert_double_close(3.0, b[i], 1e-12);
    }
  for (i=N; i<2*N; i++)
    {
      mu_assert_double_close(4.0, b[i], 1e-12);
    }
  for (i=2*N; i<3*N; i++)
    {
      mu_assert_double_close(5.0, b[i], 1e-12);
    }

  for (i=N*(N-1); i<N*N; i++)
    {
      mu_assert_double_close(3.0, b[i], 1e-12);
    }
  for (i=N*(N-2); i<N*(N-1); i++)
    {
      mu_assert_double_close(4.0, b[i], 1e-12);
    }
  for (i=N*(N-3); i<N*(N-2); i++)
    {
      mu_assert_double_close(5.0, b[i], 1e-12);
    }
  
  
}


MU_TEST(test_boxfilter1D64) {


  int N = 64;

  int R = 2;
  
  float *a = aligned_alloc(32, N*N*sizeof(float));
  float *b = aligned_alloc(32, N*N*sizeof(float));

  int i;
  
  for (i=0; i<N*N; i++)
    {
      a[i] = 1.0;
    }

  boxfilter1D(a, b, R, N, N);
    
  for (i=0; i<N; i++)
    {
      mu_assert_double_close(3.0, b[i], 1e-12);
    }
  for (i=N; i<2*N; i++)
    {
      mu_assert_double_close(4.0, b[i], 1e-12);
    }
  for (i=2*N; i<3*N; i++)
    {
      mu_assert_double_close(5.0, b[i], 1e-12);
    }

  for (i=N*(N-1); i<N*N; i++)
    {
      mu_assert_double_close(3.0, b[i], 1e-12);
    }
  for (i=N*(N-2); i<N*(N-1); i++)
    {
      mu_assert_double_close(4.0, b[i], 1e-12);
    }
  for (i=N*(N-3); i<N*(N-2); i++)
    {
      mu_assert_double_close(5.0, b[i], 1e-12);
    }
  
  
}



MU_TEST(test_boxfilter1D127) {

  int N = 127;

  int ld = 128;

  int R = 2;
  
  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));

  int i;
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = 1.0;
    }

  boxfilter1D(a, b, R, N, ld);

  for (i=0; i<N; i++)
    {
      mu_assert_double_close(3.0, b[i], 1e-12);
    }
  for (i=ld; i<ld+N; i++)
    {
      mu_assert_double_close(4.0, b[i], 1e-12);
    }
  for (i=2*ld; i<2*ld+N; i++)
    {
      mu_assert_double_close(5.0, b[i], 1e-12);
    }

  for (i=ld*(N-1); i<ld*(N-1) + N; i++)
    {
      mu_assert_double_close(3.0, b[i], 1e-12);
    }
  for (i=ld*(N-2); i<ld*(N-2) + N; i++)
    {
      mu_assert_double_close(4.0, b[i], 1e-12);
    }
  for (i=ld*(N-3); i<ld*(N-3) + N; i++)
    {
      mu_assert_double_close(5.0, b[i], 1e-12);
    }
  
  
}




MU_TEST(test_n_less_2rp1) {

  int not_ok;

  not_ok = boxfilter(NULL, NULL, 4, 8, 8, NULL);

  mu_assert(not_ok == -1, "boxfilter return value should be -1");

}



MU_TEST(test_ld_not_divisible_by_V) {

  int N = 127;

  int ld = 127;

  int R = 2;

  int not_ok;
  

  not_ok = boxfilter(NULL, NULL, R, N, ld, NULL);

  mu_assert(not_ok == -2, "boxfilter return value should be -2");


}


MU_TEST(test_n9_r4) {

  int N = 9;

  int ld = 16;

  int R = 4;

  
  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));

  int i;
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = 1.0;
    }

  boxfilter1D(a, b, R, N, ld);

  for (i=0; i<N; i++)
    {
      mu_assert_double_close(R+1, b[i], 1e-12);
    }
  for (i=ld; i<ld+N; i++)
    {
      mu_assert_double_close(R+2, b[i], 1e-12);
    }
  for (i=2*ld; i<2*ld+N; i++)
    {
      mu_assert_double_close(R+3, b[i], 1e-12);
    }
  for (i=3*ld; i<3*ld+N; i++)
    {
      mu_assert_double_close(R+4, b[i], 1e-12);
    }
  for (i=4*ld; i<4*ld+N; i++)
    {
      mu_assert_double_close(R+5, b[i], 1e-12);
    }
  for (i=5*ld; i<5*ld+N; i++)
    {
      mu_assert_double_close(R+4, b[i], 1e-12);
    }
  for (i=6*ld; i<6*ld+N; i++)
    {
      mu_assert_double_close(R+3, b[i], 1e-12);
    }
  for (i=7*ld; i<7*ld+N; i++)
    {
      mu_assert_double_close(R+2, b[i], 1e-12);
    }
  for (i=8*ld; i<8*ld+N; i++)
    {
      mu_assert_double_close(R+1, b[i], 1e-12);
    }
 
}




MU_TEST(test_n9_r4_norm) {

  int N = 9;

  int ld = 16;

  int R = 4;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));

  float *ai = aligned_alloc(32, ld*sizeof(float));
  float *bi = aligned_alloc(32, (R+1)*sizeof(float)); 

  int i, j;

  int r = R;

  for (i = 0; i < r+1; i++)
    {
      ai[i] = 1./(r+1+i)/(2*r+1);
      bi[i] = 1./(r+1+i)*(2*r+1);
    }

  for (i = r+1; i < N - (r+1); i++)
    {
      ai[i] = 1./(2*r+1)/(2*r+1);
    }

  for (i = N - (r+1); i < N; i++)
    {
      ai[i] = ai[N - 1  - i];
    }


  /* printf("\n"); */
  /* for (i = 0; i<N; i++) */
  /*   printf("%f, ", ai[i]); */

  /* printf("\n"); */
  /* for (i =0; i < r+1; i++) */
  /*   printf("%f, ", bi[i]); */
      
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = 1.0;
    }

  boxfilter1D_norm(a, b, R, N, ld, ai, bi);

  float b_res[] = {0.200000, 0.166667, 0.142857, 0.125000, 0.111111, 0.125000, 0.142857, 0.166667, 0.200000};
    //  printf("\n");
    printf("\n");

  for (i=0; i<N; i++)
    {
      for (j=0; j<N; j++)
	{
	  mu_assert_double_close(b_res[j], b[i*ld+j], 1e-6);
	}
    }
  
  
  
  /* printf("\n"); */
  /* for (i=0; i<N; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%f, ", b[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */


}



MU_TEST(test_transpose_8x8) {
  int N = 8;

  int ld = 8;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));
  
  int i, j;
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = (float) i;
    }

  transpose_8x8(a, b, ld);
  
  for (i=0; i<ld; i++)
    {
      for (j = 0; j<N; j++)
	{
	  mu_assert(a[i*ld + j] == b[j*ld + i],"tranposed values should be equal");
	}
      printf("\n");
    }
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%f, ", a[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */

  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%f, ", b[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */


}


MU_TEST(test_transpose_32_32) {

  int N = 32;

  int ld = 32;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));

  int i, j;
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = (float) i;
    }

  transpose(a, b, N, ld);


  for (i=0; i<ld; i++)
    {
      for (j = 0; j<N; j++)
	{
	  mu_assert(a[i*ld + j] == b[j*ld + i],"tranposed values should be equal");
	}
      printf("\n");
    }
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) a[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) b[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */
}


MU_TEST(test_transpose_64_64) {

  int N = 64;

  int ld = 64;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));

  int i, j;
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = (float) i;
    }

  transpose(a, b, N, ld);


  for (i=0; i<ld; i++)
    {
      for (j = 0; j<N; j++)
	{
	  	  mu_assert(a[i*ld + j] == b[j*ld + i],"tranposed values should be equal");
	}
          printf("\n");
    }
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) a[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) b[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */

}


MU_TEST(test_transpose_16_16) {

  int N = 16;

  int ld = 16;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));

  int i, j;
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = (float) i;
    }

  transpose(a, b, N, ld);


  for (i=0; i<ld; i++)
    {
      for (j = 0; j<N; j++)
	{
	    mu_assert(a[i*ld + j] == b[j*ld + i],"tranposed values should be equal");
	}
        printf("\n");
    }
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) a[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) b[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */

}


MU_TEST(test_transpose_40_40) {

  int N = 40;

  int ld = 40;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));

  int i, j;
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = (float) i;
    }

  transpose(a, b, N, ld);


  for (i=0; i<ld; i++)
    {
      for (j = 0; j<N; j++)
	{
	     mu_assert(a[i*ld + j] == b[j*ld + i],"tranposed values should be equal");
	}
        printf("\n");
    }
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) a[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */
  
  /* printf("\n"); */
  /* for (i=0; i<ld; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) b[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */

}


MU_TEST(test_boxfilter_40_40) {

  int N = 40;

  int ld = 40;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));
  float *w = aligned_alloc(32, ld*(ld+2)*sizeof(float));

  int i, j;

  int r = 4;

  for (i = 0; i<N*N; i++)
    a[i] = 1;

  /* printf("\n"); */
  /* for (i=0; i<N; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%d, ",(int) a[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */
  
  boxfilter(a, b, r, N, ld, w);

  
  for (i=0; i<N; i++)
    {
      for (j = 0; j<N; j++)
	{
	  mu_assert_double_close(1.0, b[i*ld + j], 1e-6);
	}
    }
  
  /* printf("\n"); */
  /* for (i=0; i<N; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%f, ", b[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */
  
  
}


MU_TEST(test_n40_r4_norm) {

  int N = 40;

  int ld = 40;

  int R = 4;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));

  float *ai = aligned_alloc(32, ld*sizeof(float));
  float *bi = aligned_alloc(32, (R+1)*sizeof(float)); 

  int i, j;

  int r = R;

  for (i = 0; i < r+1; i++)
    {
      ai[i] = 1./(r+1+i)/(2*r+1);
      bi[i] = 1./(r+1+i)*(2*r+1);
    }

  for (i = r+1; i < N - (r+1); i++)
    {
      ai[i] = 1./(2*r+1)/(2*r+1);
    }

  for (i = N - (r+1); i < N; i++)
    {
      ai[i] = ai[N - 1  - i];
    }


  /* printf("\n"); */
  /* for (i = 0; i<N; i++) */
  /*   printf("%f, ", ai[i]); */

  /* printf("\n"); */
  /* for (i =0; i < r+1; i++) */
  /*   printf("%f, ", bi[i]); */
      
  
  for (i=0; i<ld*N; i++)
    {
      a[i] = 1.0;
    }

  boxfilter1D_norm(a, b, R, N, ld, ai, bi);

  float b_res[] = {0.200000, 0.166667, 0.142857, 0.125000, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.125000, 0.142857, 0.166667, 0.200000};

  printf("\n");

  for (i=0; i<N; i++)
    {
      for (j=0; j<N; j++)
	{
	  mu_assert_double_close(b_res[j], b[i*ld+j], 1e-6);
	}
    }
  
  
  /* printf("\n"); */
  /* for (i=0; i<N; i++) */
  /*   { */
  /*     for (j = 0; j<N; j++) */
  /* 	{ */
  /* 	  printf("%f, ", b[i*ld + j]); */
  /* 	} */
  /*     printf("\n"); */
  /*   } */


}

MU_TEST(test_matmul_40) {

  int N = 24;

  int ld = N;

  float *a = aligned_alloc(32, ld*N*sizeof(float));
  float *b = aligned_alloc(32, ld*N*sizeof(float));
  float *c = aligned_alloc(32, ld*N*sizeof(float));

  int i;
  
  for (i=0; i<N*N; i++)
    {
      a[i] = (float) (i+1) / N;
      b[i] = (float) (i+1) / N;
    }
    
  matmul(a, b, c, N, ld);

  printf("\n");
  for (i=0; i<N*N; i++)
    {
      //printf("%f, ", (a[i]*a[i] - c[i])/c[i]);
      mu_assert_double_close(0, (a[i]*a[i]- c[i])/c[i], 1e-6);
    }

  

}


MU_TEST_SUITE(test_suite) {
  MU_RUN_TEST(test_boxfilter1D128);
  MU_RUN_TEST(test_boxfilter1D64);
  MU_RUN_TEST(test_boxfilter1D127);
  
  MU_RUN_TEST(test_n_less_2rp1);
  MU_RUN_TEST(test_ld_not_divisible_by_V);

  MU_RUN_TEST(test_n9_r4);
  MU_RUN_TEST(test_n9_r4_norm);

  MU_RUN_TEST(test_transpose_8x8);
  MU_RUN_TEST(test_transpose_32_32);
  MU_RUN_TEST(test_transpose_64_64);
  MU_RUN_TEST(test_transpose_16_16);
  MU_RUN_TEST(test_transpose_40_40);
  MU_RUN_TEST(test_boxfilter_40_40);
  MU_RUN_TEST(test_n40_r4_norm);

  MU_RUN_TEST(test_matmul_40);

  
 
}

int main(int argc, char *argv[]) {
        MU_RUN_SUITE(test_suite);
        MU_REPORT();
        return minunit_fail;
}
