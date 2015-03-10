#include <stdio.h>
#include <stdlib.h>
#include <R.h>
#include <Rmath.h>
#include <string.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#endif
#include "functions.h"
#define COMMENTS 0 

void main_loop(int *N, int *p, int *G, double *z, double *sigmar, double *invsigmar, double *mu, double *pi, int *nmax, double *atol, double *mtol, int *mmax, double *x, int *labels, char **covtype, double *logl, int *counter, int *MAPP, double *D){
    int g, i, n;
    int NN = *N;
    int pp = *p;
    int GG = *G;
    int nnmax = *nmax;
    double aatol= *atol;
    double mmtol = *mtol;
    int mmmax = *mmax;
//    double *D = malloc(sizeof(double)*pp*pp);
    double *z1          = malloc(sizeof(double)*NN*GG);
    double **Sigma      = malloc(sizeof(double*)*GG);
    double **invSigma   = malloc(sizeof(double*)*GG);
    double **sampcov    = malloc(sizeof(double*)*GG);
    double *logdet      = malloc(sizeof(double)*GG);
    for(g=0; g < *G; g++) {
        Sigma[g]        = malloc(sizeof(double)*pp*pp);
        invSigma[g]     = malloc(sizeof(double)*pp*pp);
        sampcov[g]      = malloc(sizeof(double)*pp*pp);
    }
    for(i=0; i<pp*pp; i++)
        D[i] = 0.0;
    for(i=0; i<pp; i++)
         D[i*pp + i] = (double)1.0;
    get_mu (pp, GG, NN, x, z, mu);
    mstep(x, NN, pp, GG, z, mu, sampcov, Sigma, invSigma, logdet,  mmtol, mmmax, D, covtype);
    logl[0] = loglik (x, mu, z, NN, pp, GG, invSigma, logdet);
    estep(x, NN, pp, GG, mu, Sigma, invSigma, logdet, labels, z);
    for(i=0; i<pp*pp; i++)
        D[i] = 0.0;
    for(i=0; i<pp; i++)
         D[i*pp + i] = (double)1.0;

    mstep(x, NN, pp, GG, z, mu, sampcov, Sigma, invSigma, logdet, mmtol, mmmax, D, covtype);
    logl[1] = loglik (x, mu, z, NN, pp, GG, invSigma, logdet);
    estep(x, NN, pp, GG, mu, Sigma, invSigma, logdet, labels, z);
    for(i=0; i<pp*pp; i++)
        D[i] = 0.0;
    for(i=0; i<pp; i++)
         D[i*pp + i] = (double)1.0;

    mstep(x, NN, pp, GG, z, mu, sampcov, Sigma, invSigma, logdet, mmtol, mmmax, D, covtype);
    logl[2] = loglik (x, mu, z, NN, pp, GG, invSigma, logdet);
    i = 2;
    while(getall(logl, i) > aatol && i<(nnmax-1)) {
       i = i + 1;
       estep(x, NN, pp, GG, mu, Sigma, invSigma, logdet, labels, z);
       mstep(x, NN, pp, GG, z, mu, sampcov, Sigma, invSigma, logdet, mmtol, mmmax, D, covtype);
       logl[i] = loglik (x, mu, z, NN, pp, GG, invSigma, logdet);
    }
    if(i == nnmax) 
    *counter = i;
    else
     *counter = i + 1;
    estep(x, NN, pp, GG, mu, Sigma, invSigma, logdet, labels, z);
    for(n=0; n<NN; n++) {
        for(g = 0; g < GG; g++) {
            z1[n + NN*g] =  z[n + NN*g];
        }
    }

    MAP(z1, NN, GG, labels, x, pp, mu, Sigma, invSigma, logdet, MAPP);
    for(g=0; g<GG; g++){
       for(i=0; i<pp*pp; i++){
           sigmar[g*pp*pp +i] = Sigma[g][i];
           invsigmar[g*pp*pp +i] = invSigma[g][i];
        }
     }

    get_mu (pp, GG, NN, x, z, mu);
    get_pi( NN, GG, z, pi);
    free(logdet);
    free(z1);
//    free(D);
    for(g=0; g<GG; g++) {
       free(sampcov[g]);
       free(Sigma[g]);
       free(invSigma[g]);
    }
    free(sampcov);
    free(Sigma);
    free(invSigma); 
}

void MAP(double *z, int N, int G, int *labels, double *x, int p, double *mu, double **Sigma, double **invSigma, double *logdet, int *MAPP){
    int i, g;
    double *C = malloc(sizeof(double)*G);
    weights( x, N, p, G, mu, Sigma, invSigma, logdet, z);
    for(i=0; i<N; i++) {
        for(g=0; g<G; g++) {
            C[g] = z[i + N*g];
         }
/* the one because we return the MAPP to R and arrays are indexed starting from 1 not zero like c*/
         MAPP[i] = maxi_loc(C, G)+1;
    }
    free(C);
}

double getall(double logl[], int i) {
    double val;
    double lm1  =  logl[i];
    double lm   =  logl[i-1];
    double lm_1  = logl[i-2];
// valgrind line # 114    
    double am  = (lm1 - lm)/(lm - lm_1);
    double lm1_Inf = lm + (lm1 - lm)/((double)1.0-am);
    val = lm1_Inf - lm ;
    return fabs(val);
}

double loglik (double *x, double *mu, double *z, int N, int p, int G, double **invSigma, double *logdet) {
    int i, g, n;
    double *sum = malloc(sizeof(double)*N)  ;
    double *pi = malloc(sizeof(double)*G);
    double *zlog  = malloc (sizeof(double)*N*G);
    double *delta = malloc(sizeof(double)*N*G);
    for(i=0; i< N*G; i++)
        delta[i]=0.0;
    for(g=0; g < G; g++ ){
        mahalanobis(g, N, p, x, z, G, mu, invSigma[g], delta);
    }

    for (g=0; g < G; g++ ) {
         for(n=0; n< N; n++){ 
             zlog[n + N*g] = -((double)1.0/2.0)*delta[n + N*g]-(1.0/2.0)*logdet[g] - (p/(double)2.0)*(log(2)+log(M_PI));
         }
    }

    for(g=0; g< G; g++) {
        for(n = 0; n < N; n++) {
            zlog[n + N*g] = exp(zlog[n + N*g] );
        }
    }
    for(i=0; i<N; i++)
        sum[i] = 0.0;
    get_pi( N, G, z, pi);

     for(g=0; g<G; g++)
     for ( n =0; n < N; n++ ) {
      for ( g = 0; g < G; g++ ) {
           sum[n] +=  pi[g]*zlog[n + g*N];
       }
     }   

     double sum1 = 0.0;
     for(i = 0; i < N; i++)
       { sum[i] = log(sum[i]);
         sum1 += sum[i];}
       free(sum);
       free(pi);
       free(zlog);
       free(delta);
       return sum1;
}

void weights( double *x, int N, int p, int G, double *mu, double **Sigma, double **invSigma, double *logdet, double *z ){
    int  g, n;
    double *sum = malloc(sizeof(double)*N);
    double *delta = malloc(sizeof(double)*N*G);
    double *pi = malloc(sizeof(double)*G);
    get_pi( N, G, z,  pi);
    for(g=0; g<G; g++ )
        mahalanobis(g, N, p, x, z, G, mu, invSigma[g], delta);
    for(g=0; g<G; g++) {  
        for(n =0; n<N; n++){
            z[n + N*g] = - ((double)1.0/2)*delta[n + N*g]-((double)1.0/2.0)*logdet[g] - (p/(double)2.0)*log(2*M_PI);
        }
    }
    for(n =0; n<N; n++) {
        sum[n] = 0.0;
        for(g=0; g<G; g++) {
            z[n + N*g] = exp(z[n + N*g] + log(pi[g]));
           sum[n] +=  z[n + g*N];
        }
    }  
    for(n=0; n<N; n++) {
        for(g = 0; g < G; g++) {
            z[n + N*g] =  z[n + N*g]/sum[n];
        }
    }
       free(delta);
       free(sum);
       free(pi);
}
void get_pi(int N, int G, double *z, double pi[]){
     int g, i;
     double *zt = malloc(sizeof(double)*N*G);
     for(i =0; i<N; i++ ) 
     for(g=0; g<G; g++) 
         zt[i*G+ g] = z[i + g*N];
     for( g = 0; g < G; g++) 
         pi[g] = 0;
     for( g = 0; g < G; g++) {
           for ( i = 0; i < N; i++ ) {
               pi[g] += z[i+ g*N];
           }
                  pi[g] = pi[g]/N;
        }
     free(zt);
}

void estep(double *x, int N, int p, int G, double *mu, double **Sigma, double **invSigma, double *logdet, int *labels, double *z)
{
    weights( x, N, p, G, mu, Sigma, invSigma, logdet, z);
//    if(!labels) combinewk (z, N, G, labels);
}



void mstep (double *x, int N, int p, int G, double *z, double *mu, double **sampcov,  double **Sigma, double **invSigma,  double *logdet, double mmtol, int mmax, double *D, char **covtype)


{   int g;
    double *pi = malloc(sizeof(double)*G);
    get_mu (p, G, N, x, z, mu);
     for(g = 0; g < G; g++)
         Covariance(N, p, G, x, z, mu, g, sampcov[g]);
       get_pi(N, G, z, pi);
       modeltype(p, pi, G, D, sampcov, Sigma, invSigma, logdet, mmtol, mmax,covtype);

     free(pi);
}

void modeltype(int p, double pi[], int G, double *D, double **sampcov, double **Sigma, double **invSigma, double *logdet, double mmtol, int mmax, char **covtype) {
    if(strcmp(covtype[0], "EII")==0)
       msEII(p, pi, G, sampcov, Sigma, invSigma, logdet);
    if(strcmp (covtype[0], "VII")==0)
       msVII(p, pi, G, sampcov, Sigma, invSigma, logdet);
    if(strcmp (covtype[0], "EEI")==0)
       msEEI(p, pi, G, sampcov, Sigma, invSigma, logdet);
    if(strcmp (covtype[0], "VEI")==0)
       msVEI( p, pi, G, sampcov, Sigma, invSigma, logdet, mmtol, mmax);
    if(strcmp (covtype[0], "EVI")==0)
       msEVI(p, pi, G, sampcov, Sigma, invSigma, logdet); 
    if(strcmp (covtype[0], "VVI")==0)
       msVVI(p, pi, G, sampcov, Sigma, invSigma, logdet);
    if(strcmp (covtype[0], "EEE")==0)
       msEEE(p, pi, G, sampcov, Sigma, invSigma, logdet);
    if(strcmp (covtype[0], "EEV")==0)
       msEEV(p, pi, G, sampcov, Sigma, invSigma, logdet); 
    if(strcmp (covtype[0], "VEV")==0)
       msVEV(p, pi, G, sampcov, Sigma, invSigma, logdet,  mmtol, mmax); 
    if(strcmp (covtype[0], "VVV")==0)
       msVVV(p, pi, G, sampcov, Sigma, invSigma, logdet);
    if(strcmp (covtype[0], "EVE")==0)
       msEVE(p, pi, G, sampcov, Sigma, invSigma, logdet, mmtol, mmax, D);
    if(strcmp (covtype[0], "VVE")==0)
       msVVE(p, pi, G, sampcov, Sigma, invSigma, logdet, mmtol, mmax, D);
    if(strcmp (covtype[0], "VEE")==0)
        msVEE(p, pi, G, sampcov, Sigma, invSigma, logdet, mmtol, mmax);
    if(strcmp (covtype[0], "EVV")==0)
       msEVV(p, pi, G, sampcov, Sigma, invSigma, logdet); 
}
void msEII (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet){
    int i, g;
    double sum  = 0.0;
    double sumd =0.0;
    double *W = malloc(p*p * sizeof(double));
    for(g=0; g<G; g++){
        for(i=0; i<p*p; i++) {
            Sigma[g][i]    = 0.0;
            invSigma[g][i] = 0.0;
        }
    }
    for(i=0; i<G; i++) {
        sum = sum + pi[i];
     } 
     sumSkwt(sampcov, pi, p, G, W);
     for(i=0; i< p*p; i++)
         W[i] = W[i]/sum;
     for(i=0; i<p; i++) 
         sumd += W[i*p + i]; 
     double lam = sumd/(sum*p);
     for(g=0; g<G; g++) {
         for(i=0; i<p; i++) {
              Sigma[g][i*p + i] = lam;
              invSigma[g][i*p + i] = (double)1.0/lam;
         }
     }
       for(g=0; g<G; g++)
         logdet[g] = p*log(lam);
       free(W);
}
void msEEE (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet) {
    int i, g;
    double det, logdetW;
    double *W    = malloc(p*p * sizeof(double));
    double *invW    = malloc(p*p * sizeof(double));
    double **val = malloc(sizeof(double*)*(G));
      for(g=0; g < G; g++) 
          val[g] = malloc(sizeof(double)*(p)*(p));
      for(g = 0; g < G; g++){
          for(i = 0 ; i < p*p; i++) {
              Sigma[g][i]    = 0.0;
              invSigma[g][i] = 0.0;
          }
      }
      for(g=0; g<G; g++)
          logdet[g] = 0.0;
      sumSkwt(sampcov, pi, p, G, W);
      double sum =0.0;
      for (g=0; g< G; g++) 
         sum += pi[g];
      
      for(i=0; i<p*p; i++) 
           W[i] = W[i]/sum;
      
      for (g=0; g< G; g++) 
          for(i=0 ; i<p*p;i++) 
              val[g][i] = W[i];
      determinant(W, p, p, &det);
      logdetW = log(det);
      ginv(p, p, W, invW);

      for (g=0; g< G; g++) {
          for(i=0 ; i<p*p;i++) {
              Sigma[g][i] = W[i];
              invSigma[g][i] = invW[i];
          }
      }
       for( g=0; g < G; g++)
         logdet[g] = logdetW;
    free(W);
    free(invW);
    for(g=0; g<G; g++)
       free(val[g]);
    free(val);


}
void msEEV (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet) {
    int i, g;
    char trans = 'T';
    char notrans = 'N';
    double sum = 0.0;
    double lam =1.0;
    double alpha = 1.0f;
    double beta = 0.0f;
    double *wr= malloc(sizeof(double)*p);
    double *vr= malloc(sizeof(double)*p*p);
    double *A = malloc(sizeof(double)*p*p);
    double *B = malloc(sizeof(double)*p*p);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double *dummy3 = malloc(sizeof(double)*p*p);
    double *dummy4 = malloc(sizeof(double)*p*p);
    double **WK    = malloc(sizeof(double*)*G);
    double **WK1    = malloc(sizeof(double*)*G);
    double **EWK    = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++) { 
        WK[g] = malloc(sizeof(double)*p*p);
        WK1[g] = malloc(sizeof(double)*p*p);
        EWK[g] = malloc(sizeof(double)*p*p);
    }
    for(g=0; g<G; g++) {
        for(i =0; i< p*p; i++){
            WK[g][i] = sampcov[g][i]*pi[g];
            WK1[g][i] = sampcov[g][i]*pi[g];
        }
    }
    for(i =0; i< p*p; i++) {
        A[i] = 0.0;
        B[i] = 0.0;
    }
    for(g=0; g<G; g++) {
        eigen(p, WK1[g], wr, vr);
        for(i=0; i<p*p; i++) {
            EWK[g][i]= vr[i];
            vr[i] = 0.0;
        }
    }
    for(g=0; g<G; g++) {
        dgemm_(&trans,&notrans,&p,&p,&p,&alpha,EWK[g],&p,WK[g],&p,&beta,dummy1,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,EWK[g],&p,&beta,dummy2,&p);
        for(i=0; i<p*p; i++) {
            A[i] = A[i] + dummy2[i];
        }  
    }
    for(i=0; i<p; i++)
        lam = lam*A[i*p + i];
	lam = pow(lam,(double)1.0/(double)p); 

    for(i=0; i<p*p; i++)
        A[i] /= lam;
    for(i=0; i<G; i++)
        sum = sum +pi[i];
    lam /=sum;
    for(i=0; i<p; i++)
        B[i*p + i] = (double)1.0/A[i*p + i];

    for(g=0; g<G; g++) {
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,EWK[g],&p,A,&p,&beta,dummy1,&p);
        dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy1,&p,EWK[g],&p,&beta,dummy2,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,EWK[g],&p,B,&p,&beta,dummy3,&p);
        dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy3,&p,EWK[g],&p,&beta,dummy4,&p);
        for(i=0; i<p*p; i++) {
            Sigma[g][i] = lam*dummy2[i];
            invSigma[g][i] = (double)1.0/lam * dummy4[i];
        }
    }
    for(g=0; g<G; g++)
        logdet[g] = (double)p*log(lam); 
    free(wr);
    free(vr);
    free(A);
    free(B);
    free(dummy1);
    free(dummy2);
    free(dummy3);
    free(dummy4);
    for(g=0; g<G; g++){
       free(WK[g]);
       free(WK1[g]);
       free(EWK[g]);
    }
    free(WK);
    free(WK1);
    free(EWK);

}

void getA(double **Ok, double* A, double *lam, int G, int p) {
    int g, i;
    double prod =1.0;
    double *B = malloc(sizeof(double)*p*p);
    for(i=0; i<p*p; i++){
        A[i] = 0.0;
        B[i] = 0.0;
    }
    for(g=0; g<G; g++){
        for(i=0; i<p*p; i++){
            A[i] = A[i] + Ok[g][i]/lam[g];
         }
     }
        for(i=0; i<p; i++)
            B[i*p+ i] = A[i*p+ i]; 
    for(i=0; i<p*p; i++)
        A[i] = 0.0;
     for(i=0; i<p; i++) {
         prod *= B[i*p+ i];
      } 
     for(i=0; i<p; i++)
         A[i*p+ i] = B[i*p+ i]/pow(prod,(double)1.0/(double)p);
     free(B);
}
void getOk(double **sampcov, double **Ok, double pi[], int G, int p) {
    int i, j, g;
    char notrans = 'N';
    char trans = 'T';
    double alpha = 1.0f;
    double beta = 0.0f;
    double *wr = malloc(sizeof(double)*p);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double **Ok1    = malloc(sizeof(double*)*G);
    double **Wk    = malloc(sizeof(double*)*G);
    double **Wk1    = malloc(sizeof(double*)*G);
    double **EWk    = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++) { 
        Ok1[g] = malloc(sizeof(double)*p*p);
        Wk[g] = malloc(sizeof(double)*p*p);
        Wk1[g] = malloc(sizeof(double)*p*p);
        EWk[g] = malloc(sizeof(double)*p*p);
    }
    for(g=0; g<G; g++) {
        for(i=0; i<p*p; i++) {
            Wk[g][i] = sampcov[g][i]*pi[g];
            Wk1[g][i] = sampcov[g][i]*pi[g];
        }
        eigen(p, Wk1[g], wr, EWk[g]);
    }
    for(g=0; g<G; g++) {
        for(i=0; i< p; i++){
            for(j=0; j< p; j++){
                }
            }
        }
    for(g=0; g<G; g++) {
        dgemm_(&trans,&notrans,&p,&p,&p,&alpha,EWk[g],&p,Wk[g],&p,&beta,dummy1,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,EWk[g],&p,&beta,dummy2,&p);
        for(i=0; i<p*p; i++) {
            Ok1[g][i] = dummy2[i];
        }
    }
    for(g=0; g<G; g++) 
        for(i=0; i< p*p; i++)
            Ok[g][i] = 0; 
    for(g=0; g<G; g++) 
        for(i=0; i< p; i++)
            Ok[g][i*p +i] = Ok1[g][i*p + i];
    free(wr);
    free(dummy1);
    free(dummy2);
    for(g=0; g<G; g++){ 
       free(Wk[g]);
       free(Wk1[g]);
       free(Ok1[g]);
       free(EWk);
    }   
    free(Wk);
    free(Wk1);
    free(Ok1);
    free(EWk);
} 

void getEkOk(double **sampcov, double **Ok, double **EWk, double pi[], int G, int p) {
    int g, i, j;
    char notrans = 'N';
    char trans = 'T';
    double alpha = 1.0f;
    double beta = 0.0f;
    double *wr = malloc(sizeof(double)*p);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double **Wk    = malloc(sizeof(double*)*G);
    double **Wk1    = malloc(sizeof(double*)*G);
    double **EWkt    = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++) { 
        Wk[g] = malloc(sizeof(double)*p*p);
        Wk1[g] = malloc(sizeof(double)*p*p);
        EWkt[g] = malloc(sizeof(double)*p*p);
    }
    for(g=0; g<G; g++) {
        for(i=0; i<p*p; i++) {
            Wk[g][i] = sampcov[g][i]*pi[g];
            Wk1[g][i] = sampcov[g][i]*pi[g];
        }
        eigen(p, Wk1[g], wr, EWk[g]);
    }
    for(g=0; g<G; g++) {
        for(i=0; i< p; i++){
            for(j=0; j< p; j++){
                EWkt[g][i*p + j] = EWk[g][i + j*p];
                }
            }
        }
    for(g=0; g<G; g++) {
        dgemm_(&trans,&notrans,&p,&p,&p,&alpha,EWk[g],&p,Wk[g],&p,&beta,dummy1,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,EWk[g],&p,&beta,dummy2,&p);
        for(i=0; i<p*p; i++) {
            Ok[g][i] = dummy2[i];
        }
    }
    free(wr);
    free(dummy1);
    free(dummy2);
    for(g=0; g<G; g++){
       free(Wk[g]);
       free(Wk1[g]);
       free(EWkt[g]);
   } 
    free(Wk);
    free(Wk1);
    free(EWkt);
}
void msVEV(int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double 
eplison, int maxiter) {
     int i, g;
    char notrans = 'N';
    char trans = 'T';
    double alpha = 1.0f;
    double beta = 0.0f;
    double sum = 0.0;
    double conv[2] = {0.0,0.0}; 
    int count = 1;
    double *lam = malloc(sizeof(double)*G);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double *dummy3 = malloc(sizeof(double)*p*p);
    double *dummy4 = malloc(sizeof(double)*p*p);
    double *lam1 = malloc(sizeof(double)*G);
    double *invA = malloc(sizeof(double)*p);
    double *z = malloc(sizeof(double)*p);
    double *A = malloc(sizeof(double)*p*p);
    double *B = malloc(sizeof(double)*p*p);
    double **Ok    = malloc(sizeof(double*)*G);
    double **EWk    = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++) { 
        Ok[g] = malloc(sizeof(double)*p*p);
        EWk[g] = malloc(sizeof(double)*p*p);
    }
    getEkOk(sampcov, Ok, EWk, pi, G, p);

    for(g=0; g<G; g++) {
        lam[g] = 0.0;
        for(i=0; i< p; i++){
            lam[g] += Ok[g][i*p + i];
        }
        lam[g] = lam[g]/(pi[g]*(double)p);
    }
    getA(Ok, A, lam, G, p);
    for(g=0; g<G; g++) {
        sum =0.0;
        for(i=0; i< p; i++){
            z[i] = Ok[g][i*p + i];
            invA[i] = (double)1.0/A[i*p + i];
            sum +=z[i]*invA[i];
        }
        lam[g] = sum/((double)p*pi[g]);
    }    
    sum = 0.0;
    for(g=0; g<G; g++) {
        lam1[g] = pi[g]*(1.0 + log(lam[g])); 
        sum +=lam1[g];
    }
    conv[0] = sum* (double)p;
    conv[1] = 1000000000.0;
    while((conv[1]-conv[0])/conv[0] > eplison && count < maxiter) {
       getA(Ok, A, lam, G, p);
    for(g=0; g<G; g++) {
        sum = 0.0;
        for(i=0; i< p; i++){
            z[i] = Ok[g][i*p + i];
            invA[i] = (double)1.0/A[i*p + i];
            sum +=z[i]*invA[i];
        }
        lam[g] = sum/(double)(p*pi[g]);
    }
    conv[1] = conv[0];
    sum = 0.0;
    for(g=0; g<G; g++) {
        lam1[g] = pi[g]*(1.0 + log(lam[g]));
        sum +=lam1[g];
    }
    conv[0] = sum *(double)p;
    count = count + 1;
}
    for(i=0; i<p*p; i++)
        B[i] = 0.0;
    for(i=0; i<p; i++)
        B[i*p + i] = (double)1.0/A[i*p + i];
    for(g=0; g<G; g++) {
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,EWk[g],&p,A,&p,&beta,dummy1,&p);
        dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy1,&p,EWk[g],&p,&beta,dummy2,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,EWk[g],&p,B,&p,&beta,dummy3,&p);
        dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy3,&p,EWk[g],&p,&beta,dummy4,&p);
        for(i=0; i<p*p; i++) {
            Sigma[g][i] = lam[g]*dummy2[i];
            invSigma[g][i] = ((double)1.0/lam[g])*dummy4[i];
        }
    }
    for(g=0; g<G; g++) 
        logdet[g] = (double)p*log(lam[g]);

   free(lam);
   free(lam1);
   free(invA);
   free(A);
   free(B);
   free(z);
   free(dummy1);
   free(dummy2);
   free(dummy3);
   free(dummy4);
   for(g=0; g<G; g++){
       free(Ok[g]);
       free(EWk[g]);
       }
   free(Ok);
   free(EWk);
}


void msEVE (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter, double *D){
    int i,g;
    double val;
    double sumg = 0.0;
    char notrans = 'N';
    char trans = 'T';
    double alpha = 1.0f;
    double beta = 0.0f;
    double conv[2];
    double lam = 0.0;
    int count = 1;
    double *W = malloc(sizeof(double)*p*p);
    double *D6 = malloc(sizeof(double)*p*p);
    double *prod = malloc(sizeof(double)*G);
    double *sum = malloc(sizeof(double)*G);
    double *Ak = malloc(sizeof(double)*p*G);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double *dummy3 = malloc(sizeof(double)*p*p);
    double *dummy4 = malloc(sizeof(double)*p*p);
    double **Wk    = malloc(sizeof(double*)*G);
    double **B    = malloc(sizeof(double*)*G);
    double **C    = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++) { 
        Wk[g] = malloc(sizeof(double)*p*p);
        B[g] = malloc(sizeof(double)*p*p);
        C[g] = malloc(sizeof(double)*p*p);
    }
    for(i=0; i<p*p; i++){
        W[i] = 0.0;
    }
    for(g=0; g<G; g++){ 
        for(i=0; i<p*p; i++){
            Wk[g][i] = pi[g]*sampcov[g][i];
            W[i] += Wk[g][i];
        }
    }
    
    for(g=0; g<G; g++) {
        dgemm_(&trans,&notrans,&p,&p,&p,&alpha,D,&p,Wk[g],&p,&beta,dummy1,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,D,&p,&beta,dummy2,&p);
        for(i=0; i<p*p; i++){
            B[g][i] = dummy2[i];
        }
    }
    for(g=0; g<G; g++) {
        for(i=0; i<p; i++){
            Ak[i + g*p] = B[g][i*p + i];
        }
     }
    for(g=0; g<G; g++) 
        prod[g] = 1.0;
    for(g=0; g<G; g++) 
        for(i=0; i<p; i++)
            prod[g] *= Ak[i + g*p]; 
    for(g=0; g<G; g++) 
        for(i=0; i<p; i++)
            Ak[i + g*p]= Ak[i + g*p]/(pow(prod[g],(double)1.0/(double)p));
    newD(D, p, G, Wk, Ak, D6); 
    val = testval(D6, p, G, Wk, Ak); 
    conv[0] = val;
    conv[1] = val + val*(eplison +1 ) ;
    while((conv[1]-conv[0])/fabs(conv[0]) > eplison && count < maxiter){
        for(i=0; i<p*p; i++)
            D[i] = D6[i];
    newD(D, p, G, Wk, Ak, D6); 
    for(g=0; g<G; g++) {
        dgemm_(&trans,&notrans,&p,&p,&p,&alpha,D6,&p,Wk[g],&p,&beta,dummy1,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,D6,&p,&beta,dummy2,&p);{
        for(i=0; i<p*p; i++)
            B[g][i] = dummy2[i];
        }
    }

    for(g=0; g<G; g++) {
        for(i=0; i<p; i++){
            Ak[i + g*p] = B[g][i*p + i];
        }
     }

    for(g=0; g<G; g++) 
        prod[g] = 1.0;
    for(g=0; g<G; g++) 
        for(i=0; i<p; i++)
            prod[g] *= Ak[i + g*p]; 
    for(g=0; g<G; g++) 
        for(i=0; i<p; i++)
            Ak[i + g*p]= Ak[i + g*p]/(pow(prod[g],(double)1.0/(double)p));
    conv[1] = conv[0];
    conv[0] = testval(D6, p, G, Wk, Ak);
    count = count + 1;
}

    for(g=0; g<G; g++)
        for(i=0; i<p*p; i++)
            B[g][i] = 0.0;
    for(g=0; g<G; g++)
       for(i=0; i<p; i++)
            B[g][i*p+i] = (double)1.0/Ak[i + g*p];
    for(g=0; g<G; g++){
       sum[g] = 0.0;
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,D6,&p,B[g],&p,&beta,dummy1,&p);
       dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy1,&p,D6,&p,&beta,dummy2,&p);
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy2,&p,Wk[g],&p,&beta,dummy3,&p);
        for(i=0; i<p; i++) {
            lam += dummy3[i*p + i];
        }
     }
    for(g=0; g<G; g++) 
        sumg += pi[g];
    
    lam = lam/(sumg*(double)p);
    for(g=0; g<G; g++)
        for(i=0; i<p*p; i++)
            C[g][i] = 0.0;
    for(g=0; g<G; g++)
       for(i=0; i<p; i++)
            C[g][i*p+i] = (double)1.0/(lam*Ak[i + g*p]);
    for(g=0; g<G; g++)
        for(i=0; i<p*p; i++)
            B[g][i] = 0.0;
    for(g=0; g<G; g++)
        for(i=0; i<p; i++)
            B[g][i*p+i] = lam*Ak[i + g*p];

    for(g=0; g<G; g++) {
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,D6,&p,B[g],&p,&beta,dummy1,&p);
       dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy1,&p,D6,&p,&beta,dummy2,&p);
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,D6,&p,C[g],&p,&beta,dummy3,&p);
       dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy3,&p,D6,&p,&beta,dummy4,&p);
        for(i=0; i<p*p; i++) {
            Sigma[g][i] = dummy2[i];
            invSigma[g][i] = dummy4[i];
        }
    }
    for(g=0; g<G; g++)
        logdet[g] = (double)p*log(lam);
         
        for(i=0; i<p*p; i++)
            D[i] = D6[i];  
    free(W);
    free(D6);
    free(prod);
    free(sum);
    free(Ak);
    free(dummy1);
    free(dummy2);
    free(dummy3);
    free(dummy4);
    for(g=0; g<G; g++){
       free(Wk[g]);
       free(B[g]);
       free(C[g]);
    }
    free(Wk);
    free(B);
    free(C);
}

void msVVE (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter, double *D){
    int i,g;
    double val;
    char notrans = 'N';
    char trans = 'T';
    double alpha = 1.0f;
    double beta = 0.0f;
    double conv[2];
    int count;
    double *W = malloc(sizeof(double)*p*p);
    double *D6 = malloc(sizeof(double)*p*p);
    double *prod = malloc(sizeof(double)*G);
    double *lam = malloc(sizeof(double)*G);
    double *sum = malloc(sizeof(double)*G);
    double *Ak = malloc(sizeof(double)*p*G);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double *dummy3 = malloc(sizeof(double)*p*p);
    double *dummy4 = malloc(sizeof(double)*p*p);
    double **Wk    = malloc(sizeof(double*)*G);
    double **B    = malloc(sizeof(double*)*G);
    double **C    = malloc(sizeof(double*)*G);

//    for(i=0; i<p*p; i++)
//        D[i] = 0.0;
//    for(i=0; i<p; i++)
//         D[i*p + i] = (double)1.0;

    for(g=0; g<G; g++) {
        Wk[g] = malloc(sizeof(double)*p*p);
        B[g] = malloc(sizeof(double)*p*p);
        C[g] = malloc(sizeof(double)*p*p);
    }
    for(i=0; i<p*p; i++){
        W[i] = 0.0;
    }
    for(g=0; g<G; g++){
        for(i=0; i<p*p; i++){
            Wk[g][i] = pi[g]*sampcov[g][i];
            W[i] += Wk[g][i];
        }
    }
    for(g=0; g<G; g++) {
        dgemm_(&trans,&notrans,&p,&p,&p,&alpha,D,&p,Wk[g],&p,&beta,dummy1,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,D,&p,&beta,dummy2,&p);
        for(i=0; i<p*p; i++){
            B[g][i] = dummy2[i];
        }
    }

    for(g=0; g<G; g++) {
        for(i=0; i<p; i++){
            Ak[i + g*p] = B[g][i*p + i];
        }
     }
    for(g=0; g<G; g++)
        prod[g] = 1.0;
    for(g=0; g<G; g++)
        for(i=0; i<p; i++)
            prod[g] *= Ak[i + g*p];
    for(g=0; g<G; g++)
        for(i=0; i<p; i++)
            Ak[i + g*p]= Ak[i + g*p]/(pow(prod[g],(double)1.0/(double)p));
    newD(D, p, G, Wk, Ak, D6);
    val = testval(D6, p, G, Wk, Ak);
    count = 1;
    conv[0] = val;
    conv[1] = val + val*(eplison +1 ) ;

    while((conv[1]-conv[0])/fabs(conv[0]) > eplison && count < maxiter){
    for(g=0; g<G; g++) {
        dgemm_(&trans,&notrans,&p,&p,&p,&alpha,D6,&p,Wk[g],&p,&beta,dummy1,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,D6,&p,&beta,dummy2,&p);{
        for(i=0; i<p*p; i++)
            B[g][i] = dummy2[i];
        }
    }

    for(g=0; g<G; g++) {
        for(i=0; i<p; i++){
            Ak[i + g*p] = B[g][i*p + i];
        }
     }
    for(g=0; g<G; g++)
        prod[g] = 1.0;
    for(g=0; g<G; g++)
        for(i=0; i<p; i++)
            prod[g] *= Ak[i + g*p];
    for(g=0; g<G; g++)
        for(i=0; i<p; i++)
            Ak[i + g*p]= Ak[i + g*p]/(pow(prod[g],(double)1.0/(double)p));

    for(i=0; i<p*p; i++)
        D[i] = D6[i];
    newD(D, p, G, Wk, Ak, D6);
    conv[1] = conv[0];
    conv[0] = testval(D6, p, G, Wk, Ak);
    count = count + 1;
}
    for(g=0; g<G; g++)
        for(i=0; i<p*p; i++)
            B[g][i] = 0.0;
    for(g=0; g<G; g++)
        for(i=0; i<p; i++)
            B[g][i*p+i] = (double)1.0/Ak[i + g*p];
    for(g=0; g<G; g++){
       sum[g] = 0.0;
       lam[g] = 0.0;
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,D6,&p,B[g],&p,&beta,dummy1,&p);
       dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy1,&p,D6,&p,&beta,dummy2,&p);
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy2,&p,sampcov[g],&p,&beta,dummy3,&p);
        for(i=0; i<p; i++) {
            sum[g] += dummy3[i*p + i]/(double)p;
            lam[g] += dummy3[i*p + i]/(double)p;
        }
     }
    for(g=0; g<G; g++)
        for(i=0; i<p*p; i++)
            C[g][i] = 0.0;
    for(g=0; g<G; g++)
        for(i=0; i<p; i++)
            C[g][i*p+i] = (double)1.0/(lam[g]*Ak[i + g*p]);
    for(g=0; g<G; g++)
        for(i=0; i<p*p; i++)
            B[g][i] = 0.0;
    for(g=0; g<G; g++)
        for(i=0; i<p; i++)
            B[g][i*p+i] = lam[g]*Ak[i + g*p];

    for(g=0; g<G; g++) {
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,D6,&p,B[g],&p,&beta,dummy1,&p);
       dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy1,&p,D6,&p,&beta,dummy2,&p);
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,D6,&p,C[g],&p,&beta,dummy3,&p);
       dgemm_(&notrans,&trans,&p,&p,&p,&alpha,dummy3,&p,D6,&p,&beta,dummy4,&p);
        for(i=0; i<p*p; i++) {
            Sigma[g][i] = dummy2[i];
            invSigma[g][i] = dummy4[i];
        }
    }

    for(g=0; g<G; g++)
        logdet[g] = (double)p*log((lam[g]));
        for(i=0; i<p*p; i++)
            D[i] = D6[i];  
    free(W);
    free(D6);
    free(prod);
    free(lam);
    free(sum);
    free(Ak);
    free(dummy1);
    free(dummy2);
    free(dummy3);
    free(dummy4);
    for(g=0; g<G; g++){
       free(Wk[g]);
       free(B[g]);
       free(C[g]);
    }
    free(Wk);
    free(B);
    free(C);
}

void newD3MM(double *D, int p, int G, double **Wk, double *Ak, double *xk1) {
    int i, j, g;
    char notrans = 'N';
    char trans = 'T';
    double alpha = 1.0f;
    double beta = 0.0f;
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double *dummy3 = malloc(sizeof(double)*p*p);
    double *dummy4 = malloc(sizeof(double)*p*p);
     double *wr = malloc(sizeof(double)*p);
     double *s = malloc(sizeof(double)*p);
     double *lambda = malloc(sizeof(double)*G);
     double *z = malloc(sizeof(double)*p*p);
     double *z1 = malloc(sizeof(double)*p*p);
     double *u = malloc(sizeof(double)*p*p);
     double *vtt = malloc(sizeof(double)*p*p);
     double **B = malloc(sizeof(double*)*G);
     double **Wk1 = malloc(sizeof(double*)*G);
     double **EWk = malloc(sizeof(double*)*G);
     for(g=0; g<G; g++) { 
         B[g] = malloc(sizeof(double)*p*p);
         Wk1[g] = malloc(sizeof(double)*p*p);
         EWk[g] = malloc(sizeof(double)*p*p);
     }
  
    for(i=0; i<p*p; i++)
        z[i] = 0.0;
    for(g=0; g<G; g++){ 
        for(i=0; i<p*p; i++){
            B[g][i] = 0.0;
            Wk1[g][i] = Wk[g][i];
        }
    }
    for(g=0; g<G; g++) 
        for(i=0; i<p; i++)
            B[g][i*p+i] = (double)1.0/Ak[i + g*p];

    for(g=0; g<G; g++){ 
        dgemm_(&notrans,&trans,&p,&p,&p,&alpha,B[g],&p,D,&p,&beta,dummy1,&p);
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,Wk[g],&p,&beta,dummy2,&p);
        eigen(p, Wk1[g], wr, EWk[g]);
        lambda[g] = wr[0];
        for(i=0; i<p*p; i++){
            z[i] = z[i] + dummy2[i] - lambda[g]*dummy1[i];
        }
    }
    for(i=0;i<p;i++)
        for(j=0;j<p;j++) 
            z1[i+p*j] = z[i*p+j];
    svd1(p, p, z, s, u, vtt);
    dgemm_(&notrans,&trans,&p,&p,&p,&alpha,vtt,&p,u,&p,&beta,xk1,&p);

     for(g=0; g<G; g++) { 
         free(B[g]);
         free(Wk1[g]);
         free(EWk[g]);
     }
    free(dummy1);
    free(dummy2);
    free(dummy3);
    free(dummy4);
    free(wr);
    free(lambda);
    free(s);
    free(z);
    free(z1);
    free(u);
    free(vtt);
    free(B);
    free(Wk1);
    free(EWk);
}

void newD4MM(double *xk1, int p, int G, double **Wk, double *Ak, double *xk2) {
    int i, j, g;
    char notrans = 'N';
    char trans = 'T';
    double alpha = 1.0f;
    double beta = 0.0f;
    int loc = 0;
    double *lambda = malloc(sizeof(double)*G);
    double *C = malloc(sizeof(double)*p);
    double *s = malloc(sizeof(double)*p);
    double *s1 = malloc(sizeof(double)*p);
    double *u = malloc(sizeof(double)*p*p);
    double *ut = malloc(sizeof(double)*p*p);
    double *u1 = malloc(sizeof(double)*p*p);
    double *xk = malloc(sizeof(double)*p*p);
    double *vtt = malloc(sizeof(double)*p*p);
    double *vttt = malloc(sizeof(double)*p*p);
    double *vtt1 = malloc(sizeof(double)*p*p);
    double *z = malloc(sizeof(double)*p*p);
    double *z1 = malloc(sizeof(double)*p*p);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double **B = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++)  
        B[g] = malloc(sizeof(double)*p*p);

    for(i=0; i<p*p; i++)
        z[i] = 0.0;
    for(g=0; g<G; g++) 
        for(i=0; i<p*p; i++)
            B[g][i] = 0.0;
    
    for(g=0; g<G; g++) 
        for(i=0; i<p; i++)
            B[g][i*p+i] = (double)1.0/Ak[i + g*p];
    for(g=0; g<G; g++) {
        for(i=0; i<p; i++) {
            C[i] = (double)1.0/Ak[i + g*p];
        }
        lambda[g] = maxi(C, p, loc);
    }
    for(g=0; g<G; g++){ 
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,Wk[g],&p,xk1,&p,&beta,dummy1,&p);
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,B[g],&p,&beta,dummy2,&p);
       for(i=0; i<p*p; i++){
           z[i] = z[i] +dummy2[i] - lambda[g]*dummy1[i];
       }
  }
    svd1(p, p, z, s, u, vtt);
    dgemm_(&notrans,&trans,&p,&p,&p,&alpha,vtt,&p,u,&p,&beta,xk,&p);
    for(i=0; i<p; i++)
        for(j=0; j<p; j++)
            xk2[i + p*j] = xk[i*p + j];
for(i=0;i<p;i++){
  for(j=0;j<p;j++){ 
      ut[i+p*j] = u[i*p+j];
      vttt[i+p*j] = vtt[i*p+j];
   }
 }
    
    dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,vtt,&p,vttt,&p,&beta,dummy1,&p);
    dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,ut,&p,u,&p,&beta,dummy2,&p);
//    printmx(dummy1,p,p);
//    printmx(dummy2,p,p);

    for(g=0; g<G; g++)
        free(B[g]);
    free(lambda);
    free(C);
    free(s);
    free(s1);
    free(u);
    free(u1);
    free(ut);
    free(xk);
    free(vtt);
    free(vtt1);
    free(z);
    free(z1);
    free(dummy1);
    free(dummy2);
    free(B);
    free(vttt);
}
double  testval(double *D6, int p, int G, double **Wk, double *Ak) {
    int i, g;
    double val=0.0;
    char notrans = 'N';
    char trans = 'T';
    double alpha = 1.0f;
    double beta = 0.0f;
    double *sum = malloc(sizeof(double)*G);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *dummy2 = malloc(sizeof(double)*p*p);
    double *dummy3 = malloc(sizeof(double)*p*p);
    double **B = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++)  
        B[g] = malloc(sizeof(double)*p*p);

    for(g=0; g<G; g++) 
        for(i=0; i<p*p; i++)
            B[g][i] = 0.0;
    for(g=0; g<G; g++) 
        for(i=0; i<p; i++)
            B[g][i*p+i] = (double)1.0/Ak[i + g*p];
    for(g=0; g<G; g++){ 
       sum[g] = 0.0;
       dgemm_(&trans,&notrans,&p,&p,&p,&alpha,D6,&p,Wk[g],&p,&beta,dummy1,&p);
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy1,&p,D6,&p,&beta,dummy2,&p);
       dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,dummy2,&p,B[g],&p,&beta,dummy3,&p);
        for(i=0; i<p; i++) {
            sum[g] += dummy3[i*p + i];
        }
     }
    for(g=0; g<G; g++) 
        val += sum[g];
    for(g=0; g<G; g++) 
        free(B[g]);
    free(sum);
    free(B);
    free(dummy1);
    free(dummy2);
    free(dummy3);
    return val;
}



void newD(double *D, int p, int G, double **Wk, double *Ak, double *D6) {
    int i, g;
    double *xk1 = malloc(sizeof(double)*p*p);
    double *xk2 = malloc(sizeof(double)*p*p);
//    double *sign = malloc(sizeof(double)*p);
    double *sign = malloc(sizeof(double)*p*p);
    newD3MM(D, p, G, Wk, Ak, xk1); 
    newD4MM(xk1, p, G,Wk, Ak, xk2);

    for(i=0; i<p*p; i++){
       D6[i] = 0.0;
       sign[i] = 0.0;
    }
//    for(i=0; i<p; i++){
//       if(xk2[i*p + i] < 0.0)
//         sign[i] = -1.0;
//       else if(xk2[i*p + i] == 0.0)
//         sign[i] = 0.0;
//       else if(xk2[i*p + i] > 0.0)
//         sign[i] = 1.0;
//    }
    for(i=0; i<p; i++){
       if(xk2[i*p + i] < 0.0) {
         sign[i*p + i] = -1.0;
      } else if(xk2[i*p + i] == 0.0){
         sign[i*p + i] = 0.0;
      } else if(xk2[i*p + i] > 0.0){
         sign[i*p + i] = 1.0;
      }
  }
    for(g=0; g<p; g++) 
        for(i=0; i<p; i++)
           D6[i + g*p] = sign[i*p + i]*xk2[i + g*p];
//           D6[i + g*p] = xk2[i + g*p];
    free(xk1);
    free(xk2);
    free(sign);
}



double maxi(double *array, int size, int location){
    int c;
    double max;
    max = array[0];
    location = 0;
    for(c=0; c<size; c++){
        if(array[c] > max){
            max  = array[c];
            location = c;
        }
    }
    return max;
}

/* int maxi_loc(double *array, int size, int location){
    int c;
    double max;
    max = array[0];
    location = 0;
    for(c=0; c<size; c++){
        if(array[c] > max){
            max  = array[c];
            location = c;
        }
    }
    return location;
}*/


int maxi_loc(double *array, int size){
    int c;
    double max;
    max = array[0];
    int location = 0;
    for(c=0; c<size; c++){
        if(array[c] > max){
            max  = array[c];
            location = c;
        }
    }
    return location;
}





void msVEE (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter){
    int i,g;
    char notrans = 'N';
    double alpha = 1.0f;
    double beta = 0.0f;
    double sum = 0.0;
    double conv[2];
    int count = 1;
    double det, val1;
    double *W = malloc(sizeof(double)*p*p);
    double *inv = malloc(sizeof(double)*p);
    double *z = malloc(sizeof(double)*p);
    double *lam = malloc(sizeof(double)*G);
    double *wt = malloc(sizeof(double)*G);
    double *val = malloc(sizeof(double)*G);
    double *C = malloc(sizeof(double)*p*p);
    double *dummy1 = malloc(sizeof(double)*p*p);
    double *invC = malloc(sizeof(double)*p*p);
    double **Wk    = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++)  
        Wk[g] = malloc(sizeof(double)*p*p);
    for(i=0; i<p*p; i++) 
        W[i] = 0.0;
    for(g=0; g<G; g++) { 
        for(i=0; i<p*p; i++) {
            Wk[g][i] = pi[g]*sampcov[g][i];
            W[i] = W[i] +  Wk[g][i];
        }
    }
    determinant(W,p,p,&det);
    for(i=0; i<p*p; i++) 
        C[i] = W[i]/pow(det,(double)1.0/(double)p);
        ginv(p,p,C,invC);
    for(g=0; g<G; g++)  
        lam[g] = 0.0;
    for(g=0; g<G; g++) { 
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,sampcov[g],&p,invC,&p,&beta,dummy1,&p);
        for(i=0; i<p; i++) {
            lam[g] += dummy1[i*p + i]; 
        }
        lam[g] = lam[g]/(double)p;
    }

    for(g=0; g<G; g++) {
        sum =0.0;
        for(i=0; i< p; i++){
            z[i] = Wk[g][i*p + i];
            inv[i] = invC[i*p + i];
            sum +=z[i]*inv[i];
        }
        val[g] = sum/lam[g];
    }    
    val1 = 0.0;
    for(g=0; g<G; g++) 
        val1 += val[g]; 
    sum = 0.0;
    for(g=0; g<G; g++) 
        sum += pi[g]*lam[g]; 
    val1 = val1 + (double)p*sum;
    conv[0] = val1;
    conv[1] = val1 + val1*(eplison +1 ) ;
    while((conv[1]-conv[0])/conv[0] > eplison && count < maxiter) {
       for(g=0; g<G; g++) 
         wt[g] = (double)1.0/lam[g];
    sumSkwt(Wk, wt, p, G, C);
    determinant(C,p,p,&det);
    for(i=0; i< p*p; i++)
        C[i] = C[i]/pow(det,(double)1.0/(double)p);
        ginv(p,p,C,invC);
    for(g=0; g<G; g++)  
        lam[g] = 0.0;
    for(g=0; g<G; g++) { 
        dgemm_(&notrans,&notrans,&p,&p,&p,&alpha,sampcov[g],&p,invC,&p,&beta,dummy1,&p);
        for(i=0; i<p; i++) {
            lam[g] += dummy1[i*p + i]; 
        }
        lam[g] = lam[g]/(double)p;
    }
    for(g=0; g<G; g++) {
        sum =0.0;
        for(i=0; i< p; i++){
            z[i] = Wk[g][i*p + i];
            inv[i] = invC[i*p + i];
            sum +=z[i]*inv[i];
        }
        val[g] = sum/lam[g];
    }    
    val1 = 0.0;
    for(g=0; g<G; g++) 
        val1 += val[g]; 
    sum = 0.0;
    for(g=0; g<G; g++) 
        sum += pi[g]*lam[g]; 
    val1 = val1 + (double)p*sum;
    conv[1] = conv[0];
    conv[0] = val1; 
    count = count + 1;
} 

    ginv(p,p,C,invC);
    for(g=0; g<G; g++) { 
        for(i=0; i<p*p; i++) {
            Sigma[g][i] = lam[g]*C[i];
            invSigma[g][i] = ((double)1.0/lam[g])*invC[i];
        }
    }

    for(g=0; g<G; g++)  
        logdet[g] = (double)p*log(lam[g]);
    free(W);
    free(inv);
    free(z);
    free(lam);
    free(wt);
    free(val);
    free(C);
    free(dummy1);
    free(invC);
    for(g=0; g<G; g++)
       free(Wk[g]);
    free(Wk);
}


void msVEI (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter) {
    int i,g;
    int count;
    double sum;
    double conv[2] = {0.0,0.0}; 
    double prod = 1.0;
    double *invB = malloc(sizeof(double)*p);
    double *z = malloc(sizeof(double)*p);
    double *lam = malloc(sizeof(double)*G);
    double *lam1 = malloc(sizeof(double)*G);
    double *wt = malloc(sizeof(double)*G);
    double *W = malloc(sizeof(double)*p*p);
    double *A = malloc(sizeof(double)*p*p);
    double *B = malloc(sizeof(double)*p*p);
    for(g=0; g<G; g++) {
        sum = 0.0; 
        for(i=0; i< p; i++){
             sum += sampcov[g][i*p + i];
        }
        lam[g] = sum/(double)p;
    }
    
    for(g=0; g<G; g++) 
        wt[g] = pi[g]/lam[g];
    sumSkwt(sampcov, wt, p, G, W);
    for(i=0; i<p*p; i++) {
        A[i] = 0.0;
        B[i] = 0.0;
    }
    for(i=0; i<p; i++) {
        A[i*p + i] = W[i*p + i];
        prod *= A[i*p + i];
    }
    for(i=0; i<p; i++) {
        B[i*p + i] = A[i*p + i]/(pow(prod ,(double)1.0/(double)p)) ;
    }

    for(g=0; g<G; g++) {
        sum =0.0;
        for(i=0; i< p; i++){
            z[i] = sampcov[g][i*p + i];
            invB[i] = (double)1.0/B[i*p + i];
            sum +=z[i]*invB[i];
        }
        lam[g] = sum/(double)p;
    }    
    sum = 0.0;
    for(g=0; g<G; g++) {
        lam1[g] = pi[g]*(1.0 + log(lam[g])); 
        sum +=lam1[g];
    }
    conv[0] = sum* (double)p;
    conv[1] = 100000.0;
    count = 1;
    while(fabs(conv[1]-conv[0]) > eplison && count < maxiter) {
    for(g=0; g<G; g++) 
        wt[g] = pi[g]/lam[g];
        sumSkwt(sampcov, wt, p, G, W);
        prod = 1.0;
    for(i=0; i<p; i++) {
        A[i*p + i] = W[i*p + i];
        prod *= A[i*p + i];
    }
    for(i=0; i<p; i++) {
        B[i*p + i] = A[i*p + i]/(pow(prod ,(double)1.0/(double)p)) ;
    }
    for(g=0; g<G; g++) {
        sum =0.0;
        for(i=0; i< p; i++){
            z[i] = sampcov[g][i*p + i];
            invB[i] = (double)1.0/B[i*p + i];
            sum +=z[i]*invB[i];
        }
        lam[g] = sum/(double)p;
    }    

    conv[1] = conv[0];
    sum = 0.0;
    for(g=0; g<G; g++) {
        lam1[g] = pi[g]*(1.0 + log(lam[g])); 
        sum +=lam1[g];
    }
    conv[0] = sum* (double)p;
    count = count + 1;
    }
    for(g=0; g<G; g++) {
        for(i=0; i< p*p; i++){
            Sigma[g][i] = lam[g]*B[i];
            invSigma[g][i] = 0.0;
        }
    }
    for(g=0; g<G; g++){ 
        for(i=0; i<p; i++){
          invSigma[g][i*p + i] = ((double)1.0/B[i*p + i])*((double)1.0/lam[g]);
       }
   }
    for(g=0; g<G; g++)
        logdet[g] = (double)p*log(lam[g]);

    free(lam);
    free(lam1);
    free(invB);
    free(z);
    free(A);
    free(B);
    free(wt);
    free(W);
}
void msVVV (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet) {
    int i,g;
    double det;
    double **inv    = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++)  
        inv[g] = malloc(sizeof(double)*p*p);
    for(g=0; g<G; g++) {
         ginv(p,p,sampcov[g],inv[g]);
        for(i=0; i< p*p; i++){
            Sigma[g][i] = sampcov[g][i];
            invSigma[g][i] = inv[g][i];
        }
    }
    for(g=0; g<G; g++){ 
        determinant(sampcov[g],p,p,&det);
        logdet[g] = log(det);
      }
    for(g=0; g<G; g++) 
       free(inv[g]);
      free(inv);
}
void msEVV (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet) {
    int i, g;
    double det, sum;
    double* lam = malloc(sizeof(double)*G);
    double **WK    = malloc(sizeof(double*)*G);
    double **CK    = malloc(sizeof(double*)*G);
    double **inv    = malloc(sizeof(double*)*G);
    for(g=0; g<G; g++) { 
        WK[g] = malloc(sizeof(double)*p*p);
        CK[g] = malloc(sizeof(double)*p*p);
        inv[g] = malloc(sizeof(double)*p*p);
    }
    for(g=0; g<G; g++) {
        for(i =0; i< p*p; i++) {
            WK[g][i] = sampcov[g][i]*pi[g];
        }
        determinant(WK[g], p, p, &det);
        lam[g] = pow(det,(double)1.0/(double)p)  ;
    }
    for(g=0; g<G; g++) 
        for(i =0; i< p*p; i++) 
            CK[g][i] = WK[g][i]*((double)1.0/lam[g]);
    sum = 0.0;
    for(g=0; g<G; g++) 
       sum += lam[g];

      for(g = 0; g < G; g++)
         ginv(p,p,CK[g],inv[g]);
      for(g = 0; g < G; g++){
          for(i = 0 ; i < p*p; i++){ 
              Sigma[g][i]    = sum*CK[g][i];
              invSigma[g][i] = ((double)1.0/sum)*inv[g][i];
          }
       }
       for( g=0; g < G; g++)
         logdet[g] = (double)p*log(sum);
    free(lam);
    for(g=0; g<G; g++){
       free(WK[g]);
       free(CK[g]);
       free(inv[g]);
    }
    free(WK);
    free(CK);
    free(inv);
}

void msEEI (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet) {

    int i, g;
    double sum =0.0;
    double *W = malloc(p*p * sizeof(double));
    double *B = malloc(p*p * sizeof(double));
    sumSkwt(sampcov, pi, p, G, W);
    for(g=0; g<G; g++)
        sum += pi[g];
    for(i=0; i<p*p; i++)
        W[i] /=sum;
    for(i=0; i<p*p; i++)
        B[i] = 0.0;
    for(i=0; i<p; i++)
        B[i*p +i] = W[i*p + i];
  
    for(g=0; g < G; g++) {
        for(i =0; i < p*p; i++){
            Sigma[g][i]    = 0.0;
            invSigma[g][i] = 0.0;
        }
    }
    for(g=0; g < G; g++)
        logdet[g] = 0.0;

    for(g=0; g<G; g++) {
        for(i=0; i<p; i++) {
            Sigma[g][i*p + i] = B[i*p+ i];
            invSigma[g][i*p + i] = (double)1.0/B[i*p + i];
        }
    }
    sum = 0.0;
    for(i=0; i<p; i++)
        sum += log(B[i*p + i]); 
    for(g=0; g < G; g++) 
        logdet[g] = sum;
     

    free(W);
    free(B);
}


void msVII (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet)
{
    int i, g;
    double *sumdiagSkg = malloc(sizeof(double)*G);
    for(g = 0; g < G; g++){
        for(i = 0 ; i < p*p; i++) {
            Sigma[g][i]    = 0.0;
            invSigma[g][i] = 0.0;
        }
    }
    for(g=0; g < G; g++)
        logdet[g] = 0.0;

    for(g = 0; g < G; g++){
         sumdiagSkg[g] = 0.0;
        for(i=0; i<p; i++) {
            sumdiagSkg[g] += sampcov[g][i*p +i];
        }
    }
    for(g = 0; g < G; g++){
        for(i=0; i<p; i++) {
            Sigma[g][i*p + i]    = sumdiagSkg[g]/(double)p;
            invSigma[g][i*p + i] = (double)p/sumdiagSkg[g];
        }
    }

    for(g = 0; g < G; g++)
        logdet[g] = p*log(sumdiagSkg[g])- p*log(p);
    free(sumdiagSkg);
}

void msEVI (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet) {
    int i, g;
    double suml = 0.0;
    double sumn = 0.0;
    double *Bk = malloc(sizeof(double)*p*G);
    double *lam = malloc(sizeof(double)*G);
    for(g=0; g < G; g++ ) {
        for(i =0; i < p; i++) {
            Bk[i + p*g] =  sampcov[g][i*p +i]*pi[g];
        }
    }

    for(g=0; g < G; g++) {
        lam[g] = 1.0;
        for(i =0; i < p; i++){
            lam[g] *= Bk[i + p*g]; 
        }
        lam[g] = pow(lam[g], (double)1.0/(double)p);
    }   
    
    for(g=0; g < G; g++) {
        suml += lam[g];
        sumn += pi[g];
    }
    suml = suml/sumn;
   
    for(i =0; i < p; i++) {
        for(g=0; g < G; g++) {
            Bk[i + g*p] = Bk[i + g*p] *(double)1.0/lam[g]; 
        }
     }
    for(g = 0; g < G; g++){
        for(i = 0 ; i < p*p; i++) {
            Sigma[g][i]    = 0.0;
            invSigma[g][i] = 0.0;
        }
    }
    for(g=0; g < G; g++)
        logdet[g] = 0.0;

    for(g=0; g < G; g++ ) {
        for(i =0; i < p; i++) {
            Sigma[g][i*p + i] = suml* Bk[i + p*g];
            invSigma[g][i*p + i] = ((double)1.0/suml)*((double)1.0/Bk[i + p*g]);
        }
    }
    for(g = 0; g < G; g++)
        logdet[g] = p*log(suml);
    free(Bk);
    free(lam);

}

void msVVI(int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet) {
    int i, g;
    double *sumlog = malloc(sizeof(double)*G);
    for(g = 0; g < G; g++){
        for(i = 0 ; i < p*p; i++) {
            Sigma[g][i]    = 0.0;
            invSigma[g][i] = 0.0;
        }
    }
    for(g=0; g < G; g++)
        logdet[g] = 0.0;

    for(g = 0; g < G; g++){
        sumlog[g] = 0.0;
        for(i=0; i<p; i++) {
            Sigma[g][i*p + i]    = sampcov[g][i*p + i];
            sumlog[g] += log(sampcov[g][i*p + i]); 
            invSigma[g][i*p + i] =(double)1.0/sampcov[g][i*p + i]; 
        }
    }
    for(g = 0; g < G; g++){
        logdet[g] = sumlog[g];
      }
    free(sumlog);
}


void  sumSkwt (double **sampcov , double pi[], int p, int G, double *W)
{

       int i, g ;
       for( i=0; i < p*p; i++ )
          W[i] = 0;
       for( g = 0; g < G; g++ ) {
          for ( i = 0; i < p*p; i++ )  {
               W[i] = W[i] + pi[g]*sampcov[g][i];
              }
         }
}

void printmx(double *A, int r, int c) {
    int i, j;
        for(i=0; i < r; i++) {
                for(j=0; j < c; j++)
                        Rprintf("%12.8f ", A[i + r*j]);
                Rprintf("\n");
        }
        Rprintf("\n");
}


void mahalanobis(int g, int N, int p, double *x, double *z, int G, double *mu, double *cov, double *delta)
{
    get_mu ( p,  G,  N, x, z, mu);
    int i, j, n;
    double sum, insum;
    double *inv = (double *) malloc(sizeof(double)*p*p);
//    ginv(pp, pp, cov, inv);
//    inverse(cov, pp);

    for (n = 0; n < N; n++){ 
              sum = 0.0;
         for(j = 0; j < p; j++){
            insum = 0;
               for(i = 0; i < p; i++)
                  insum += (x[n+ N*i] - mu[g+G*i])*cov[i+j*p];
                   sum += insum*(x[n+ N*j] - mu [g+G*j]);
            }
       delta[n+N*g] = sum;
       }
       free(inv);
}


// compute the reverse condition number of the matrix A.
int determinant(double *A, __CLPK_integer k, __CLPK_integer lda, double *res) {
        int i;
        __CLPK_integer *ipiv;
        double *C;
        __CLPK_integer info=0;

        C = (double*)malloc(sizeof(double)*k*k);
        copymx(A, k, k, lda, C);

        ipiv = (__CLPK_integer *)malloc(sizeof(__CLPK_integer)*k);

        dgetrf_(&k, &k, C, &k, ipiv, &info);

        if(COMMENTS && info != 0)
                Rprintf("Failed in computing matrix determinant.\n");

        *res=1.0;
        for(i=0; i < k*k; i++) {
          if(i%k == i/k)
                  (*res) *= C[i];
        }

        if(*res < 0)     //computing absolute value of the determinant
          *res = -(*res);

        free(ipiv);
        free(C);

        return info;
}

void svd(int M, int N, double *A, double *s, double *u, double *vtt) {
#define LDA M               //M is number of rows of A
#define LDU M
#define LDVT N              //N is number of columns of A
// double s[N], u[LDU*M], vt[LDVT*N], vtt[LDVT*N];

        int i, j;
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
        double wkopt;
        double* work;
        int iwork[8*N];
        double vt[LDVT*N];
        lwork = -1;
        dgesdd_( "Singular vectors", &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, &wkopt,
        &lwork, iwork, &info );
        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );
        dgesdd_( "Singular vectors", &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work,
        &lwork, iwork, &info );
//        if( info > 0 ) {
//                Rprintf( "The algorithm computing SVD failed to converge.\n" );
//          }
      for(i=0;i<N;i++) 
          for(j=0;j<N;j++)
              vtt[i+N*j] = vt[i*N+j];
           
//       print_matrix( "Singular values", 1, n, s, 1 );      
//       print_matrix( "Left singular vectors: U (stored columnwise)", m, n, u, ldu );
//       print_matrix( "Left singular vectors: U ", m, n, u, ldu );
//       print_matrix( "Right singular vectors: V (stored rowwise)", n, n, vtt, ldvt );
//       print_matrix( "Right singular vectors: V ", n, n, vtt, ldvt );
       free( (void*)work );
}
       void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        Rprintf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) Rprintf( " %6.2f", a[i+j*lda] );
                Rprintf( "\n" );
        }
}


void svd1(int M, int N, double *A, double *s, double *u, double *vtt) {
#define LDA M               //M is number of rows of A
#define LDU M
#define LDVT N              //N is number of columns of A
        int i, j;
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
        double wkopt;
        double* work;
        double vt[LDVT*N];
        lwork = -1;
        dgesvd_( "All", "All", &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
         &info );
        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );
        dgesvd_( "All", "All", &m, &n, A, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
         &info );
//        if( info > 0 ) {
//                Rprintf( "The algorithm computing SVD failed to converge.\n" );
//                exit( 1 );
//         }
      for(i=0;i<M;i++) 
          for(j=0;j<N;j++)
              vtt[i*N+j] = vt[i+j*M];

//        print_matrix( "Left singular vectors U", m, n, u, ldu );
//        print_matrix( "Right singular vectors V", n, n, vtt, ldvt );
        free( (void*)work );
}
//void print_matrix( char* desc, int m, int n, double* a, int lda ) {
//        int i, j;
//        Rprintf( "\n %s\n", desc );
//        for( i = 0; i < m; i++ ) {
//                for( j = 0; j < n; j++ ) Rprintf( " %8.4f", a[i+j*lda] );
//                Rprintf( "\n" );
//        }
//}


void eigen(int N, double *A, double *wr, double *vr) {
    int lda = N, ldvl = N, ldvr = N, info, lwork;
    double wkopt;
    double* work;
    double  *wi = malloc(sizeof(double)*N); 
     double vl[ldvl*N];
   
    /* Query and allocate the optimal workspace */
     lwork = -1;
     dgeev_( "Vectors", "Vectors", &N, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr,
     &wkopt, &lwork, &info );
     lwork = (int)wkopt;
     work = (double*)malloc( lwork*sizeof(double) );
     /* Solve eigenproblem */
     dgeev_( "Vectors", "Vectors", &N, A, &lda, wr, wi, vl, &ldvl, vr, &ldvr,
     work, &lwork, &info );
      /* Check for convergence */
//        if( info > 0 ) {
//                Rprintf( "The algorithm failed to compute eigenvalues.\n" );
//                exit( 1 );
//       }
//        print_eigenvalues( "Eigenvalues", N, wr, wi );
//        print_eigenvectors( "Left eigenvectors", N, wi, vl, ldvl );
//        print_eigenvectors( "Right eigenvectors", N, wi, vr, ldvr );
        dgeev_sort(wr, wi, vr, N);
//       printmx(vr, N, N); 
        free(wi);
        free( (void*)work );
//        exit( 0 );
}

void dgeev_sort(double *Er, double *Ei, double *vr, int N)
{
	double temp;
        double *E2 = malloc(sizeof(double)*N);
	int i, j, k;

	for (i=0; i<N; i++)
		E2[i] = Er[i]*Er[i]+Ei[i]*Ei[i];

	for (j=0; j<N; j++)
		for (i=0; i<N-1; i++)
			if (fabs(E2[i])<fabs(E2[i+1]))
			{
				temp = E2[i]; E2[i] = E2[i+1]; E2[i+1] = temp;
				temp = Er[i]; Er[i] = Er[i+1]; Er[i+1] = temp;
				temp = Ei[i]; Ei[i] = Ei[i+1]; Ei[i+1] = temp;

				for (k=0; k<N; k++)
				{
					temp = vr[k + i*N];
					vr[k + i*N] = vr[k + (i+1)*N];
					vr[k + (i+1)*N] = temp;
				}
			}

       free(E2);
}

void print_eigenvalues( char* desc, int n, double* wr, double* wi ) {
        int j;
        Rprintf( "\n %s\n", desc );
   for( j = 0; j < n; j++ ) {
      if( wi[j] == (double)0.0 ) {
         Rprintf( " %6.3f", wr[j] );
      } else {
         Rprintf( " (%6.3f,%6.3f)", wr[j], wi[j] );
      }
   }
   Rprintf( "\n" );
}

void print_eigenvectors( char* desc, int n, double* wi, double* v, int ldv ) {
        int i, j;
        Rprintf( "\n %s\n", desc );
   for( i = 0; i < n; i++ ) {
      j = 0;
      while( j < n ) {
         if( wi[j] == (double)0.0 ) {
            Rprintf( " %6.3f", v[i+j*ldv] );
            j++;
         } else {
            Rprintf( " (%6.3f,%6.2f)", v[i+j*ldv], v[i+(j+1)*ldv] );
            Rprintf( " (%6.3f,%6.2f)", v[i+j*ldv], -v[i+(j+1)*ldv] );
            j += 2;
         }
      }
      Rprintf( "\n" );
   }
}


// compute the pseudo-inverse of the matrix A and store it in B
int ginv(__CLPK_integer n, __CLPK_integer lda, double *A, double *B) {
        int i;
        __CLPK_integer info;
        __CLPK_integer NRHS = n;
        __CLPK_integer LDWORK = -1;
        __CLPK_integer LDB= n;
        __CLPK_integer IRANK = -1;
        double RCOND = -1.0f;
        double *WORK = (double *)malloc(sizeof(double));
        double *sing = (double *)malloc(sizeof(double)*n);
        double *C = (double *) malloc(sizeof(double)*n*n);

        for(i=0; i < n*n; i++) {
                if(i/n == i%n )
                        B[i] = 1.0;
                else
                        B[i] = 0.0;
        }

        copymx(A, n, n, lda, C);

        dgelss_(&n, &n, &NRHS, C, &n, B, &LDB, sing, &RCOND, &IRANK, WORK, &LDWORK, &info);
        if(COMMENTS && info != 0) {
                Rprintf("Failed in computing pseudo inverse.\n");
        } else {
                LDWORK = WORK[0];
          free(WORK);
          WORK = (double *)malloc(sizeof(double)*LDWORK);
          dgelss_(&n, &n, &NRHS, C, &n, B, &LDB, sing, &RCOND, &IRANK, WORK, &LDWORK, &info);

          if(COMMENTS && info != 0) {
                  Rprintf("Failed in computing pseudo inverse.\n");
          }
        }
        free(WORK);
        free(sing);
        free(C);

        return info;
}

void copymx(double *A, int r, int c, int lda, double *C) {
  int i,j;
  for(i=0; i < r; i++) {
        for(j=0; j < c; j++)
                C[i + j*r] = A[i + j*lda];
  }
}





void inverse( double* A, int N )
{
    int *IPIV = (int*)malloc (N *sizeof(int)) ;
    int LWORK = N*N;
    double *WORK = (double*)malloc(LWORK *sizeof( double));
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
}

void Covariance(int N, int p, int G, double *x, double *z, double *mu, int g, double *sampcov) {
  int j,k,i;
        double sum,  wsum, wsum2;
        double *Wt = (double*)malloc(sizeof(double)*N);

  sum=0;
        for(i=0; i < N; i++) {
                Wt[i] = z[i + N*g];
                sum  += Wt[i];
        }
        for(i=0; i < N; i++) 
          Wt[i] /= sum;

        for(j=0; j < p; j++) {
          for(k=0; k < p; k++) {
                        wsum = 0.0;
                        wsum2 = 0.0;
                        sampcov[j + k*p] = 0;
                for(i=0; i < N; i++) {
                sampcov[j + k*p] += Wt[i]*(x[i+N*j]-mu[g+G*j])*(x[i+N*k]-mu[g+G*k]);
                    wsum += Wt[i];
                    wsum2 += Wt[i]*Wt[i];
                                }
                  
                   }
                }
  
        free(Wt);
}

void get_mu (int p, int G, int N, double *x, double *z, double *mu)
{     int j, g, i;
       double *ng = malloc(sizeof(double)*G);
        get_ng( N,  G, z, ng);

         for(g=0; g < G; g++) {
            for(j=0; j < p; j++) {
               mu[g + j*G] = 0.0;
                 for(i=0; i < N; i++)
                    { mu[g + j*G] += x[i + j*N]*z[i + g*N];
                  }
                 }
               }

    for( g = 0; g < G; g++)
       for( j=0; j < p; j++)
         mu[g + j*G] = mu[g + j*G]/ng[g];
free(ng);

}


void get_ng(int N, int G, double *z, double ng[])
{
   int i, g;
   for(g = 0; g < G; g++){
       ng[g] = 0;
       for(i = 0; i < N; i++){
           ng[g] += z[i + g*N];
        }
     }
}


void rwgpar (double *z, int *N, int *G, int *labels)
{
      int NN = *N;
      int GG = *G;
      

      GetRNGstate();
      int i, g;
      double *z1 = malloc(sizeof(double)*NN*GG);
      double *sum = malloc(sizeof(double)*NN);
      for(i=0; i < NN; i++)
       { for(g=0; g < GG; g++)
          {  z1[i + g*NN] = unif_rand();
             Rprintf("%f\n", exp_rand());
          }
        }
      PutRNGstate();

     for(i=0; i < NN; i++)
       {   sum[i]=0.0;
           for(g=0; g < GG; g++)
          {sum[i] += z1[i + g*NN];
           }
        }
      for(i=0; i < NN; i++)
       { for(g=0; g < GG; g++)
         {z[i + g*NN] = z1[i + g*NN]/sum[i];}
        }

      for(i=0; i < NN; i++)
       { for(g=0; g < GG; g++)
         {Rprintf("%f",z[i + g*NN]);}
           Rprintf("\n") ;
        }
       combinewk(z, NN, GG, labels); 
}
void combinewk (double *z, int N, int G, int *labels)
{
     int i, g, n;
     int sum =0;
     for(i = 0; i < N; i++)
     {  if(labels[i] ==0)
           break;
        else
        sum = sum +1;
     }
     for(n=0; n<sum; n++)
         for(g=0; g<G; g++){
             z[n + N*g] = 0.0;
         }
       for (i = 0; i<N; i++){
           if(labels[i] ==0){
               Rprintf("broke\n");
               break;
           }else{
               z[i + N*(labels[i]-1)] =  (double)1.00;
           }
       }
}






