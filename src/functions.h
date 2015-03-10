#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
typedef int __CLPK_integer;
extern void dgeev( char* jobvl, char* jobvr, int* n, double* a,
                int* lda, double* wr, double* wi, double* vl, int* ldvl,
                double* vr, int* ldvr, double* work, int* lwork, int* info );
//extern void dgesdd_( char* jobz, int* m, int* n, double* a,
//                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
//                double* work, int* lwork, int* iwork, int* info );
extern void print_matrix( char* desc, int m, int n, double* a, int lda );
void newD(double *D, int p, int G, double **Wk, double *Ak, double *xk1); 
double  testval(double *D, int p, int G, double **Wk, double *Ak); 
void MAP(double *z, int N, int G, int *labels, double *x, int p, double *mu, double **Sigma, double **invSigma, double *logdet, int *vec);
int maxi_loc(double *array, int size);

void Covariance(int N, int p, int G, double *x, double *z, double *mu, int g, double *sampcov); 
void get_pi(int N, int G, double *z, double pi[]);
void mahalanobis(int g, int N, int p, double *x, double *z, int G, double *mu, double *cov, double *delta);
void inverse( double* A, int N );
void copymx(double *A, int r, int c, int lda, double *C); 
int ginv(__CLPK_integer n, __CLPK_integer lda, double *A, double *B); 
void combinewk (double *z, int N, int G, int labels[]);
void get_mu (int p, int G, int N, double *x, double *z, double *mu);
void get_ng(int N, int G, double *z, double ng[]);
int determinant(double *A, __CLPK_integer k, __CLPK_integer lda, double *res);
void msEII (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet);
void msEEI (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet);
void msVII (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet);
void msVEE (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter);
void msEVI (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet); 
void msVVI(int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet);
void msEEE (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet);
void msEEV (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet); 
void msVEV(int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter); 
void msVEI (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter);
void  sumSkwt (double **sampcov , double ng[], int p, int G, double *W);
void printmx(double *A, int r, int c);
double loglik (double *x, double *mu, double *z, int N, int p, int G, double **invSigma, double *logdet );
void msVVV (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet);
void msEVV (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet); 
void msEVE (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter, double *D);

void msVVE (int p, double pi[], int G, double **sampcov, double **Sigma, double **invSigma, double *logdet, double eplison, int maxiter, double *D) ;
double maxi(double *array, int size, int location);
void dgeev_sort(double *Er, double *Ei, double *vr, int N);
void eigen(int N, double *A, double *wr, double *vr);
void svd(int M, int N, double *A, double *s, double *u, double *vtt); 
void svd1(int M, int N, double *A, double *s, double *u, double *vtt) ;
void print_eigenvalues( char* desc, int n, double* wr, double* wi ); 
void print_eigenvectors( char* desc, int n, double* wi, double* v, int ldv ); 
void weights( double *x, int N, int p, int G, double *mu, double **Sigma, double **invSigma, double *logdet, double *z );
double getall(double logl[], int i);
void estep(double *x, int N, int p, int G, double *mu, double **Sigma, double **invSigma, double *logdet, int *labels, double *z);
void mstep (double *x, int N, int p, int G, double *z, double *mu, double **sampcov,  double **Sigma, double **invSigma,  double *logdet, double mmtol, int mmax, double *D, char **covtype);
void modeltype(int pp, double pi[], int GG, double *D, double **sampcov, double **Sigma, double **invSigma, double *logdet,
double eplison, int maxiter, char **covtype); 
void main_loop(int *N, int *p, int *G, double *z, double *sigmar, double *invsigmar, double *mu, double *pi, int *nmax, double *atol, double *mtol, int *mmax, double *x, int *labels, char **covtype, double *logl, int *counter, int *MAPP, double *D);
