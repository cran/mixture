#include <armadillo> 
#include <random>
#include <array>  
#include "math.h"
#include <Rmath.h>



// RANDOM GENERATOR 
#pragma once 
std::default_random_engine generator_latent;
#pragma once
std::uniform_real_distribution<double> unif(0.0, 1.0);

#pragma once 
double _gig_mode(double lambda, double omega); 
#pragma once
double _ratio_of_uniforms_shift(double lambda, double omega, double alpha);
#pragma once
double _ratio_of_uniforms_noshift(double lambda, double omega, double alpha); 
#pragma once 
double _leydold(double lambda, double omega, double alpha); 




#pragma once 
double random_gig_draw(double lambda, double chi, double psi)
/*---------------------------------------------------------------------------*/
/* Draw sample from GIG distribution.                                        */
/* Based on certain criteria.                                                */
/*               Modified by Nikola Pocuca (2021)                            */
/*---------------------------------------------------------------------------*/
{
    // modifications by Nikola Pocuca
    const double ZTOL = (DBL_EPSILON*10.0); 
    double omega_in, alpha_in;     /* parameters of standard distribution */
    double res; // return result. 


    if (chi < ZTOL) { 
        /* special cases which are basically Gamma and Inverse Gamma distribution */
        if (lambda > 0.0) {
        res = R::rgamma(lambda, 2.0/psi);
        return res;  
        }
        else {
        res = 1.0/R::rgamma(-lambda, 2.0/psi);
        return res;  
        }    
    }

    else if (psi < ZTOL) {
        /* special cases which are basically Gamma and Inverse Gamma distribution */
        if (lambda > 0.0) {
            res = 1.0/R::rgamma(lambda, 2.0/chi); 
            return res; 
        }
        else {
            res = R::rgamma(-lambda, 2.0/chi);
            return res;  
        }    
    }

    /* BEING SPECIAL CASES */
    else {

        alpha_in = sqrt(chi/psi);
        omega_in = sqrt(psi*chi);

        /* run generator */
     
        if (lambda > 2. || omega_in > 3.) {
            /* Ratio-of-uniforms with shift by 'mode', alternative implementation */
            return _ratio_of_uniforms_shift(lambda,omega_in,alpha_in);
        }

        if (lambda >= 1.-2.25*omega_in*omega_in || omega_in > 0.2) {
            /* Ratio-of-uniforms without shift */
            return _ratio_of_uniforms_noshift(lambda,omega_in,alpha_in);
        }

        if (lambda >= 0. && omega_in > 0.) {
            /* New approach, constant hat in log-concave part. */
            return _leydold(lambda,omega_in,alpha_in);
        }
        
    }

    return -1.0; 
}






#pragma once 
double _leydold(double lambda, double omega, double alpha)
/*---------------------------------------------------------------------------*/
/* Original code by Josef Leydold and Wolfgang Hormann                       */
/* New approach, constant hat in log-concave part.                           */
/* Draw sample from GIG distribution.                                        */
/*                                                                           */
/* Case: 0 < lambda < 1, 0 < omega < 1                                       */
/*                                                                           */
/* Parameters:                                                               */
/*   n ....... sample size (positive integer)                                */
/*   lambda .. parameter for distribution                                    */
/*   omega ... parameter for distribution                                    */
/*                                                                           */
/*               Modified by Nikola Pocuca (2021)                            */
/*---------------------------------------------------------------------------*/
{
    /* parameters for hat function */
    std::array<double,3> A; 
    double lambda_old = lambda; 
    double res; 

    // Nikola Pocuca additions. 
    if(lambda < 0){
      lambda = -lambda; 
    }

    double Atot;  /* area below hat */
    double k0;          /* maximum of PDF */
    double k1, k2;      /* multiplicative constant */

    double xm;          /* location of mode */
    double x0;          /* splitting point T-concave / T-convex */
    double a;           /* auxiliary variable */

    double U, V, X;     /* random numbers */
    double hx;          /* hat at X */

    size_t count = 0;      /* counter for total number of iterations */

    /* -- Setup -------------------------------------------------------------- */

    /* mode = location of maximum of sqrt(f(x)) */
    xm = _gig_mode(lambda, omega);

    /* splitting point */
    x0 = omega/(1.-lambda);

    /* domain [0, x_0] */
    k0 = exp((lambda-1.)*log(xm) - 0.5*omega*(xm + 1./xm));     /* = f(xm) */
    A[0] = k0 * x0;

    /* domain [x_0, Infinity] */
    if (x0 >= 2./omega) {
        k1 = 0.;
        A[1] = 0.;
        k2 = pow(x0, lambda-1.);
        A[2] = k2 * 2. * exp(-omega*x0/2.)/omega;
    } else 
    {
        /* domain [x_0, 2/omega] */
        k1 = exp(-omega);
        A[1] = (lambda == 0.) 
        ? k1 * log(2./(omega*omega))
        : k1 / lambda * ( pow(2./omega, lambda) - pow(x0, lambda) );

        /* domain [2/omega, Infinity] */
        k2 = pow(2/omega, lambda-1.);
        A[2] = k2 * 2 * exp(-1.)/omega;
    }

    /* total area */
    Atot = A[0] + A[1] + A[2];

    /* -- Generate sample ---------------------------------------------------- */
    do {
        if(count == 100){ return -1.0; }
        /* get uniform random number */
        V = Atot * unif(generator_latent);

        do {

            /* domain [0, x_0] */
            if (V <= A[0]) {
                X = x0 * V / A[0];
                hx = k0;
                break;
            }

            /* domain [x_0, 2/omega] */
            V -= A[0];
            if (V <= A[1]) {

                if (lambda == 0.) {
                    X = omega * exp(exp(omega)*V);
                    hx = k1 / X;
                }
                else {
                    X = pow(pow(x0, lambda) + (lambda / k1 * V), 1./lambda);
                    hx = k1 * pow(X, lambda-1.);
                }

                break;
            }

            /* domain [max(x0,2/omega), Infinity] */
            V -= A[1];
            a = (x0 > 2./omega) ? x0 : 2./omega;
            X = -2./omega * log(exp(-omega/2. * a) - omega/(2.*k2) * V);
            hx = k2 * exp(-omega/2. * X);
            break;

        } while(false); // now I understand why they did it this way. 


        /* accept or reject */
        U = unif(generator_latent) * hx;

        if (log(U) <= (lambda-1.) * log(X) - omega/2. * (X+1./X)) {
            res = (lambda_old < 0.) ? (alpha / X) : (alpha * X);
            break;
        }

        count++; 
    } while(true);


  /* -- End ---------------------------------------------------------------- */

  return res;
}





#pragma once 
double _ratio_of_uniforms_shift(double lambda, double omega, double alpha)
/*---------------------------------------------------------------------------*/
/* Type 8:                                                                   */
/* Ratio-of-uniforms with shift by 'mode', alternative implementation.       */
/*   Dagpunar (1989)                                                         */
/*   Lehner (1989)                                                           */
/*   Modified by Nikola Pocuca (2021)                                        */
/*---------------------------------------------------------------------------*/
{
    double res;         /* result variable */
    double xm, nc;     /* location of mode; c=log(f(xm)) normalization constant */
    double s, t;       /* auxiliary variables */
    double U, V, X;    /* random variables */
    double lambda_old = lambda; /* old lambda calculation here instead of outside. */ 

    // Nikola Pocuca additions. 
    if(lambda < 0)
    { lambda = -lambda; }

    /* int i;              loop variable (number of generated random variables) */

    double a, b, c;    /* coefficent of cubic */
    double p, q;       /* coefficents of depressed cubic */
    double fi, fak;    /* auxiliary results for Cardano's rule */

    double y1, y2;     /* roots of (1/x)*sqrt(f((1/x)+m)) */

    double uplus, uminus;  /* maximum and minimum of x*sqrt(f(x+m)) */

    /* -- Setup -------------------------------------------------------------- */

    /* shortcuts */
    t = 0.5 * (lambda-1.);
    s = 0.25 * omega;

    /* mode = location of maximum of sqrt(f(x)) */
    xm = _gig_mode(lambda, omega);

    /* normalization constant: c = log(sqrt(f(xm))) */
    nc = t*log(xm) - s*(xm + 1./xm);

    /* location of minimum and maximum of (1/x)*sqrt(f(1/x+m)):  */

    /* compute coeffients of cubic equation y^3+a*y^2+b*y+c=0 */
    a = -(2.*(lambda+1.)/omega + xm);       /* < 0 */
    b = (2.*(lambda-1.)*xm/omega - 1.);
    c = xm;

    /* we need the roots in (0,xm) and (xm,inf) */
    /* substitute y=z-a/3 for depressed cubic equation z^3+p*z+q=0 */
    p = b - a*a/3.;
    q = (2.*a*a*a)/27. - (a*b)/3. + c;

    
    /* use Cardano's rule */
    fi = acos(-q/(2.*sqrt(-(p*p*p)/27.)));
    fak = 2.*sqrt(-p/3.);
    y1 = fak * cos(fi/3.) - a/3.;
    y2 = fak * cos(fi/3. + 4./3.*M_PI) - a/3.;

    /* boundaries of minmal bounding rectangle:                  */
    /* we us the "normalized" density f(x) / f(xm). hence        */
    /* upper boundary: vmax = 1.                                 */
    /* left hand boundary: uminus = (y2-xm) * sqrt(f(y2)) / sqrt(f(xm)) */
    /* right hand boundary: uplus = (y1-xm) * sqrt(f(y1)) / sqrt(f(xm)) */
    uplus  = (y1-xm) * exp(t*log(y1) - s*(y1 + 1./y1) - nc);
    uminus = (y2-xm) * exp(t*log(y2) - s*(y2 + 1./y2) - nc);

    /* -- Generate sample ---------------------------------------------------- */

    size_t count = 0; 
    do {
        if(count == 100)
            return -1.0; 

        U = uminus + unif(generator_latent) * (uplus - uminus);    /* U(u-,u+)  */
        V = unif(generator_latent);                                /* U(0,vmax) */
        X = U/V + xm;
        ++count;
    }                                         /* Acceptance/Rejection */
    while ((X <= 0.) || ((log(V)) > (t*log(X) - s*(X + 1./X) - nc)));

    /* store random point */
    res = (lambda_old < 0.) ? (alpha / X) : (alpha * X);

    /* -- End ---------------------------------------------------------------- */

    return res; 
}




#pragma once
double _ratio_of_uniforms_noshift(double lambda, double omega, double alpha)
/*---------------------------------------------------------------------------*/
/* Ratio-of-uniforms without shift.                                          */
/*   Dagpunar (1988), Sect.~4.6.2                                            */
/*   Lehner (1989)                                                           */
/*   Modified by Nikola Pocuca (2021)                                        */
/*---------------------------------------------------------------------------*/
{
    double res = 0.0; // instatiate result. 
    double xm, nc;     /* location of mode; c=log(f(xm)) normalization constant */
    double ym, um;     /* location of maximum of x*sqrt(f(x)); umax of MBR */
    double s, t;       /* auxiliary variables */
    double U, V, X;    /* random variables */
    double lambda_old = lambda; 

    // Nikola Pocuca additions. 
    if(lambda < 0){
      lambda = -lambda; 
    }

    /* -- Setup -------------------------------------------------------------- */

    /* shortcuts */
    t = 0.5 * (lambda-1.);
    s = 0.25 * omega;

    /* mode = location of maximum of sqrt(f(x)) */
    xm = _gig_mode(lambda, omega);

    /* normalization constant: c = log(sqrt(f(xm))) */
    nc = t*log(xm) - s*(xm + 1./xm);

    /* location of maximum of x*sqrt(f(x)):           */
    /* we need the positive root of                   */
    /*    omega/2*y^2 - (lambda+1)*y - omega/2 = 0    */
    ym = ((lambda+1.) + sqrt((lambda+1.)*(lambda+1.) + omega*omega))/omega;


    /* boundaries of minmal bounding rectangle:                   */
    /* we us the "normalized" density f(x) / f(xm). hence         */
    /* upper boundary: vmax = 1.                                  */
    /* left hand boundary: umin = 0.                              */
    /* right hand boundary: umax = ym * sqrt(f(ym)) / sqrt(f(xm)) */
    um = exp(0.5*(lambda+1.)*log(ym) - s*(ym + 1./ym) - nc);

    size_t i = 0; 
    /* -- Generate sample ---------------------------------------------------- */
    do
    {
        if(i == 100){
          return -1.0; 
        }
        U = um * unif(generator_latent);      /* U(0,umax) */
        V = unif(generator_latent);             /* U(0,vmax) */
        X = U/V;
        i++; 
        /* code */
    } while (((log(V)) > (t*log(X) - s*(X + 1./X) - nc)));
    

    /* store random point */
    res = (lambda_old < 0.) ? (alpha / X) : (alpha * X);
    
    /* -- End ---------------------------------------------------------------- */

    return res; 
}

#pragma once 
double _gig_mode(double lambda, double omega)
/*---------------------------------------------------------------------------*/
/* Compute mode of GIG distribution.                                         */
/*                                                                           */
/* Parameters:                                                               */
/*   lambda .. parameter for distribution                                    */
/*   omega ... parameter for distribution                                    */
/*                                                                           */
/* Return:                                                                   */
/*   mode                                                                    */
/*---------------------------------------------------------------------------*/
{
  if (lambda >= 1.)
    /* mode of fgig(x) */
    return (sqrt((lambda-1.)*(lambda-1.) + omega*omega)+(lambda-1.))/omega;
  else
    /* 0 <= lambda < 1: use mode of f(1/x) */
    return omega / (sqrt((1.-lambda)*(1.-lambda) + omega*omega)+(1.-lambda));
} /* end of _gig_mode() */






