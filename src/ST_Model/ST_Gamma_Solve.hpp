
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/digamma.hpp> 
#include <boost/math/tools/roots.hpp>

using boost::math::policies::policy;
using boost::math::tools::halley_iterate; //
using boost::math::tools::eps_tolerance; // Binary functor for specified number of bits.

// macros 
#define digam(x) boost::math::digamma(x)
#define trigam(x) boost::math::trigamma(x)
#define ngam(n,x) boost::math::polygamma(n,x)

#pragma once
template <class T>
struct gamma_solve_functor
{
  // Functor returning both 1st and 2nd derivatives.
  gamma_solve_functor(T const& to_find_root_of) : eta(to_find_root_of)
  { // Constructor stores value a to find root of, for example:
    // calling cbrt_functor_2deriv<T>(x) to get cube root of x,
  }
  std::tuple<T, T, T> operator()(T const& x)
  {
    // Return both f(x) and f'(x) and f''(x).
    T fx = digam(x*0.5) - log(x*0.5) + eta;                   
    T dx = 0.5*trigam(x*0.5) - 1/x;                     
    T d2x = 0.25*ngam(2,x*0.5) + 1/(x*x);                        
    return std::make_tuple(fx, dx, d2x);  // 'return' fx, dx and d2x.
  }
private:
  T eta; // it is essentially -cbar + abar - 1 but I call this the 
};

#pragma once
template <class T>
T gamma_solve(T x,T guess, T min_in)
{
  // return gamma solve using 1st and 2nd derivatives and Halley.
  //using namespace std;  // Help ADL of std functions.
  using namespace boost::math::tools;
  
  T min = min_in;                     // Minimum possible value is half our guess.
  T max = 20;                      
  const int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  // digits used to control how accurate to try to make the result.
  int get_digits = static_cast<int>(digits * 0.4);    // Accuracy triples with each step, so stop when just
                                                      // over one third of the digits are correct.
  boost::uintmax_t maxit = 40;
  T result = halley_iterate(gamma_solve_functor<T>(x), guess, min, max, get_digits, maxit);
  return result;
}
