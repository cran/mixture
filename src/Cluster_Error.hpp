#include <exception> 
#include <string> 

// exception thrower.
#pragma once
struct bad_sym_except : std::exception
{
  const char* what() const noexcept {return "Error is thrown\n";}

  std::string error_message = "Default error message"; // replace message as needed. 
};

struct loglik_decreasing : std::exception
{
    const char* what() const noexcept {return "logliklihood is decreasing";}
};

struct infinite_loglik_except : std::exception 
{
  const char* what() const noexcept {return "logliklihood is infinite";}
}; 

struct below_1_ng_except : std::exception 
{
  const char* what() const noexcept {return "one of the N_gs dropped below 1";}
};


struct sym_matrix_error : std::exception 
{
  const char* what() const noexcept {return "matrix determined not to be spd";}
};
