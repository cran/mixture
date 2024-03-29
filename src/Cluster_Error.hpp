#include <exception> 
#include <string> 

#pragma once 
bool is_string_comparison(std::exception e_ , std::string message_) 
{
  bool check = (0 == std::string(e_.what()).compare(message_)); // compare if string and message match. 
  return check;
}

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

struct infinite_loglik_with_return_except : std::exception
{
  const char* what() const noexcept {return "logliklihood was infinite, back to previous step and returned results";}
  
};


struct below_1_ng_except : std::exception 
{
  const char* what() const noexcept {return "one of the N_gs dropped below 1";}
};


struct sym_matrix_error : std::exception 
{
  const char* what() const noexcept {return "matrix determined not to be spd";}
};
