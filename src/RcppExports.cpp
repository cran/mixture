// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <Rcpp.h>

using namespace Rcpp;

// main_loop_gh
Rcpp::List main_loop_gh(arma::mat X, int G, int model_id, int model_type, arma::mat in_zigs, int in_nmax, double in_l_tol, int in_m_iter_max, double in_m_tol, arma::vec anneals, int t_burn);
RcppExport SEXP _mixture_main_loop_gh(SEXP XSEXP, SEXP GSEXP, SEXP model_idSEXP, SEXP model_typeSEXP, SEXP in_zigsSEXP, SEXP in_nmaxSEXP, SEXP in_l_tolSEXP, SEXP in_m_iter_maxSEXP, SEXP in_m_tolSEXP, SEXP annealsSEXP, SEXP t_burnSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type model_id(model_idSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type in_zigs(in_zigsSEXP);
    Rcpp::traits::input_parameter< int >::type in_nmax(in_nmaxSEXP);
    Rcpp::traits::input_parameter< double >::type in_l_tol(in_l_tolSEXP);
    Rcpp::traits::input_parameter< int >::type in_m_iter_max(in_m_iter_maxSEXP);
    Rcpp::traits::input_parameter< double >::type in_m_tol(in_m_tolSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type anneals(annealsSEXP);
    Rcpp::traits::input_parameter< int >::type t_burn(t_burnSEXP);
    rcpp_result_gen = Rcpp::wrap(main_loop_gh(X, G, model_id, model_type, in_zigs, in_nmax, in_l_tol, in_m_iter_max, in_m_tol, anneals, t_burn));
    return rcpp_result_gen;
END_RCPP
}
// gh_e_step_internal
Rcpp::List gh_e_step_internal(arma::mat X, int G, int model_id, int model_type, Rcpp::List in_m_obj, arma::mat init_zigs, double in_nu);
RcppExport SEXP _mixture_gh_e_step_internal(SEXP XSEXP, SEXP GSEXP, SEXP model_idSEXP, SEXP model_typeSEXP, SEXP in_m_objSEXP, SEXP init_zigsSEXP, SEXP in_nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type model_id(model_idSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type in_m_obj(in_m_objSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type init_zigs(init_zigsSEXP);
    Rcpp::traits::input_parameter< double >::type in_nu(in_nuSEXP);
    rcpp_result_gen = Rcpp::wrap(gh_e_step_internal(X, G, model_id, model_type, in_m_obj, init_zigs, in_nu));
    return rcpp_result_gen;
END_RCPP
}
// main_loop
Rcpp::List main_loop(arma::mat X, int G, int model_id, int model_type, arma::mat in_zigs, int in_nmax, double in_l_tol, int in_m_iter_max, double in_m_tol, arma::vec anneals, int t_burn);
RcppExport SEXP _mixture_main_loop(SEXP XSEXP, SEXP GSEXP, SEXP model_idSEXP, SEXP model_typeSEXP, SEXP in_zigsSEXP, SEXP in_nmaxSEXP, SEXP in_l_tolSEXP, SEXP in_m_iter_maxSEXP, SEXP in_m_tolSEXP, SEXP annealsSEXP, SEXP t_burnSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type model_id(model_idSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type in_zigs(in_zigsSEXP);
    Rcpp::traits::input_parameter< int >::type in_nmax(in_nmaxSEXP);
    Rcpp::traits::input_parameter< double >::type in_l_tol(in_l_tolSEXP);
    Rcpp::traits::input_parameter< int >::type in_m_iter_max(in_m_iter_maxSEXP);
    Rcpp::traits::input_parameter< double >::type in_m_tol(in_m_tolSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type anneals(annealsSEXP);
    Rcpp::traits::input_parameter< int >::type t_burn(t_burnSEXP);
    rcpp_result_gen = Rcpp::wrap(main_loop(X, G, model_id, model_type, in_zigs, in_nmax, in_l_tol, in_m_iter_max, in_m_tol, anneals, t_burn));
    return rcpp_result_gen;
END_RCPP
}
// e_step_internal
Rcpp::List e_step_internal(arma::mat X, int G, int model_id, int model_type, Rcpp::List in_m_obj, arma::mat init_zigs, double in_nu);
RcppExport SEXP _mixture_e_step_internal(SEXP XSEXP, SEXP GSEXP, SEXP model_idSEXP, SEXP model_typeSEXP, SEXP in_m_objSEXP, SEXP init_zigsSEXP, SEXP in_nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type model_id(model_idSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type in_m_obj(in_m_objSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type init_zigs(init_zigsSEXP);
    Rcpp::traits::input_parameter< double >::type in_nu(in_nuSEXP);
    rcpp_result_gen = Rcpp::wrap(e_step_internal(X, G, model_id, model_type, in_m_obj, init_zigs, in_nu));
    return rcpp_result_gen;
END_RCPP
}
// main_loop_st
Rcpp::List main_loop_st(arma::mat X, int G, int model_id, int model_type, arma::mat in_zigs, int in_nmax, double in_l_tol, int in_m_iter_max, double in_m_tol, arma::vec anneals, int t_burn);
RcppExport SEXP _mixture_main_loop_st(SEXP XSEXP, SEXP GSEXP, SEXP model_idSEXP, SEXP model_typeSEXP, SEXP in_zigsSEXP, SEXP in_nmaxSEXP, SEXP in_l_tolSEXP, SEXP in_m_iter_maxSEXP, SEXP in_m_tolSEXP, SEXP annealsSEXP, SEXP t_burnSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type model_id(model_idSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type in_zigs(in_zigsSEXP);
    Rcpp::traits::input_parameter< int >::type in_nmax(in_nmaxSEXP);
    Rcpp::traits::input_parameter< double >::type in_l_tol(in_l_tolSEXP);
    Rcpp::traits::input_parameter< int >::type in_m_iter_max(in_m_iter_maxSEXP);
    Rcpp::traits::input_parameter< double >::type in_m_tol(in_m_tolSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type anneals(annealsSEXP);
    Rcpp::traits::input_parameter< int >::type t_burn(t_burnSEXP);
    rcpp_result_gen = Rcpp::wrap(main_loop_st(X, G, model_id, model_type, in_zigs, in_nmax, in_l_tol, in_m_iter_max, in_m_tol, anneals, t_burn));
    return rcpp_result_gen;
END_RCPP
}
// st_e_step_internal
Rcpp::List st_e_step_internal(arma::mat X, int G, int model_id, int model_type, Rcpp::List in_m_obj, arma::mat init_zigs, double in_nu);
RcppExport SEXP _mixture_st_e_step_internal(SEXP XSEXP, SEXP GSEXP, SEXP model_idSEXP, SEXP model_typeSEXP, SEXP in_m_objSEXP, SEXP init_zigsSEXP, SEXP in_nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type model_id(model_idSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type in_m_obj(in_m_objSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type init_zigs(init_zigsSEXP);
    Rcpp::traits::input_parameter< double >::type in_nu(in_nuSEXP);
    rcpp_result_gen = Rcpp::wrap(st_e_step_internal(X, G, model_id, model_type, in_m_obj, init_zigs, in_nu));
    return rcpp_result_gen;
END_RCPP
}
// main_loop_vg
Rcpp::List main_loop_vg(arma::mat X, int G, int model_id, int model_type, arma::mat in_zigs, int in_nmax, double in_l_tol, int in_m_iter_max, double in_m_tol, arma::vec anneals, int t_burn);
RcppExport SEXP _mixture_main_loop_vg(SEXP XSEXP, SEXP GSEXP, SEXP model_idSEXP, SEXP model_typeSEXP, SEXP in_zigsSEXP, SEXP in_nmaxSEXP, SEXP in_l_tolSEXP, SEXP in_m_iter_maxSEXP, SEXP in_m_tolSEXP, SEXP annealsSEXP, SEXP t_burnSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type model_id(model_idSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type in_zigs(in_zigsSEXP);
    Rcpp::traits::input_parameter< int >::type in_nmax(in_nmaxSEXP);
    Rcpp::traits::input_parameter< double >::type in_l_tol(in_l_tolSEXP);
    Rcpp::traits::input_parameter< int >::type in_m_iter_max(in_m_iter_maxSEXP);
    Rcpp::traits::input_parameter< double >::type in_m_tol(in_m_tolSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type anneals(annealsSEXP);
    Rcpp::traits::input_parameter< int >::type t_burn(t_burnSEXP);
    rcpp_result_gen = Rcpp::wrap(main_loop_vg(X, G, model_id, model_type, in_zigs, in_nmax, in_l_tol, in_m_iter_max, in_m_tol, anneals, t_burn));
    return rcpp_result_gen;
END_RCPP
}
// vg_e_step_internal
Rcpp::List vg_e_step_internal(arma::mat X, int G, int model_id, int model_type, Rcpp::List in_m_obj, arma::mat init_zigs, double in_nu);
RcppExport SEXP _mixture_vg_e_step_internal(SEXP XSEXP, SEXP GSEXP, SEXP model_idSEXP, SEXP model_typeSEXP, SEXP in_m_objSEXP, SEXP init_zigsSEXP, SEXP in_nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type model_id(model_idSEXP);
    Rcpp::traits::input_parameter< int >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type in_m_obj(in_m_objSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type init_zigs(init_zigsSEXP);
    Rcpp::traits::input_parameter< double >::type in_nu(in_nuSEXP);
    rcpp_result_gen = Rcpp::wrap(vg_e_step_internal(X, G, model_id, model_type, in_m_obj, init_zigs, in_nu));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mixture_main_loop_gh", (DL_FUNC) &_mixture_main_loop_gh, 11},
    {"_mixture_gh_e_step_internal", (DL_FUNC) &_mixture_gh_e_step_internal, 7},
    {"_mixture_main_loop", (DL_FUNC) &_mixture_main_loop, 11},
    {"_mixture_e_step_internal", (DL_FUNC) &_mixture_e_step_internal, 7},
    {"_mixture_main_loop_st", (DL_FUNC) &_mixture_main_loop_st, 11},
    {"_mixture_st_e_step_internal", (DL_FUNC) &_mixture_st_e_step_internal, 7},
    {"_mixture_main_loop_vg", (DL_FUNC) &_mixture_main_loop_vg, 11},
    {"_mixture_vg_e_step_internal", (DL_FUNC) &_mixture_vg_e_step_internal, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_mixture(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
