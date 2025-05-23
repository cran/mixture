\name{mixture}
\alias{mixture}
\alias{mixture-package}
\docType{package}
\title{Mixture Models for Clustering and Classification}
\description{
An implementation of 14 parsimonious clustering models for finite mixtures with components that are Gaussian, generalized hyperbolic, variance-gamma, Student's t, or skew-t, for model-based clustering and model-based classification, even with missing data. 
}
\details{
\tabular{ll}{
Package: \tab mixture \cr
Type: \tab Package\cr
Version: \tab 2.1.2\cr
Date: \tab 2025-05-06\cr
License: \tab GPL (>=2)\cr
}
This package contains the functions \code{gpcm}, \code{tpcm}, \code{ghpcm}, \code{vgpcm}, \code{stpcm}, \code{e_step}, \code{ARI}, and \code{get_best_model}, plus three simulated data sets.

This package also contains advanced functions for large system use which are:
 \code{main_loop} \code{main_loop_vg} , \code{main_loop_gh}, \code{main_loop_t} , \code{main_loop_st} ,\code{z_ig_random_soft}, \code{z_ig_random_hard}, \code{z_ig_kmeans}.
}
\author{
Nik Pocuca, Ryan P. Browne, Paul D. McNicholas, and Alexa A. Sochaniwsky.

Maintainer: Paul D. McNicholas <mcnicholas@math.mcmaster.ca>
}
\seealso{
Details, examples, and references are given under \code{\link{gpcm}}, \code{\link{tpcm}}, \code{\link{ghpcm}},  \code{\link{stpcm}}, and  \code{\link{vgpcm}}.
}
