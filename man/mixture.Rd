\name{mixture}
\alias{mixture}
\docType{package}
\title{Mixture Models for Clustering and Classification}
\description{
An implementation of 14 parsimonious clustering models for model-based clustering and model-based classification, even with missing data. Four different mixture distributions are avaialble, i.e., component densities may be Gaussian, generalized hyperbolic, variance-gamma, or skew-t. 
}
\details{
\tabular{ll}{
Package: \tab mixture \cr
Type: \tab Package\cr
Version: \tab 2.0.1\cr
Date: \tab 2021-01-15\cr
License: \tab GPL (>=2)\cr
}
This package contains the functions \code{gpcm}, \code{ghpcm}, \code{vgpcm}, \code{stpcm}, \code{e_step}, \code{ARI}, and \code{get_best_model}, plus three simulated data sets.

This package also contains advanced functions for large system use which are:
 \code{main_loop} \code{main_loop_vg} , \code{main_loop_gh}, , \code{main_loop_st}, ,\code{z_ig_random_soft}, \code{z_ig_random_hard}, \code{z_ig_kmeans}.
}
\author{
Nik Pocuca, Ryan P. Browne, and Paul D. McNicholas.

Maintainer: Paul D. McNicholas <mcnicholas@math.mcmaster.ca>
}
\seealso{
Details, examples, and references are given under \code{\link{gpcm}},  \code{\link{ghpcm}},  \code{\link{stpcm}}, and  \code{\link{vgpcm}}.
}
