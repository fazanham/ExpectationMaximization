#pragma once

/*
* GaussianMixture.h
* Helper class for computing the Gauss Params
* Created on: September, 2017
* Author: Faizaan Naveed
*/

#ifndef GAUSSIANMIXTURE_H
#define GAUSSIANMIXTURE_H

#include "LH.h"

class GaussianMixture {

public:

	GaussianMixture(int Clusters);
	~GaussianMixture();

	// Methods

	/* COMPUTE NORMAL DISTRIBUTION
	@param CovarianceMat Covariance matrix for each cluster
	@param MeanVec Mean vector for each cluster
	@param X Input vector (the length of the vector is the number of channels)
	*/
	long double ComputeND(const Eigen::Matrix3d &CovarianceMat, const Eigen::Vector3d &MeanVec, const Eigen::Vector3d &X);

	/* COMPUTE INITIAL ESTIMATE OF GAUSS PARAMS USING K-MEANS
	@param Iterations The number of iterations till convergence for k-means
	@param Epsilon The minimum SSE for convegence 
	*/
	cv::Mat ComputeInitialGaussParams(size_t Iterations, double Epsilon, std::vector<cv::Mat_<float>> Input_Image);

	/* UPDATE THE GAUSSIAN PARAMETERS
	@param ClassProbabilities Initial class probabilities to be used for updating the Gaussian parameters - E-step
	*/	
	void UpdateGaussParams(Eigen::VectorXd *ClassProbabilities, std::vector<cv::Mat> Input_Image);

	// Attributes

	struct Gauss_Params {
		Eigen::Vector3d Mean; // Mean of th clusters
		Eigen::Matrix3d Covariance; // Covariance of the clusters
	};

	Gauss_Params *GMM;
	int Clusters; // Total number of clusters

};

#endif