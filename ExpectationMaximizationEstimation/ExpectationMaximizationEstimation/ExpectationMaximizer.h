#pragma once

/*
* EM.h
* Expectation Maximization classification for any number of channels
* Created on: September, 2017
* Author: Faizaan Naveed
*/

#ifndef EXPECTATIONMAXIMIZER_H
#define EXPECTATIONMAXIMIZER_H

#include "GaussianMixture.h"

class ExpectationMaximizer {

public:
	ExpectationMaximizer(cv::Mat Input_Data, int clusters, size_t Iterations, double Epsilon);
	~ExpectationMaximizer();

	// Methods

	/* COMPUTE THE INITIAL CLASS PROBABILITIES*/
	Eigen::VectorXd * ComputeClassProb();

	/* UPDATE THE CLASS PROBABILITIES
	@param ClassProbabilities Update the class probabilities based on the Gauss params - M-step
	*/
	Eigen::VectorXd * UpdateClassProb(Eigen::VectorXd *ClassProbabilities);

	/* COMPUTE THE INITIAL WEIGHT MIXTURES
	@param Initial_Clusters Compute the initial weight mixtures using initial clusters from k-means
	*/
	void ComputeInitialWM(cv::Mat Initial_Clusters); 

	/* UPDATE THE WEIGHT MIXTURES
	@param ClassProbabilities Update the weight mixtures using the class probabilities
	*/
	void UpdateWM(Eigen::VectorXd *ClassProbabilities);

	/* COMPUTE THE LOG-LIKELIHOOD
	@param ClassProbabilities Compute the log-likelihood to check for convergence
	*/
	double ComputeLogLikelihood(const Eigen::VectorXd *ClassProbabilities);

	/* COMPUTES THE POSTERIOR CLASSIFICATION BASED ON GAUSS PARAMS*/
	cv::Mat posteriorClassification(); 

	// Attributes

	std::vector<cv::Mat_<float>> Input_Data; // Input Image
	double * Weight_Mixtures; // Weight mixtures for clusters
	int Clusters; // Total number of clusters
	GaussianMixture *GaussiansM; // Gaussian Mixture object
	cv::Mat Initial_Clusters; // Initial clusters determined from k-means


};

#endif