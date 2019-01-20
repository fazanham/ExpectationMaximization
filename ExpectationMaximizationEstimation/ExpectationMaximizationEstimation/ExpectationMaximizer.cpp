#include "ExpectationMaximizer.h"

ExpectationMaximizer::ExpectationMaximizer(cv::Mat Input_Data, int clusters, size_t Iterations, double Epsilon) : Clusters(clusters)
{
	cv::split(Input_Data, ExpectationMaximizer::Input_Data);
	GaussiansM = new GaussianMixture(Clusters);
	Weight_Mixtures = new double[Clusters];

	ExpectationMaximizer::Initial_Clusters = GaussiansM->ComputeInitialGaussParams(Iterations, Epsilon, ExpectationMaximizer::Input_Data);
}

Eigen::VectorXd * ExpectationMaximizer::UpdateClassProb(Eigen::VectorXd * ClassProbabilities)
{
	Eigen::VectorXd *Prob_UP = new Eigen::VectorXd[Clusters];
	Eigen::VectorXd Prob_SUM(ClassProbabilities->rows());

	for (size_t i = 0; i < ClassProbabilities->rows(); i++) {
		Prob_SUM[i] = 0.;
		for (size_t j = 0; j < Clusters; j++) {
			Prob_SUM[i] += (Weight_Mixtures[j]*ClassProbabilities[j][i]);
		}
	}

	for (size_t j = 0; j < Clusters; j++) {
		Prob_UP[j] = Eigen::VectorXd::Zero(ClassProbabilities->rows());
		for (size_t i = 0; i < ClassProbabilities->rows(); i++) {
			/*Compute the weighted probability*/
			Prob_UP[j][i] = (Weight_Mixtures[j]*ClassProbabilities[j][i]) / Prob_SUM[i];
		}
	}

	return Prob_UP;
}

Eigen::VectorXd * ExpectationMaximizer::ComputeClassProb()
{
	int Samples = Input_Data[0].rows*Input_Data[0].cols;
	Eigen::VectorXd *Prob = new Eigen::VectorXd[ExpectationMaximizer::Clusters];

	for (size_t j = 0; j < Clusters; j++) {
		Prob[j] = Eigen::VectorXd::Zero(Samples);
		for (size_t i = 0; i < Samples; i++) {
			Eigen::VectorXd X(chnls);
			for (size_t c = 0; c < chnls; c++) {
				X[c] = Input_Data[c].at<float>(i / Input_Data[c].cols, i % Input_Data[c].cols);
			}
			Prob[j][i] = GaussiansM->ComputeND(GaussiansM->GMM[j].Covariance, GaussiansM->GMM[j].Mean, X);
		}
	}

	return Prob;
}

void ExpectationMaximizer::ComputeInitialWM(cv::Mat Initial_Clusters)
{

	for (size_t i = 0; i < ExpectationMaximizer::Clusters; i++) 
		ExpectationMaximizer::Weight_Mixtures[i] = 0.;

	for (size_t j = 0; j < Initial_Clusters.rows; j++) 
		ExpectationMaximizer::Weight_Mixtures[Initial_Clusters.at<int>(j, 0)]++;
	
	for (size_t i = 0; i < ExpectationMaximizer::Clusters; i++) {
		ExpectationMaximizer::Weight_Mixtures[i] = ExpectationMaximizer::Weight_Mixtures[i] / Initial_Clusters.rows;
	}
	
}

void ExpectationMaximizer::UpdateWM(Eigen::VectorXd * ClassProbabilities)
{
	for (size_t i = 0; i < ExpectationMaximizer::Clusters; i++) {
		ExpectationMaximizer::Weight_Mixtures[i] = 0.;
		for (size_t j = 0; j < ClassProbabilities->rows(); j++) {
			ExpectationMaximizer::Weight_Mixtures[i] += ClassProbabilities[i][j];
		}
	}
	for (size_t i = 0; i < ExpectationMaximizer::Clusters; i++) {
		ExpectationMaximizer::Weight_Mixtures[i] = ExpectationMaximizer::Weight_Mixtures[i] / ClassProbabilities->rows();
	}
}

double ExpectationMaximizer::ComputeLogLikelihood(const Eigen::VectorXd * ClassProbabilities)
{
	double LOGLK = 0.;
	for (size_t i = 0; i < ClassProbabilities->rows(); i++) {
		double LOGLH = 0.;
		for (size_t k = 0; k < ExpectationMaximizer::Clusters; k++) {
			LOGLH += (ExpectationMaximizer::Weight_Mixtures[k]*ClassProbabilities[k](i));
		}
		LOGLK += log10(LOGLH);
	}

	return -LOGLK;
}

cv::Mat ExpectationMaximizer::posteriorClassification()
{
	cv::Mat Classified(Input_Data[0].rows, Input_Data[0].cols, CV_8U, cv::Scalar(0));
	Eigen::VectorXd Prob(ExpectationMaximizer::Clusters);
	cv::Mat labels_EM(Input_Data[0].rows*Input_Data[0].cols, 1, CV_8U);

	int *colors = new int[ExpectationMaximizer::Clusters];
	for (int i = 0; i<ExpectationMaximizer::Clusters; i++) {
		colors[i] = 255 / (i + 1);
	}

	for (size_t i = 0; i < Input_Data[0].rows*Input_Data[0].cols; i++) {

		Eigen::Vector3d X(chnls);
		for (size_t c = 0; c < chnls; c++) 
			X[c] = Input_Data[c].at<float>(i / Input_Data[0].cols, i % Input_Data[0].cols);
		
		for (size_t j = 0; j < ExpectationMaximizer::Clusters; j++) 
			Prob[j] = GaussiansM->ComputeND(GaussiansM->GMM[j].Covariance, GaussiansM->GMM[j].Mean, X);
		
		int max_id;

		/*Assign the point to the cluster with max probability*/
		double Prob_Max = Prob.maxCoeff(&max_id);
		Classified.at<uint8_t>(i / Input_Data[0].cols, i % Input_Data[0].cols) = colors[max_id];
		labels_EM.at<uint8_t>(i, 0) = max_id;
	}

	return Classified;
}

