#include "ExpectationMaximizer.h"

cv::Mat convertImage28U(cv::Mat image);


int main() {

	cv::Mat IMg = cv::imread("..\\fruits.jpg");
	cv::GaussianBlur(IMg, IMg, cv::Size(5, 5), 1.525);

	IMg.convertTo(IMg, CV_32FC3);

	std::vector<cv::Mat> split_IMG;
	split(IMg, split_IMG);
	cv::Mat Input_Array[chnls];

	for (int i = 0; i < chnls; i++) {
		Input_Array[i] = split_IMG[i];
	}

	cv::imshow("Original Image", convertImage28U(IMg));

	/*Specify the clusters and parameters for initial clusters*/
	ExpectationMaximizer *EM_Obj = new ExpectationMaximizer(IMg, 8, 100, 0.1);

	/*Initial Clusters*/
	cv::Mat out = EM_Obj->posteriorClassification();
	cv::imshow("Initial Clusters", out);
	
	/*Compute the initial weights*/
	EM_Obj->ComputeInitialWM(EM_Obj->Initial_Clusters);

	/*First Iteration - LogLikelihood*/
	Eigen::VectorXd *INI_PROB; // Initial probability
	double LOGLK_INI = 0.;
	double THR_INI = std::numeric_limits<double>::max();
	std::vector<double> likelihooddata;

	double Converge_THR = 0.001; // Converge threshold

	while (abs(THR_INI) > Converge_THR) {

		printf("Log_Likelihood_Difference: %f\n", THR_INI);

		/*iterative E-step*/
		INI_PROB = EM_Obj->UpdateClassProb(EM_Obj->ComputeClassProb());

		/*iterative M-step*/
		EM_Obj->GaussiansM->UpdateGaussParams(INI_PROB, split_IMG);
		EM_Obj->UpdateWM(INI_PROB);

		/*Compute the Log-likelihood difference*/
		THR_INI = (EM_Obj->ComputeLogLikelihood(INI_PROB) - LOGLK_INI);
		LOGLK_INI = EM_Obj->ComputeLogLikelihood(INI_PROB);
		likelihooddata.push_back(LOGLK_INI);
	}

	/*Posterior classification results*/
	cv::Mat classified_EM = EM_Obj->posteriorClassification();

	imshow("Classified_EM", classified_EM);
	cv::waitKey(0);

	return 0;

}

/*Scales and converts the image type to 8bit unsigned int*/
cv::Mat convertImage28U(cv::Mat image) 
{
	double min, max;
	cv::minMaxIdx(image, &min, &max);

	cv::Mat out;
	image.convertTo(out, CV_8U, 255.0 / (max - min), -255.0*min / (max - min));
	return out;
}