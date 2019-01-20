#include "GaussianMixture.h"

GaussianMixture::GaussianMixture(int Clusters) : Clusters(Clusters)
{
	// Resize the Gauss params to the no. of clusters
	GMM = new Gauss_Params[Clusters];
}

long double GaussianMixture::ComputeND(const Eigen::Matrix3d & CovarianceMat, const Eigen::Vector3d & MeanVec, const Eigen::Vector3d & X)
{
	// Compute the Normal distribution for a given input point
	const long double logSqrt2Pi = 0.5*std::log(2 * pi);
	typedef Eigen::LLT<Eigen::Matrix3d> Chol;
	Chol chol(CovarianceMat);

	// Handle non positive definite covariance matrix
	if (chol.info() != Eigen::Success) throw "decomposition failed!";

	const Chol::Traits::MatrixL& L = chol.matrixL();
	long double quadform = (L.solve(X - MeanVec)).squaredNorm();
	return std::exp(-X.rows()*logSqrt2Pi - 0.5*quadform) / L.determinant();
}

cv::Mat GaussianMixture::ComputeInitialGaussParams(size_t Iterations, double Epsilon, std::vector<cv::Mat_<float>> Input_Image)
{
	int Samples = Input_Image[0].rows*Input_Image[0].cols; // Total number of samples

	cv::Mat p = cv::Mat::zeros(Samples, chnls, CV_32F);
	cv::Mat centers, clustered;

	for (int i = 0; i < Samples; i++) {
		for (size_t j = 0; j < chnls; j++) {
			p.at<float>(i, j) = Input_Image[j].at<float>(i / Input_Image[j].cols, i % Input_Image[j].cols);
		}
	}

	/*Initialize the Mean and Covariance matrices*/
	int *mean_count = new int[Clusters];
	for (size_t i = 0; i < Clusters; i++) {
		mean_count[i] = 0;
		for (size_t j = 0; j < GMM[i].Covariance.rows()*GMM[i].Covariance.cols(); j++)
			GMM[i].Covariance(j / GMM[i].Covariance.cols(), j % GMM[i].Covariance.cols()) = 0.;
	}

	/*Kmeans with PP centre initialization - OpenCV implementation. See:
	https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
	*/
	cv::Mat Initial_labels;
	kmeans(p, Clusters, Initial_labels, cv::TermCriteria(CV_TERMCRIT_EPS, Iterations, Epsilon), 3, cv::KMEANS_PP_CENTERS, centers);

	/*Compute the mean and covariance using the clusters from k-means*/
	for (size_t i = 0; i < GaussianMixture::Clusters; i++) {
		for (size_t j = 0; j < chnls; j++) {
			GMM[i].Mean[j] = centers.at<float>(i, j);
		}
	}

	for (size_t i = 0; i < GaussianMixture::Clusters; i++) {
		for (size_t j = 0; j < Samples; j++) {
			int row_ = j / Input_Image[0].cols;
			int col_ = j % Input_Image[0].cols;
			Eigen::Vector3d XX(Input_Image[0].at<float>(row_, col_), Input_Image[1].at<float>(row_, col_), Input_Image[2].at<float>(row_, col_));
			GMM[i].Covariance += ((XX - GMM[i].Mean)*((XX - GMM[i].Mean).transpose()));
		}
	}
	for (size_t i = 0; i < GaussianMixture::Clusters; i++) {
		GMM[i].Covariance = GMM[i].Covariance / (Initial_labels.rows - 1);

		for (size_t j = 0; j < GMM[i].Covariance.rows()*GMM[i].Covariance.cols(); j++)
			if (j / GMM[i].Covariance.cols() != j % GMM[i].Covariance.cols())
				GMM[i].Covariance(j / GMM[i].Covariance.cols(), j % GMM[i].Covariance.cols()) = 0.;
	}

	return Initial_labels;
}

void GaussianMixture::UpdateGaussParams(Eigen::VectorXd * ClassProbabilities, std::vector<cv::Mat> Input_Image)
{
	// Compute the sum of probabilities for each cluster
	Eigen::VectorXd ClassProbSum(Clusters);
	for (size_t j = 0; j < Clusters; j++) {
		ClassProbSum(j) = 0.;
		for (size_t i = 0; i < ClassProbabilities->rows(); i++) {
			ClassProbSum(j) += ClassProbabilities[j](i);
		}
	}

	/*Update the mean vector*/
	for (size_t j = 0; j < Clusters; j++) {
		GMM[j].Mean = Eigen::Vector3d::Zero();
		for (size_t i = 0; i < ClassProbabilities->rows(); i++) {
			for (size_t c = 0; c < chnls; c++) {
				GMM[j].Mean[c] += (ClassProbabilities[j](i)*Input_Image[c].at<float>(i / Input_Image[c].cols, i % Input_Image[c].cols));
			}
		}
		for (size_t c = 0; c < chnls; c++) {
			GMM[j].Mean[c] /= ClassProbSum(j);
		}
	}

	/*Update the covariance matrix*/
	for (size_t j = 0; j < Clusters; j++) {
		GMM[j].Covariance = Eigen::Matrix3d::Zero();
		for (size_t i = 0; i < ClassProbabilities->rows(); i++) {
			Eigen::Vector3d X;
			for (size_t c = 0; c < chnls; c++) {
				X[c] = Input_Image[c].at<float>(i / Input_Image[c].cols, i % Input_Image[c].cols);
			}
			GMM[j].Covariance += ((ClassProbabilities[j](i))*((X - GMM[j].Mean)*((X - GMM[j].Mean).transpose())));
		}
		GMM[j].Covariance = (GMM[j].Covariance / ClassProbSum(j));

		/*Add a TINY to prevent divergence when inverting the covariance matrices. 
		A singular covariance matrix can exist for homogeneous images where there 
		is little variation in the pixel values.*/
		for (size_t i = 0; i < GMM[j].Covariance.rows()*GMM[j].Covariance.cols(); i++) {
			GMM[j].Covariance(i / GMM[j].Covariance.cols(), i % GMM[j].Covariance.cols()) += TINY;
			if (i / GMM[j].Covariance.cols() != i % GMM[j].Covariance.cols())
				GMM[j].Covariance(i / GMM[j].Covariance.cols(), i % GMM[j].Covariance.cols()) = 0.;
		}
	}
}

