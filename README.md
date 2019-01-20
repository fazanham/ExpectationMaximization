# ExpectationMaximization
A class for unsupervised classification using Expectation Maximization

# Installation
The project requires a few open-source libraries:
OpenCV (v3.1)
Eigen (v3.3.4)

## Setup
You can find a detailed tutorial on setting up OpenCV within your Visual Studio environment here:
https://www.deciphertechnic.com/install-opencv-with-visual-studio/

## Project Properties
Inside Properties of your project,

1. Go to C/C++ > General. Copy the path to include folder of opencv and paste it inside Additional Include Directories. The path will look similar to C:\opencv\build\include. Then, click Apply.

2. Go to linker > General. Copy the path to folder containing opencv  lib files and paste it inside Additional Library Directories. The path will look similar to C:\opencv\build\x64\vc14\lib. Then, click Apply.

3. Go to linker > Input > Additional Dependencies. Add the following lib file: opencv_world310d.lib

## Usage
To run the program, you can use the EMClassification.cpp file. The program finds initial clusters using k-means and then computes the first approximation of the gaussian parameters. Then the program iteratively computes the E-step and the M-step to find better approximations for the clusters.

## Example
![original](https://user-images.githubusercontent.com/33495209/51443375-e51c3480-1cb5-11e9-8c5c-1e38405de24f.png)
![initial clusters](https://user-images.githubusercontent.com/33495209/51443378-ee0d0600-1cb5-11e9-86b9-bae5fe9d8215.png)
![classified em](https://user-images.githubusercontent.com/33495209/51443381-f402e700-1cb5-11e9-822e-8266b8a934fd.png)

## License
Free-to-use (MIT), please use at your own risk

