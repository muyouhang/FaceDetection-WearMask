#pragma once
#include <map>
#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace mxnet::cpp;

class FaceDetection
{
public:
	FaceDetection();
	~FaceDetection();
	bool LoadModel(string net_name, string epoch, string layer_name,int input_size);
	std::vector<cv::Rect> Detect(cv::Mat image);
	cv::Mat resize(cv::Mat inputImage, cv::Size size);
private:
	Symbol net;
	float *mask_data;
	float *input_data;
	Executor *executor;
	map<string, NDArray> aux_map;
	map<string, NDArray> args_map;
	Context ctx = Context::cpu();
	Context ctx_cpu = Context::cpu();
	int image_size = 128;
	NDArray nd_input;
	
	NDArray Mat2NDArray(cv::Mat& image);
	cv::Mat Forward(cv::Mat image);
	std::vector<cv::Rect> findRect(std::vector<std::vector<cv::Point>> contours);
	void drawRect(cv::Mat& image, vector<cv::Rect> rects);

};

