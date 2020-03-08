#pragma once
#include<opencv2/opencv.hpp>
class Function
{
public:
	Function();
	~Function();
	cv::Mat drawMask(cv::Mat frame);
	cv::Mat drawCardMask(cv::Mat frame);

};

