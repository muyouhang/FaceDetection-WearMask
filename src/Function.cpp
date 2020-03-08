#include "Function.h"



Function::Function()
{
}


Function::~Function()
{
}
cv::Mat Function::drawMask(cv::Mat frame) {
	cv::Rect rect(frame.cols / 3, (frame.rows - frame.cols / 3) / 2, frame.cols / 3, frame.cols / 3);
	cv::line(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width / 5, rect.y), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x, rect.y + rect.height / 5), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x + rect.width * 4 / 5, rect.y), cv::Point(rect.x + rect.width, rect.y), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x + rect.width, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height / 5), cv::Scalar(0, 255, 0), 2);

	cv::line(frame, cv::Point(rect.x, rect.y + rect.height * 4 / 5), cv::Point(rect.x, rect.y + rect.height), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x, rect.y + rect.height), cv::Point(rect.x + rect.width / 5, rect.y + rect.height), cv::Scalar(0, 255, 0), 2);

	cv::line(frame, cv::Point(rect.x + rect.width, rect.y + rect.height * 4 / 5), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x + rect.width * 4 / 5, rect.y + rect.height), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 255, 0), 2);

	return frame;
}
cv::Mat Function::drawCardMask(cv::Mat frame) {
	cv::Rect rect(frame.cols * 3 / 5, (frame.rows - frame.cols / 4) / 2, frame.cols / 5, frame.cols / 5);

	cv::line(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width / 5, rect.y), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x, rect.y + rect.height / 5), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x + rect.width * 4 / 5, rect.y), cv::Point(rect.x + rect.width, rect.y), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x + rect.width, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height / 5), cv::Scalar(0, 255, 0), 2);

	cv::line(frame, cv::Point(rect.x, rect.y + rect.height * 4 / 5), cv::Point(rect.x, rect.y + rect.height), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x, rect.y + rect.height), cv::Point(rect.x + rect.width / 5, rect.y + rect.height), cv::Scalar(0, 255, 0), 2);

	cv::line(frame, cv::Point(rect.x + rect.width, rect.y + rect.height * 4 / 5), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 255, 0), 2);
	cv::line(frame, cv::Point(rect.x + rect.width * 4 / 5, rect.y + rect.height), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 255, 0), 2);
	return frame;
}