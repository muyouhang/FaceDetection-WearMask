#pragma once
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <opencv2\opencv.hpp>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
class FaceAlignment
{
public:
	FaceAlignment();
	~FaceAlignment();
	cv::Mat findMaxFace(cv::Mat frame);
	std::vector<cv::Rect> detect(cv::Mat frame);
	int test_detect(cv::Mat image);
	std::vector<cv::Point2d> localize(cv::Mat image, cv::Rect rect);
private:
	dlib::shape_predictor sp;
	dlib::frontal_face_detector detector;
	int lmk[5] = { 30,39,42,48,54 };
	void line_one_face_detections(cv::Mat img, std::vector<dlib::full_object_detection> fs);
};
