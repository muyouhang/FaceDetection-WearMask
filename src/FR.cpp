#include <iostream>
#include "../include/2DFR/FR.h"

FR::FR()
{
	std::cout << "Init ..." << std::endl;
	fd.LoadModel("models/mod1", "0000", "segnet0_relu8_fwd_output", 128);
	std::cout << "Done" << std::endl;
}


FR::~FR()
{
}
int FR::setGallery(std::vector<cv::Mat> images, std::vector<std::string> names, std::string task) {
	for (int i = 0; i < images.size(); i++) {
		cv::Mat face = this->detect(images[i]);
		if (face.empty()) {
			std::cout << "Error: " << names[i] << std::endl;
			continue;
		}
		std::vector<double> feature = fe.Extract(face);
		fr.AddGallery(feature, this->gallery_names.size());
		this->gallery_names.push_back(names[i]);
		//std::cout << names[i] << std::endl;
	}
	return 0;
}
void  FR::releaseGallery() {
	fr.ReleaseGallery();
}
std::string FR::recProbe(cv::Mat image, std::string task) {
	cv::Mat face = this->detect(image);
    if(face.empty()){
        return "noface";
    }
	cv::Rect detected_area = this->getDetectedArea();
	if (detected_area.width * detected_area.height / 512.0 / 512.0 < MIN_FACE_AREA_RATIO) {
		return "smallface";
	}
}
float FR::verify(cv::Mat face_image, cv::Mat card_image){
	cv::Mat card_face = this->detect(card_image);
	if (card_face.empty()) {
		return 0;
	}
	cv::Mat camera_face = this->detect(face_image);
	if (camera_face.empty()) {
		return 0;
	}
	std::vector<double> card_face_feature = fe.Extract(card_face);
	std::vector<double> camera_face_feature = fe.Extract(camera_face);
	return fr.CalcSimilarity(card_face_feature,camera_face_feature);
}
void FR::detectFace(cv::Mat& frame)
{
	this->detected_area = cv::Rect(0,0,0,0);
	this->detect(frame);
	//cv::Rect detected_area = this->getDetectedArea();
	//cv::rectangle(frame,cv::Rect(int(this->detected_area.x*0.416), int(this->detected_area.y*0.416), 
	//	int(this->detected_area.width*0.416), int(this->detected_area.height*0.416)),cv::Scalar(0,0,255),2);
	//if (int(this->detected_area.x*0.416) > 10) {
	//	cv::putText(frame, "muguodong", cv::Point(int(this->detected_area.x*0.416) + 30, int(this->detected_area.y*0.416) + 15),
	//		cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
	//}
}
cv::Mat FR::detect(cv::Mat image) {
    if(image.empty()){
        return cv::Mat();
    }
	cv::Mat face_tmp = fd.resize(image, cv::Size(512, 512));
	normlized_image = face_tmp;
	cv::Mat face = fd.resize(image, cv::Size(128, 128));
	std::vector<cv::Rect> rects = fd.Detect(face);
	this->detected_area = cv::Rect(0,0,0,0);
	if (rects[0].width == 0 ||
		rects[0].height == 0 ||
		rects[0].x >= 128 ||
		rects[0].y >= 128 ||
        rects[0].x <= 0   ||
        rects[0].y <= 0   ||
		rects[0].width + rects[0].y > 128 ||
		rects[0].height + rects[0].x > 128)
		return cv::Mat();
//    std::cout << rects[0].x * 4 << " " << rects[0].y * 4 << " " << rects[0].width * 4 << " " << rects[0].height * 4 << std::endl;
	cv::Mat face_roi = face_tmp(cv::Rect(rects[0].x * 4, rects[0].y * 4, rects[0].width * 4, rects[0].height * 4));
	cv::resize(face_roi, face_roi, cv::Size(112, 112));
	this->detected_face = face_roi;
	this->detected_area = cv::Rect(rects[0].x * 4, rects[0].y * 4, rects[0].width * 4, rects[0].height * 4);
	return face_roi;
}
cv::Mat FR::getNomrlizedImage() {
	return this->normlized_image;
}
cv::Mat FR::getDetectedFace() {
	return this->detected_face;
}
cv::Rect FR::getDetectedArea() {
	return this->detected_area;
}
cv::Mat FR::preCropImage(cv::Mat& frame) {
	if (frame.empty()) return cv::Mat();
	cv::Rect rect(frame.cols / 3, (frame.rows - frame.cols / 3) / 2, frame.cols / 3, frame.cols / 3);
	return frame(rect);
}
cv::Mat FR::preCropCardImage(cv::Mat frame) {
	if (frame.empty()) return cv::Mat();
	cv::Rect rect(frame.cols *3/ 5, (frame.rows - frame.cols / 4) / 2, frame.cols / 5, frame.cols / 5);
	return frame(rect);
}