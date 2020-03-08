#include "FaceDetection.h"



FaceDetection::FaceDetection()
{
	mask_data = (float *)malloc(image_size * image_size * sizeof(float));
	input_data= (float *)malloc(image_size * image_size * 3*sizeof(float));
	nd_input = NDArray(Shape(1, 3, image_size, image_size), ctx_cpu,false);
}


FaceDetection::~FaceDetection()
{
	free(mask_data);
	free(input_data);
}
NDArray FaceDetection::Mat2NDArray(cv::Mat& image) {

	int width, height;
	width = image.cols;
	height = image.rows;
	std::vector<float> array;
	int count = 0;
	for (int c = 0; c < 3; ++c) {
		for (int i = 0; i < width; ++i) {
			for (int j = 0; j < height; ++j) {
				array.push_back(static_cast<float>(image.data[(i * width + j) * 3 + c]));
			}
		}
	}
	NDArray data(array.data(), Shape(1, 3, width, height), ctx);
	return data;
}
bool FaceDetection::LoadModel(string net_name, string epoch, string layer_name,int input_size) {
	this->image_size = input_size;
	net = Symbol::Load(net_name + "-symbol.json").GetInternals()[layer_name];
	map<string, NDArray> paramters;
	NDArray::Load(net_name + "-" + epoch + ".params", 0, &paramters);
	for (const auto &k : paramters) {
		if (k.first.substr(0, 4) == "aux:") {
			auto name = k.first.substr(4, k.first.size() - 4);
			aux_map[name] = k.second.Copy(ctx);
		}
		if (k.first.substr(0, 4) == "arg:") {
			auto name = k.first.substr(4, k.first.size() - 4);
			args_map[name] = k.second.Copy(ctx);
		}
	}
	NDArray::WaitAll();
	cv::Mat temp = cv::Mat::zeros(cv::Size(image_size, image_size), CV_8UC3);
	NDArray data= Mat2NDArray(temp);
	args_map["data"] = data;
	executor = net.SimpleBind(ctx, args_map, map<string, NDArray>(),map<string, OpReqType>(), aux_map);
	return true;
}
cv::Mat FaceDetection::Forward(cv::Mat image) {
	NDArray data = Mat2NDArray(image);
	data.CopyTo(&args_map["data"]);
	//data.CopyTo(&(executor->arg_dict()["data"]));
	executor->Forward(false);
	auto array = executor->outputs[0].Copy(ctx_cpu);
	array = array.Reshape(Shape(image.rows, image.cols));
	NDArray::WaitAll();

	memcpy(mask_data, array.GetData(), image.cols*image.rows * sizeof(float));
	cv::Mat result(cv::Size(1, image.cols*image.rows), CV_32FC1, mask_data);
	result = result.reshape(0, image.rows);
	result *= 255;
	result.convertTo(result, CV_8UC1);
	return result;
}
std::vector<cv::Rect> FaceDetection::Detect(cv::Mat image) {
	std::vector<std::vector<cv::Point>> contours;
	//cv::resize(image, image, cv::Size(256, 256));
	cv::Mat result = Forward(image);
	cv::findContours(result, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::vector<cv::Rect> rects = findRect(contours);
	return rects;
}
std::vector<cv::Rect> FaceDetection::findRect(std::vector<std::vector<cv::Point>> contours) {
	vector<cv::Rect> rects;
	cv::Rect face_rect(0,0,0,0);
	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = boundingRect(contours.at(i));
		cv::Point center = cv::Point(int(rect.x + rect.width/ 2), int(rect.y + rect.height/ 2));
		//int edge_size = max(rect.width, rect.height);
		int edge_size =(rect.width+rect.height)/2;

		rect.x = int(center.x - edge_size / 2);
		rect.y= int(center.y - edge_size / 2);
		rect.width = edge_size;
		rect.height = edge_size;
		if (rect.height > 20 && rect.width > 20) {
			rects.push_back(rect);
		}
		if (rect.width > face_rect.width) {
			face_rect = rect;
		}
	}
	rects.clear();
	if (face_rect.width < 30) {
		face_rect = cv::Rect(0, 0, 0, 0);
	}
	//std::cout << "face area = " << face_rect.width*face_rect.height / (128.0*128.0) << std::endl;
	//if ((face_rect.width*face_rect.height / (128.0*128.0))<0.7) {
	//	face_rect = cv::Rect(0, 0, 0, 0);
	//}
	rects.push_back(face_rect);
	return rects;
}
void FaceDetection::drawRect(cv::Mat& image, vector<cv::Rect> rects) {
	if (rects.size() > 0) {
		for (int i = 0; i < rects.size(); i++) {
			if (rects.at(i).height < 30) {
				continue;
			}
			else {
				rectangle(image, rects.at(i), cv::Scalar(255, 0, 255));
				cv::Mat buf_img(image(rects.at(i)));
			}
		}
	}
}
cv::Mat FaceDetection::resize(cv::Mat inputImage, cv::Size size) {
	int s;
	if (inputImage.rows > inputImage.cols) {
		s = inputImage.rows;
	}
	else {
		s = inputImage.cols;
	}
	cv::Mat image_temp_ep(s, s, CV_8UC3, cv::Scalar(0, 0, 0));
	if (inputImage.channels() == 1) {
		cvtColor(image_temp_ep, image_temp_ep, cv::COLOR_BGR2GRAY);
	}
	cv::Mat image_temp_ep_roi = image_temp_ep(cv::Rect((s - inputImage.cols) / 2, (s - inputImage.rows) / 2, inputImage.cols, inputImage.rows));
	cv::Mat dstNormImg;
	addWeighted(image_temp_ep_roi, 0., inputImage, 1.0, 0., image_temp_ep_roi);
	cv::resize(image_temp_ep, dstNormImg, size, 0, 0, 1);    //大小归一化
	return dstNormImg;
}