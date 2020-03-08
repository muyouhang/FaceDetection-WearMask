#define NOMINMAX
#undef max
#undef min
#pragma once
#include <iostream>
#include <fstream>
#include <map>
#include <future>
#include <string>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace mxnet::cpp;

class FaceVerification
{
public:
	FaceVerification();
	~FaceVerification();
	void loadParams(std::string json,std::string param);
	float inference(vector<double> feature1,vector<double> feature2);
private:
	Symbol net;
	Executor *executor;
	map<string, NDArray> args_map;
	map<string, NDArray> aux_map;
	Context ctx = Context::cpu();
};

