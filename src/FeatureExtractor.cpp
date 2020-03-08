#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor()
{
}

FeatureExtractor::~FeatureExtractor()
{
}
NDArray FeatureExtractor::Mat2NDArray(cv::Mat& image) {
	int width, height;
	width = image.cols;
	height = image.rows;
	std::vector<float> array;
	cv::cvtColor(image,image,cv::COLOR_BGR2RGB);
	int count = 0;
	for (int c = 0; c < 3; ++c) {
		for (int i = 0; i < width; ++i) {
			for (int j = 0; j < height; ++j) {
				array.push_back( static_cast<float>(image.data[(i * width + j) * 3 + c]));
			}
		}
	}
	NDArray ret(Shape(1, 3, width, height), ctx, true);
	ret.SyncCopyFromCPU(array.data(), 1 * 3 * width * height);
	NDArray::WaitAll();
	return ret;
}
int  FeatureExtractor::LoadModel(string net_name, string epoch, string layer_name, int image_size) {
	//std::cout<<"init GetFeatureSymbol..."<<std::endl;
	GetFeatureSymbol(net_name + "-symbol.json", layer_name);
	//std::cout<<"init LoadParamtes..."<<std::endl;
	LoadParamtes(net_name + "-" + epoch + ".params");
	//std::cout<<"init Mat2NDArray..."<<std::endl;
	cv::Mat temp = cv::Mat::zeros(cv::Size(image_size, image_size), CV_8UC3);
	NDArray data = Mat2NDArray(temp);
	args_map["data"] = data;
	//std::cout<<"init SimpleBind..."<<std::endl;
	executor = net.SimpleBind(ctx, args_map, map<string, NDArray>(),
		map<string, OpReqType>(), aux_map);
	//std::cout<<"finish SimpleBind..."<<std::endl;
	return 0;
}
void FeatureExtractor::GetFeatureSymbol(string symbol_name,string layer_name) {
	net = Symbol::Load(symbol_name).GetInternals()[layer_name];
}
void FeatureExtractor::GetFeatureSymbol(string symbol_name) {
	net = Symbol::Load(symbol_name).GetInternals();
}
void FeatureExtractor::LoadParamtes(string model_name) {
	map<string, NDArray> paramters;
	NDArray::Load(model_name, 0, &paramters);
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
	/*WaitAll is need when we copy data between GPU and the main memory*/
	NDArray::WaitAll();
}
std::vector<double> FeatureExtractor::Extract(cv::Mat image) {
	NDArray data = Mat2NDArray(image);
	data.CopyTo(&args_map["data"]);
	executor->Forward(false);
	auto array = executor->outputs[0].Copy(ctx_cpu);
	NDArray::WaitAll();

	std::vector<double> feature;
	for (int i = 0; i < array.Size(); i++) {
		feature.push_back(array.At(0, i));
	}
	return feature;
}
std::vector<double> FeatureExtractor::Extract(string image_name) {
	cv::Mat image = cv::imread(image_name);

	if (image.empty()) {
		cout << "File Error : "<<image_name << endl;
	}
	NDArray data = Mat2NDArray(image);
	data.CopyTo(&args_map["data"]);
	//args_map["data"] = data;
	
	executor->Forward(false);
	/*print out the features*/
	auto array = executor->outputs[0].Copy(ctx_cpu);
	NDArray::WaitAll();

	std::vector<double> feature;
	for (int i = 0; i < array.Size(); i++) {
		feature.push_back(array.At(0,i));
		//std::cout << array.At(0, i) << std::endl;
	}
	return feature;
}