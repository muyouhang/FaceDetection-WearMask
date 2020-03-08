#include "stdafx.h"
#include "FaceVerification.h"
#include <numeric>


FaceVerification::FaceVerification()
{

}


FaceVerification::~FaceVerification()
{
}
void FaceVerification::loadParams(std::string json, std::string param) {
	//net = Symbol::Load(json).GetInternals()["dense22_fwd_output"];
	net = Symbol::Load(json);
	map<string, NDArray> paramters;
	NDArray::Load(param, 0, &paramters);
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
	float temp[960] = {0};
	NDArray ret(Shape(1, 960), ctx, true);
	ret.SyncCopyFromCPU(temp, 1 * 960);

	args_map["data"] = ret;

	executor = net.SimpleBind(ctx, args_map, map<string, NDArray>(),
		map<string, OpReqType>(), aux_map);
}
void softmax( std::vector<double> &v) {
	double sum = 0.0;
	for (int i = 0; i<v.size(); i++)
		sum+=v[i];
	for (int i = 0; i<v.size(); i++)
		v.at(i) /= sum;
}
float FaceVerification::inference(vector<double> feature1, vector<double> feature2) {
	vector<float> f;
	for (int i = 0; i < feature1.size(); i++) {
		f.push_back((float)abs(feature1[i]-feature2[i]));
	}
	NDArray ret(Shape(1,f.size()), this->ctx, true);
	ret.SyncCopyFromCPU(f.data(), 1 * f.size());
	NDArray::WaitAll();
	ret.CopyTo(&args_map["data"]);
	executor->Forward(false);
	auto array = executor->outputs[0].Copy(this->ctx);
	
	NDArray::WaitAll();
	std::vector<double> feature;
	
	for (int i = 0; i < array.Size(); i++) {
		feature.push_back(array.At(0, i));
	}
	return feature[1];
}