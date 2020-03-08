#include "../include/2DFR/Function.h"
#include "../include/2DFR/FR.h"
#include "../include/2DFR/FeatureExtractor.h"
int main() {
	FR fr;
	Function fc;
	FeatureExtractor fe;
	fe.LoadModel("models/mod2", "0000", "softmax_output", 128);
	cv::VideoCapture cap;
	cap.open(0);
	cv::Mat frame;
	cap >> frame;
	std::vector<cv::Scalar> color = { cv::Scalar(0,0,255),cv::Scalar(0,255,255),cv::Scalar(0,255,0) };
	cv::VideoWriter video;
	video.open("demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(cv::CAP_PROP_FPS), cv::Size(640, 480),true);
	while (true && cv::waitKey(33)) {
		cap >> frame;
		if (frame.empty()) continue;
		double st = (double)cv::getTickCount();
		fr.detectFace(fr.preCropImage(frame));
		double t = ((double)cv::getTickCount()-st) / cv::getTickFrequency();
		cout << "detect fps " << 1 / t << endl;
		cv::Mat detected_face = fr.getDetectedFace().clone();
		if (!detected_face.empty()) {
			cv::Rect detected_area = fr.getDetectedArea();
			detected_area.x = int(detected_area.x*0.416+frame.cols/3);
			detected_area.y = int(detected_area.y*0.416+ (frame.rows - frame.cols / 3) / 2);
			detected_area.width = int(detected_area.width*0.416);
			detected_area.height = int(detected_area.height*0.416);
			cv::resize(detected_face,detected_face,cv::Size(128, 128));
			
			st = (double)cv::getTickCount();
			std::vector<double> feature = fe.Extract(detected_face);
			t = ((double)cv::getTickCount()-st) / cv::getTickFrequency();
			cout << "recognition fps " << 1 / t << endl;
			
			int max_id = 0;
			float prob = 0;
			for (int f = 0; f < feature.size(); f++)
			{
				if (feature.at(f)>prob)
				{
					max_id = f;
					prob = feature.at(f);
				}
			}
			if (detected_area.width > 0 && detected_area.height > 0) {
				cv::rectangle(frame, detected_area, color.at(max_id), 2);
				cv::putText(frame, std::to_string(prob), cv::Point(detected_area.x, detected_area.y), 1, 1, cv::Scalar(255, 0, 255));
			}
		}
		t = ((double)cv::getTickCount()-st) / cv::getTickFrequency();
		cout << "total fps "<< 1/t << endl;
		frame = fc.drawMask(frame);
		video << frame;
		cv::imshow("Eneye", frame);
	}
	cap.release();
	video.release();
}
