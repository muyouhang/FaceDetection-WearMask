#include "FaceAlignment.h"

FaceAlignment::FaceAlignment()
{
	detector = dlib::get_frontal_face_detector();
	dlib::deserialize("D:/Code/fr.longxin/FR/build/2DFR/2DFR/shape_predictor_68_face_landmarks.dat") >> sp;
}


FaceAlignment::~FaceAlignment()
{
}
int FaceAlignment::test_detect(cv::Mat image) {
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::cv_image<dlib::bgr_pixel> img(image);
	std::vector<dlib::rectangle> dets = detector(img);
	std::cout << "Number of faces detected: " << dets.size() << std::endl;
}
std::vector<cv::Rect> FaceAlignment::detect(cv::Mat frame) {
	std::vector<cv::Rect> rects;
	//获取一系列人脸所在区域
	dlib::cv_image<dlib::bgr_pixel> dimg(frame);
	std::vector<dlib::rectangle> dets = detector(dimg);
	std::cout << "Number of faces detected: " << dets.size() << std::endl;
	//指出每个检测到的人脸的位置
	for (int i = 0; i<dets.size(); i++)
	{
		//画出人脸所在区域
		cv::Rect r;
		r.x = dets[i].left();
		r.y = dets[i].top();
		r.width = dets[i].width();
		r.height = dets[i].height();
		cv::rectangle(frame, r, cv::Scalar(255, 0, 255), 1, 1, 0);

	}

	////获取人脸特征点分布
	std::vector<dlib::full_object_detection> shapes;
	int i = 0;
	for (i = 0; i < dets.size(); i++)
	{
		dlib::full_object_detection shape = sp(dimg, dets[i]); //获取指定一个区域的人脸形状
		shapes.push_back(shape);
	}
	line_one_face_detections(frame, shapes);
	cv::imshow("face", frame);

	//cv::waitKey(33);
}

void FaceAlignment::line_one_face_detections(cv::Mat img, std::vector<dlib::full_object_detection> fs)
{
	int i, j;
	for (j = 0; j<fs.size(); j++)
	{
		cv::Point p1, p2;
		for (i = 0; i<67; i++)
		{
			// 下巴到脸颊 0 ~ 16
			//左边眉毛 17 ~ 21
			//右边眉毛 21 ~ 26
			//鼻梁     27 ~ 30
			//鼻孔        31 ~ 35
			//左眼        36 ~ 41
			//右眼        42 ~ 47
			//嘴唇外圈  48 ~ 59
			//嘴唇内圈  59 ~ 67
			switch (i)
			{
			case 16:
			case 21:
			case 26:
			case 30:
			case 35:
			case 41:
			case 47:
			case 59:
				i++;
				break;
			default:
				break;
			}

			p1.x = fs[j].part(i).x();
			p1.y = fs[j].part(i).y();
			p2.x = fs[j].part(i + 1).x();
			p2.y = fs[j].part(i + 1).y();
			cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2, 4, 0);
		}
	}
}
std::vector<cv::Point2d> FaceAlignment::localize(cv::Mat image,cv::Rect rect) {
	dlib::cv_image<dlib::bgr_pixel> dimg(image);
	dlib::rectangle det(rect.x,rect.y,rect.width+rect.x,rect.height+rect.y);
	dlib::full_object_detection shape = sp(dimg, det);
	std::vector<cv::Point2d> lmks;
	for (int i = 0; i < 5; i++) {
		lmks.push_back(cv::Point2d(shape.part(this->lmk[i]).x(), shape.part(this->lmk[i]).y()));
	}
	return lmks;
}
