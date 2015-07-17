#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <windows.h>
#include <io.h>
#include <time.h>

using namespace cv;
using namespace std;


#define HORIZONTAL    1
#define VERTICAL    0

CvANN_MLP  ann;

const char strCharacters[] = { '0','1','2','3','4','5',\
'6','7','8','9'};
const int numCharacter = 10; 

const int numNeurons = 40;
const int predictSize = 10;


void generateRandom(int n, int test_num, int min, int max, vector<int>*mark_samples)
{
	int range = max - min;
	int index = rand() % range + min;
	if (mark_samples->at(index) == 0)
	{
		mark_samples->at(index) = 1;
		n++;
	}

	if (n < test_num)
		generateRandom(n, test_num, min, max, mark_samples);

}


vector<string> getFiles(const string &folder,
	const bool all /* = true */) {
	vector<string> files;
	list<string> subfolders;
	subfolders.push_back(folder);

	while (!subfolders.empty()) {
		string current_folder(subfolders.back());

		if (*(current_folder.end() - 1) != '/') {
			current_folder.append("/*");
		}
		else {
			current_folder.append("*");
		}

		subfolders.pop_back();

		struct _finddata_t file_info;
		long file_handler = _findfirst(current_folder.c_str(), &file_info);

		while (file_handler != -1) {
			if (all &&
				(!strcmp(file_info.name, ".") || !strcmp(file_info.name, ".."))) {
				if (_findnext(file_handler, &file_info) != 0) break;
				continue;
			}

			if (file_info.attrib & _A_SUBDIR) {
				// it's a sub folder
				if (all) {
					// will search sub folder
					string folder(current_folder);
					folder.pop_back();
					folder.append(file_info.name);

					subfolders.push_back(folder.c_str());
				}
			}
			else {
				// it's a file
				string file_path;
				// current_folder.pop_back();
				file_path.assign(current_folder.c_str()).pop_back();
				file_path.append(file_info.name);

				files.push_back(file_path);
			}

			if (_findnext(file_handler, &file_info) != 0) break;
		}  // while
		_findclose(file_handler);
	}

	return files;
}

void AppendText(string filename, string text)
{
	fstream ftxt;
	ftxt.open(filename, ios::out | ios::app);
	if (ftxt.fail())
	{
		cout << "创建文件失败!" << endl;
		getchar();
	}
	ftxt << text << endl;
	ftxt.close();
}

// ！获取垂直和水平方向直方图
Mat ProjectedHistogram(Mat img, int t)
{
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j<sz; j++) {
		Mat data = (t) ? img.row(j) : img.col(j);
		mhist.at<float>(j) = countNonZero(data);	//统计这一行或一列中，非零元素的个数，并保存到mhist中
	}

	//Normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);

	if (max>0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);//用mhist直方图中的最大值，归一化直方图

	return mhist;
}

Mat features(Mat in, int sizeData)
{
	//Histogram features
	
	Mat vhist = ProjectedHistogram(in, VERTICAL);
	Mat hhist = ProjectedHistogram(in, HORIZONTAL);
	//Low data feature
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));

	
	//Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols*lowData.cols;
	//int numCols = vhist.cols + hhist.cols;
	Mat out = Mat::zeros(1, numCols, CV_32F);
	//Asign values to feature,ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
	int j = 0;
	for (int i = 0; i<vhist.cols; i++)
	{
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i<hhist.cols; i++)
	{
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x<lowData.cols; x++)
	{
		for (int y = 0; y<lowData.rows; y++) {
			out.at<float>(j) = (float)lowData.at<unsigned char>(x, y);
			j++;
		}
	}
	//if(DEBUG)
	//	cout << out << "\n===========================================\n";
	return out;
}

void annTrain(Mat TrainData, Mat classes, int nNeruns)
{
	ann.clear();
	Mat layers(1, 3, CV_32SC1);
	layers.at<int>(0) = TrainData.cols;
	layers.at<int>(1) = nNeruns;
	layers.at<int>(2) = numCharacter;
	ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

	//Prepare trainClases
	//Create a mat with n trained data by m classes
	Mat trainClasses;
	trainClasses.create(TrainData.rows, numCharacter, CV_32FC1);
	for (int i = 0; i < trainClasses.rows; i++)
	{
		for (int k = 0; k < trainClasses.cols; k++)
		{
			//If class of data i is same than a k class
			if (k == classes.at<int>(i))
				trainClasses.at<float>(i, k) = 1;
			else
				trainClasses.at<float>(i, k) = 0;
		}
	}
	Mat weights(1, TrainData.rows, CV_32FC1, Scalar::all(1));

	//Learn classifier
	// ann.train( TrainData, trainClasses, weights );

	//Setup the BPNetwork

	// Set up BPNetwork's parameters
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.1;
	params.bp_moment_scale = 0.1;

	//params.train_method=CvANN_MLP_TrainParams::RPROP;
	// params.rp_dw0 = 0.1; 
	// params.rp_dw_plus = 1.2; 
	// params.rp_dw_minus = 0.5;
	// params.rp_dw_min = FLT_EPSILON; 
	// params.rp_dw_max = 50.;

	ann.train(TrainData, trainClasses, Mat(), Mat(), params);

}

int recog(Mat features)
{
	int result = -1;
	Mat Predict_result(1, numCharacter, CV_32FC1);
	ann.predict(features, Predict_result);
	Point maxLoc;
	double maxVal;

	minMaxLoc(Predict_result, 0, &maxVal, 0, &maxLoc);

	return maxLoc.x;
}

float ANN_test(Mat samples_set, Mat sample_labels)
{
	int correctNum = 0;
	float accurate = 0;
	for (int i = 0; i < samples_set.rows; i++)
	{
		int result = recog(samples_set.row(i));
		if (result == sample_labels.at<int>(i))
			correctNum++;
	}
	accurate = (float)correctNum / samples_set.rows;
	return accurate;
}

int saveTrainData()
{
	cout << "Begin saveTrainData" << endl;
	Mat classes;
	Mat trainingDataf5;
	Mat trainingDataf10;
	Mat trainingDataf15;
	Mat trainingDataf20;

	vector<int> trainingLabels;
	string path = "charSamples";

	for (int i = 0; i < numCharacter; i++)
	{
		cout << "Character: " << strCharacters[i] << "\n";
		stringstream ss(stringstream::in | stringstream::out);
		ss << path << "/" << strCharacters[i];

		auto files = getFiles(ss.str(),1);

		int size = files.size();
		for (int j = 0; j < size; j++)
		{
			cout << files[j].c_str() << endl;
			Mat img = imread(files[j].c_str(), 0);
			Mat f5 = features(img, 5);
			Mat f10 = features(img, 10);
			Mat f15 = features(img, 15);
			Mat f20 = features(img, 20);

			trainingDataf5.push_back(f5);
			trainingDataf10.push_back(f10);
			trainingDataf15.push_back(f15);
			trainingDataf20.push_back(f20);
			trainingLabels.push_back(i);			//每一幅字符图片所对应的字符类别索引下标
		}
	}

	

	trainingDataf5.convertTo(trainingDataf5, CV_32FC1);
	trainingDataf10.convertTo(trainingDataf10, CV_32FC1);
	trainingDataf15.convertTo(trainingDataf15, CV_32FC1);
	trainingDataf20.convertTo(trainingDataf20, CV_32FC1);
	Mat(trainingLabels).copyTo(classes);

	FileStorage fs("train/features_data.xml", FileStorage::WRITE);
	fs << "TrainingDataF5" << trainingDataf5;
	fs << "TrainingDataF10" << trainingDataf10;
	fs << "TrainingDataF15" << trainingDataf15;
	fs << "TrainingDataF20" << trainingDataf20;
	fs << "classes" << classes;
	fs.release();

	cout << "End saveTrainData" << endl;

	return 0;
}


void ANN_Cross_Train_and_Test(int Imagsize, int Layers )
{
	
	String training;
	Mat TrainingData;
	Mat Classes;

	FileStorage fs;
	fs.open("train/features_data.xml", FileStorage::READ);

	
	cout << "Begin to ANN_Cross_Train_and_Test " << endl;

	char *txt = new char[50];
	sprintf(txt, "交叉训练，特征维度%d,网络层数%d", 40 + Imagsize * Imagsize, Layers);
	AppendText("output.txt", txt);
	cout << txt << endl;
	stringstream ss(stringstream::in | stringstream::out);
	ss << "TrainingDataF" << Imagsize;
	training = ss.str();

	fs[training] >> TrainingData;
	fs["classes"] >> Classes;
	fs.release();

	float result = 0.0;

	srand(time(NULL));

	vector<int> markSample(TrainingData.rows, 0);

	generateRandom(0, 100, 0, TrainingData.rows - 1, &markSample);

	Mat train_set, train_labels;
	Mat sample_set, sample_labels;

	for (int i = 0; i < TrainingData.rows; i++)
	{
		if (markSample[i] == 1)
		{
			sample_set.push_back(TrainingData.row(i));
			sample_labels.push_back(Classes.row(i));
		}
		else
		{
			train_set.push_back(TrainingData.row(i));
			train_labels.push_back(Classes.row(i));
		}
	}

	annTrain(train_set, train_labels, Layers);

	result = ANN_test(sample_set, sample_labels);

			
	sprintf(txt, "正确率%f\n", result);
	cout << txt << endl;
	AppendText("output.txt", txt);

	cout << "End the ANN_Cross_Train_and_Test" << endl;
	
	cout << endl;
	

}


void ANN_test_Main()
{
	int DigitSize[4] = { 5, 10, 15, 20};
	int LayerNum[14] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 500 };
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 14; j++)
		{
			ANN_Cross_Train_and_Test(DigitSize[i], LayerNum[j]);
		}
	}
}


void ANN_saveModel(int _predictsize, int _neurons)
{
	FileStorage fs;
	fs.open("train/features_data.xml", FileStorage::READ);

	Mat TrainingData;
	Mat Classes;

	string training;
	if (1)
	{
		stringstream ss(stringstream::in | stringstream::out);
		ss << "TrainingDataF" << _predictsize;
		training = ss.str();
	}

	fs[training] >> TrainingData;
	fs["classes"] >> Classes;

	//train the Ann
	cout << "Begin to saveModelChar predictSize:" << _predictsize
		<< " neurons:" << _neurons << endl;


	annTrain(TrainingData, Classes, _neurons);

	

	cout << "End the saveModelChar" << endl;


	string model_name = "train/ann10_40.xml";
	//if(1)
	//{
	//	stringstream ss(stringstream::in | stringstream::out);
	//	ss << "ann_prd" << _predictsize << "_neu"<< _neurons << ".xml";
	//	model_name = ss.str();
	//}

	FileStorage fsTo(model_name, cv::FileStorage::WRITE);
	ann.write(*fsTo, "ann");
}

int main()
{
	cout << "To be begin." << endl;

	saveTrainData();

	//ANN_saveModel(10, 40);
	
	ANN_test_Main();

	

	

	cout << "To be end." << endl;
	int end;
	cin >> end;
	return 0;
}