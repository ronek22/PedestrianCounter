#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#pragma once
class Person
{
private:
	int id;
	vector<Point> tracks;
	Point center;
	int age, max_age;
	bool done;

public:
	Person(int _id, Point centroid, int _max_age);
	void update_coords(int xn, int yn);
	void set_done();
	void age_one();
	bool cross(int crossline);
};

