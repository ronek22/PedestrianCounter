#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<conio.h>
#include<ctime>
#include "Person.cpp"
#include "Person.h"

using namespace cv;
using namespace std;

// GLOBAL VARIABLES
// ================
VideoCapture cap;
vector<Person> people;
HersheyFonts FONT = FONT_HERSHEY_SIMPLEX;
Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
Mat kernelBig = getStructuringElement(MORPH_ELLIPSE, Size(40, 40));

Mat filter_mask(Mat frame) {
	// TEST
}