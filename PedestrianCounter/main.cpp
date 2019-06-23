#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/video.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<conio.h>
#include<ctime>
#include<string>
#include<stdlib.h>
#include "Person.h"

using namespace cv;
using namespace std;

// GLOBAL VARIABLES
// ================
VideoCapture cap;
vector<Person> people;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
HersheyFonts FONT = FONT_HERSHEY_SIMPLEX;
Mat frame, gauss_frame, fgmask, contourImg;
Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
Mat kernelBig = getStructuringElement(MORPH_ELLIPSE, Size(40, 40));
Mat closing, opening, dilation, thresh; // for morphology
int H, W, FPS, AREA_FRAME, MAX_AREA, MIN_AREA, UP_LIMIT, DOWN_LIMIT;
Ptr<BackgroundSubtractorMOG2> subtractor = createBackgroundSubtractorMOG2();
Moments M;
bool isNew;

// COUNTERS
int peopleCounted = 0;
int pID = 1;
int R_WIDTH = 500;

// COUNTING LINE
int BREAKLINE = 0;

void configVar(VideoCapture cap) {
	FPS = cap.get(CAP_PROP_FPS);
	W = cap.get(CAP_PROP_FRAME_WIDTH);
	H = cap.get(CAP_PROP_FRAME_HEIGHT);
	// after resizing
	float r = R_WIDTH / float(W);
	int height = int(H * r);
	W = R_WIDTH;
	H = height;
	BREAKLINE = int(H / 2);
	AREA_FRAME = H * W;
	MAX_AREA = int(AREA_FRAME / 8);
	MIN_AREA = int(AREA_FRAME / 100);
	UP_LIMIT = int(0.6 * H);
	DOWN_LIMIT = int(0.4 * H);
}


void filter_mask(Mat frame) {
	morphologyEx(frame, closing, MORPH_CLOSE, kernelBig);
	morphologyEx(closing, opening, MORPH_OPEN, kernel);
	dilate(opening, dilation, kernel, Point(-1, -1), 5);
	threshold(dilation, thresh, 220, 240, THRESH_BINARY);
}

bool inRange(unsigned low, unsigned high, unsigned x)
{
	return  ((x - low) <= (high - low));
}

void tracking_algorithm() {
	vector<Point> contour;
	Point center;
	Rect bounding;

	for (size_t i = 0; i < contours.size(); ++i) {
		contour = contours[i];

		if (hierarchy[i][3] != -1)
			continue;

		//cout << H << ", " << W << ", " << AREA_FRAME << ", " << MAX_AREA << ", " << MIN_AREA << endl;

		if (!inRange(MIN_AREA, MAX_AREA, int(contourArea(contour))))
			continue;


		// CALC CENTER OF CONTOUR
		M = moments(contour);
		double M01 = M.m01;
		double M10 = M.m10;
		double area = M.m00;
		center = Point(M10 / area, M01 / area);
		bounding = boundingRect(contour);

		isNew = true;
		if (inRange(DOWN_LIMIT, UP_LIMIT, center.y)){
			for (auto it = people.begin(); it != people.end(); ) {
				if ((*it).isDone()) {
					it = people.erase(it);
				}
				else {
					Point pc = (*it).get_centroid(); // person centroid
					if ((std::abs(bounding.x - pc.x) <= bounding.width) && (std::abs(bounding.y - pc.y) <= bounding.height)) {
						//cout << "ABS: " << std::abs(bounding.x - pc.x) << " <= " << bounding.width << endl;
						isNew = false;
						(*it).update_coords(center.x, center.y);
						if ((*it).cross(BREAKLINE)) { // SOMETHING DOESNT WORK HERE
							peopleCounted++;
							(*it).set_done();
							break;
						}
					}
					it++;
				}
			}
			if (isNew) {
				people.push_back(Person(pID++, center, 5));
			}

		}

		cv::rectangle(frame, bounding, Scalar(255, 0, 0), 2);
		cv::drawMarker(frame, center, Scalar(0, 0, 255), MARKER_STAR, 5, 1, LINE_AA);
	}
}

int main() {
	cap.open("example_02.mp4");
	configVar(cap);

	while (cap.read(frame) && waitKey(10) != 27) {
		for (auto& i : people)
			i.age_one();

		// background substraction
		resize(frame, frame, Size(W, H), INTER_LINEAR);
		GaussianBlur(frame, gauss_frame, Size(25, 25), 0);
		subtractor->apply(gauss_frame, fgmask);
		filter_mask(fgmask);

		// findContors may modify input image so it's worth to copy 
		thresh.copyTo(contourImg);
		findContours(contourImg, contours, hierarchy,  RETR_TREE, CHAIN_APPROX_SIMPLE);
		tracking_algorithm();

		/*for (auto& i : people)
			putText(frame, to_string(i.get_id()), i.get_centroid(), FONT, 0.3, Scalar(0, 0, 255), 1);*/

		line(frame, Point(0, H / 2), Point(W, H / 2), Scalar(0, 255, 255), 2);
		line(frame, Point(0, UP_LIMIT), Point(W,UP_LIMIT), Scalar(0, 255, 255), 2);
		line(frame, Point(0, DOWN_LIMIT), Point(W, DOWN_LIMIT), Scalar(0, 255, 255), 2);
		putText(frame, to_string(peopleCounted), Point(10, 40), FONT, 0.5, Scalar(255, 255, 255), 2, LINE_AA);


		imshow("Frame", frame);
		imshow("Mask", fgmask);
		imshow("Binary", thresh);
	}

	cap.release();
	destroyAllWindows();
}