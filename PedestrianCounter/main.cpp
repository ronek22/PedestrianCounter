#define CVUI_IMPLEMENTATION

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
#include "cvui.h"

#define WINDOW_NAME "Pedestrian Counting System"

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

// GUI VARIABLES
bool play = true;
bool turnOn = true;
bool videoEnded = false;
int view = 1;



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
		if (inRange(DOWN_LIMIT, UP_LIMIT, center.y)) {
			for (auto it = people.begin(); it != people.end(); ) {
				if ((*it).isDone()) {
					it = people.erase(it);
				}
				else {
					Point pc = (*it).get_centroid(); // person centroid
					if ((std::abs(bounding.x - pc.x) <= bounding.width) && (std::abs(bounding.y - pc.y) <= bounding.height)) {
						isNew = false;
						(*it).update_coords(center.x, center.y);
						if ((*it).cross(BREAKLINE)) {
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

void run_system() {
	for (auto& i : people)
		i.age_one();

	// background substraction
	GaussianBlur(frame, gauss_frame, Size(25, 25), 0);
	subtractor->apply(gauss_frame, fgmask);
	filter_mask(fgmask);

	// findContors may modify input image so it's worth to copy 
	thresh.copyTo(contourImg);
	findContours(contourImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	tracking_algorithm();

	// for GUI purpose
	cvtColor(fgmask, fgmask, COLOR_GRAY2RGB);
	cvtColor(thresh, thresh, COLOR_GRAY2RGB);

	line(frame, Point(0, H / 2), Point(W, H / 2), Scalar(0, 255, 255), 2);
	line(frame, Point(0, UP_LIMIT), Point(W, UP_LIMIT), Scalar(0, 255, 255), 2);
	line(frame, Point(0, DOWN_LIMIT), Point(W, DOWN_LIMIT), Scalar(0, 255, 255), 2);
}


int main() {
	Mat GUI = Mat(700, 950, CV_8UC3);
	cap.open("example.mp4");
	int video_length = cap.get(CAP_PROP_FRAME_COUNT);
	configVar(cap);
	int progress = 0;
	cvui::init(WINDOW_NAME);
	while (true) {
		progress = cap.get(CAP_PROP_POS_FRAMES);
		if (play) {
			if (!cap.read(frame)) {
				cap.set(CAP_PROP_POS_FRAMES, 0);
				cap.read(frame);
			}
			resize(frame, frame, Size(W, H), INTER_LINEAR);
		}

		GUI = Scalar(49, 52, 49);
		if (turnOn && play)
			run_system();

		cvui::beginRow(GUI, 10, 50, 20, 150, 10);
		cvui::beginColumn(550, 500, 20);
		switch (view)
		{
			case 1:
				cvui::image(frame);
				break;
			case 2:
				cvui::image(fgmask);
				break;
			case 3:
				cvui::image(thresh);
				break;
			default:
				cvui::image(frame);
				break;
		}
		cvui::trackbar(500, &progress, 0, video_length);

		cvui::endColumn();

		cvui::beginColumn(150, 150, 20);
		cvui::printf(0.8, 0x00ff00, "Counted: %d", peopleCounted);

		cvui::beginRow();
		if (cvui::button("Original"))
			view = 1;
		if (cvui::button("FgMask"))
			view = 2;
		if (cvui::button("Binary"))
			view = 3;
		cvui::endRow();



		cvui::text("Controls", 0.7);
		cvui::text("Playback");
		cvui::beginRow();
		if (cvui::button("Play")) {
			play = true;
		}
		if (cvui::button("Pause")) {
			play = false;
		}
		if (cvui::button("Restart")) {
			cap.set(CAP_PROP_POS_FRAMES, 0);
		}
		cvui::endRow();


		cvui::text("System");
		cvui::beginRow();
		if(cvui::button("On"))
			turnOn = true;
		if (cvui::button("Off"))
			turnOn = false;
		if (cvui::button("Reset counter")) {
			peopleCounted = 0;
		}
		cvui::endRow();

		cvui::text("Settings");
		cvui::text("Minimal Area Detected", 0.3);
		cvui::trackbar(230, &MIN_AREA, 100, 10000);
		cvui::text("Maximum Area Detected", 0.3);
		cvui::trackbar(230, &MAX_AREA, 1000, 100000);
		cvui::endColumn();
		cvui::endRow();



		cvui::update();

		cv::imshow(WINDOW_NAME, GUI);

		if (cv::waitKey(20) == 27) {
			break;
		}

	}

	cap.release();
	cv::destroyAllWindows();

}
