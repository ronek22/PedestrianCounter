#include "Person.h"

Person::Person(int _id, Point centroid, int _max_age) {
	id = _id;
	center = centroid;
	max_age = _max_age;
}

void Person::update_coords(int xn, int yn) {
	age = 0;
	tracks.push_back(Point(center.x, center.y));
	center = Point(xn, yn);
}

void Person::set_done() {
	done = true;
}

void Person::age_one() {
	age += 1;
	if (age > max_age)
		done = true;
}

bool Person::cross(int crossline) {
	if (tracks.size() >= 2) {
		if ((tracks.front().y < crossline <= tracks.back().y) || (tracks.front().y > crossline >= tracks.back().y)) {
			return true;
		}
		return false;
	}
	return false;
}



