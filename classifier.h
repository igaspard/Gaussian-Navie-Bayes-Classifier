#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

	//vector<string> possible_labels = {"left","keep","right"};
	vector<string> possible_labels;
	vector<double> left_mean;
  vector<double> keep_mean;
  vector<double> right_mean;

	vector<double> left_stdev;
	vector<double> keep_stdev;
	vector<double> right_stdev;

	double left_prior, keep_prior, right_prior;
	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

	string predict(vector<double>);
};

#endif
