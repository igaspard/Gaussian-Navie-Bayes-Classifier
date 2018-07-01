#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <numeric>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {
  possible_labels.push_back("left");
  possible_labels.push_back("keep");
  possible_labels.push_back("right");
}

GNB::~GNB() {}

void GNB::train(vector< vector<double> > data, vector<string> labels)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d,
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
  vector< vector<double> > left_data, keep_data, right_data;
  vector<double> left_sum(4);
  vector<double> keep_sum(4);
  vector<double> right_sum(4);

  for (unsigned int i = 0; i < data.size(); ++i) {
    if (labels[i] == possible_labels[0]) {
      left_data.push_back(data[i]);
      for (size_t j = 0; j < data[0].size(); j++) {
        left_sum[j] += data[i][j];
      }
    }
    else if (labels[i] == possible_labels[1]) {
      keep_data.push_back(data[i]);
      for (size_t j = 0; j < data[0].size(); j++) {
        keep_sum[j] += data[i][j];
      }
    }
    else if (labels[i] == possible_labels[2]) {
      right_data.push_back(data[i]);
      for (size_t j = 0; j < data[0].size(); j++) {
        right_sum[j] += data[i][j];
      }
    }
    else {
      cout << "Invalid Labels !" << endl;
    }
  }
  // cout << "Number of Left Data: " << left_data.size();
  // cout << ", Keep data: " << keep_data.size();
  // cout << ", Right data: " << right_data.size() << endl;
  // cout << "Size of Labels: " << labels.size();
  left_prior  = (double) left_data.size() / labels.size();
  keep_prior  = (double) keep_data.size() / labels.size();
  right_prior = (double) right_data.size() / labels.size();

  cout << "Ratio: " << left_prior << " ," << keep_prior << " ," << right_prior << endl;

  for (unsigned int i = 0; i < left_sum.size(); ++i) {
    left_mean.push_back(left_sum[i] / left_data.size());
    keep_mean.push_back(keep_sum[i] / keep_data.size());
    right_mean.push_back(right_sum[i] / right_data.size());
  }
  cout << "Left Mean: ";
  for (unsigned int i = 0; i < left_mean.size(); ++i) {
    cout << left_mean[i] << " ,";
  }
  cout << endl;
  cout << "Keep Mean: ";
  for (unsigned int i = 0; i < keep_mean.size(); ++i) {
    cout << keep_mean[i] << " ,";
  }
  cout << endl;
  cout << "Right Mean: ";
  for (unsigned int i = 0; i < right_mean.size(); ++i) {
    cout << right_mean[i] << " ,";
  }
  cout << endl;

  vector<double> left_accum(4);
  for (unsigned int i = 0; i < left_data.size(); ++i) {
    for (unsigned int j = 0; j < left_data[0].size(); ++j) {
      left_accum[j] += pow(left_data[i][j] - left_mean[j], 2);
    }
  }

  vector<double> keep_accum(4);
  for (unsigned int i = 0; i < keep_data.size(); ++i) {
    for (unsigned int j = 0; j < keep_data[0].size(); j++) {
      keep_accum[j] += pow(keep_data[i][j] - keep_mean[j], 2);
    }
  }

  vector<double> right_accum(4);
  for (unsigned int i = 0; i < right_data.size(); ++i) {
    for (unsigned int j = 0; j < right_data[0].size(); ++j) {
      right_accum[j] += pow(right_data[i][j] - right_mean[j], 2);
    }
  }

  // vector<double> left_stdev(4);
  cout << "Left Stdev: ";
  for (unsigned int i = 0; i < left_accum.size(); ++i) {
    left_stdev.push_back(left_accum[i] / left_data.size());
    cout << left_stdev[i] << " ,";
  }
  cout << endl;

  // vector<double> keep_stdev(4);
  cout << "Keep Stdev: ";
  for (unsigned int i = 0; i < keep_accum.size(); ++i) {
    keep_stdev.push_back(keep_accum[i] / keep_data.size());
    cout << keep_stdev[i] << " ,";
  }
  cout << endl;

  // vector<double> right_stdev(4);
  cout << "Right Stdev: ";
  for (unsigned int i = 0; i < right_accum.size(); ++i) {
    right_stdev.push_back(right_accum[i] / right_data.size());
    cout << right_stdev[i] << " ,";
  }
  cout << endl;
}

double gaussian_prob(double v, double mean, double stdev)
{
  double num = pow(v - mean, 2);
  double denum = pow(2*stdev, 2);
  double norm = 1 / sqrt(2*M_PI*stdev*stdev);
  return norm * exp(-num / denum);
}

string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/
  // Compute the conditional probability for each feature
  double left_prob = 1.0;
  double keep_prob = 1.0;
  double right_prob = 1.0;
  for (unsigned int i = 0; i < sample.size(); ++i) {
     left_prob  *= gaussian_prob(sample[i], left_mean[i], left_stdev[i]);
     keep_prob  *= gaussian_prob(sample[i], keep_mean[i], keep_stdev[i]);
     right_prob *= gaussian_prob(sample[i], right_mean[i], right_stdev[i]);
  }

  left_prob *= left_prior;
  keep_prob *= keep_prior;
  right_prob *= right_prior;

  // cout << "Left Prob: " << left_prob << endl;
  // cout << "Keep Prob: " << keep_prob << endl;
  // cout << "Right Prob: "<< right_prob<< endl;

  double probs[3] = {left_prob, keep_prob, right_prob};
  double max_prob = left_prob;
  int index = 0;
  for (size_t i = 1; i < 3; ++i) {
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      index = i;
    }
  }

	return this->possible_labels[index];
}
