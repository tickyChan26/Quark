// Math.h
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;


class MathUtils
{
public:
    static double gelu(double x);

    static double gelu_derivative(double x);

    static vector<double> softmax(const vector<double>& logits);

    static double crossEntropyLoss(const vector<double>& logits, int target_id);

    static double layer_norm_eps() { return 1e-6; }
};
