// файл  Math.cpp
#include "Math.h"
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

double MathUtils::gelu(double x)
{
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
}


double MathUtils::gelu_derivative(double x)
{

    double tanh_input = sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3));
    double tanh_val = tanh(tanh_input);
    double sech2 = 1.0 - tanh_val * tanh_val;
    double deriv = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * sqrt(2.0 / M_PI) * (1.0 + 0.134145 * x * x);

    return deriv;
}

vector<double> MathUtils::softmax(const vector<double>& logits)
{
    vector<double> result(logits.size());
    double max_logit = *max_element(logits.begin(), logits.end());

    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] = exp(logits[i] - max_logit);
        sum += result[i];
    }

    for (size_t i = 0; i < result.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

double MathUtils::crossEntropyLoss(const vector<double>& logits, int target_id)
{
    double max_logit = *max_element(logits.begin(), logits.end());

    double sum_exp = 0.0;
    for (double logit : logits) {
        sum_exp += exp(logit - max_logit);
    }

    double log_softmax_target = logits[target_id] - max_logit - log(sum_exp);

    return -log_softmax_target;
}
