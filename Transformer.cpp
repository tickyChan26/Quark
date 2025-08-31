// Файл Transformer.cpp, классы и нужные переменные обьявлены в Transformer.h

#define NOMINMAX
#define _HAS_STD_BYTE 0


#include "Transformer.h"
#include "Math.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <openblas/cblas.h>  
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <map>

/*
на будущее другие виды загрузок и сохранений, пока опустим этот момент так как получается даже хуже чем с json

#include <rocksdb/db.h>   
#include <rocksdb/options.h>
#include <rocksdb/write_batch.h>
#include <rpc.h>
#include <rpcndr.h>

*/


using namespace std;
using json = nlohmann::json;

////////////////////////// ===== TENSOR IMPLEMENTATION ===== //////////////////////////

// === 1. Конструктор: создаёт матрицу, подготавливает память 
Tensor::Tensor(int r, int c, bool requires_grad_param) : rows(r), cols(c), requires_grad(requires_grad_param), d_k(0), backward_cache_len(0)
{
    data.resize(r, vector<double>(c, 0.0));
    if (requires_grad) {  grad.resize(r, vector<double>(c, 0.0));  }
}

                 /* ФУНКЦИИ ДЛЯ TRANSORMER */
// ==== 2. Случайная инициализация: заполняет матрицу случайными числами ====
void Tensor::randomize() 
{
    double limit = sqrt(6.0 / (rows + cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

void Tensor::zero() 
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = 0.0;
        }
    }
}
void Tensor::zero_grad() 
{
    if (!requires_grad)
    {
        wcout << L" !!! zero_grad: Пропущен \n" << endl;
        return;
    }

    if (grad.empty() || grad.size() != rows || (rows > 0 && grad[0].size() != cols)) 
    {
        grad.resize(rows, vector<double>(cols, 0.0));
    }
    for (int i = 0; i < rows; i++) {
        fill(grad[i].begin(), grad[i].end(), 0.0);
    }
}


// ==== 4. Матрица умножение: базовая операция для attention и feedforward ====

// Вспомогательный метод для конвертации 2D вектора в 1D массив (row-major)
std::vector<double> Tensor::to_flat_array(const std::vector<std::vector<double>>& matrix) const {
    std::vector<double> flat;
    flat.reserve(matrix.size() * matrix[0].size());
    for (const auto& row : matrix) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}
// Вспомогательный метод для конвертации 1D массива обратно в 2D вектор
void Tensor::from_flat_array(const std::vector<double>& flat, std::vector<std::vector<double>>& matrix, int rows, int cols) const {
    matrix.resize(rows);
    for (int i = 0; i < rows; i++) {
        matrix[i].resize(cols);
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = flat[i * cols + j];
        }
    }
}

// Оптимизированный matmul с OpenBLAS
Tensor Tensor::matmul(const Tensor& other) const 
{

    if (cols != other.rows) {
        throw std::runtime_error("Dimension mismatch in matmul: this.cols != other.rows");
    }
    if (data.empty() || other.data.empty()) {
        throw std::runtime_error("Empty data in matmul");
    }
    if (data[0].empty() || other.data[0].empty()) {
        throw std::runtime_error("Empty row data in matmul");
    }

    Tensor result(rows, other.cols, requires_grad);

    // Конвертируем матрицы в плоские массивы для OpenBLAS
    std::vector<double> A = to_flat_array(data);
    std::vector<double> B = to_flat_array(other.data);
    std::vector<double> C(rows * other.cols, 0.0);

    // Используем cblas_dgemm для матричного умножения
    // C = alpha * A * B + beta * C
    // A: [rows x cols], B: [cols x other.cols], C: [rows x other.cols]
    cblas_dgemm(CblasRowMajor,      // Порядок хранения (row-major)
        CblasNoTrans,       // A не транспонирована
        CblasNoTrans,       // B не транспонирована
        rows,               // M - количество строк A и C
        other.cols,         // N - количество столбцов B и C
        cols,               // K - количество столбцов A и строк B
        1.0,                // alpha
        A.data(),           // матрица A
        cols,               // LDA - leading dimension A
        B.data(),           // матрица B
        other.cols,         // LDB - leading dimension B
        0.0,                // beta
        C.data(),           // матрица C (результат)
        other.cols);        // LDC - leading dimension C

    // Проверяем на конечность результата
    for (size_t i = 0; i < C.size(); i++) {
        if (!std::isfinite(C[i])) {
            throw std::runtime_error("Non-finite value in matmul result");
        }
    }

    // Конвертируем результат обратно в 2D вектор
    from_flat_array(C, result.data, rows, other.cols);

    if (rows == 0 || other.cols == 0)
    {
        std::wcout << L"matmul result shape: [" << result.rows << ", " << result.cols << "]" << L"   matmul result sample value [0][0]: " << result.data[0][0] << std::endl;
    }

    return result;
}
Tensor Tensor::matmul_grad_data(const Tensor& other) const 
{
    if (cols != other.rows) {
        throw std::runtime_error("Dimension mismatch in matmul_grad_data: this.cols != other.rows");
    }
    if (grad.empty() || other.data.empty()) {
        throw std::runtime_error("Empty grad or data in matmul_grad_data");
    }

    Tensor result(rows, other.cols, true);

    // Конвертируем матрицы в плоские массивы
    std::vector<double> A = to_flat_array(grad);
    std::vector<double> B = to_flat_array(other.data);
    std::vector<double> C(rows * other.cols, 0.0);

    // Используем cblas_dgemm
    cblas_dgemm(CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        rows,
        other.cols,
        cols,
        1.0,
        A.data(),
        cols,
        B.data(),
        other.cols,
        0.0,
        C.data(),
        other.cols);

    // Проверяем на конечность
    for (size_t i = 0; i < C.size(); i++) {
        if (!std::isfinite(C[i])) {
            throw std::runtime_error("Non-finite value in matmul_grad_data result");
        }
    }

    // Конвертируем результат в grad
    from_flat_array(C, result.grad, rows, other.cols);

    if (rows == 0 || other.cols == 0)
    {
        std::wcout << L"matmul_grad_data result shape: [" << result.rows << ", " << result.cols << "] " << L"      result sample value [0][0]: " << result.grad[0][0] << std::endl;
    }

    return result;
}
Tensor Tensor::matmul_data_grad(const Tensor& other) const {
    if (cols != other.rows) {
        throw std::runtime_error("Dimension mismatch in matmul_data_grad: this.cols != other.rows");
    }
    if (data.empty() || other.grad.empty()) {
        throw std::runtime_error("Empty data or grad in matmul_data_grad");
    }

    Tensor result(rows, other.cols, true);

    // Конвертируем матрицы в плоские массивы
    std::vector<double> A = to_flat_array(data);
    std::vector<double> B = to_flat_array(other.grad);
    std::vector<double> C(rows * other.cols, 0.0);

    // Используем cblas_dgemm
    cblas_dgemm(CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        rows,
        other.cols,
        cols,
        1.0,
        A.data(),
        cols,
        B.data(),
        other.cols,
        0.0,
        C.data(),
        other.cols);

    // Проверяем на конечность
    for (size_t i = 0; i < C.size(); i++) {
        if (!std::isfinite(C[i])) {
            throw std::runtime_error("Non-finite value in matmul_data_grad result");
        }
    }

    // Конвертируем результат в grad
    from_flat_array(C, result.grad, rows, other.cols);

    if (rows == 0 || other.cols == 0)
    {
        std::wcout << L"matmul_data_grad result shape: [" << result.rows << ", " << result.cols << "] " << L"      result sample value [0][0]: " << result.grad[0][0] << std::endl;
    }

    return result;
}



// ====  помогает использовать все backward для обработки операции сложения   ====
void Tensor::backward()
{
    if (operation == "layer_norm" && parents.size() == 1)
    {
        backward_layer_norm(*this, *parents[0], *parents[0]);
    }
    else if (operation == "layer_norm_with_params" && parents.size() == 3)
    {
        Tensor& input = *parents[0];
        Tensor& gamma = *parents[1];
        Tensor& beta = *parents[2];

        const double eps = 1e-6;
        Tensor grad_input(input.rows, input.cols, input.requires_grad);
        grad_input.zero_grad();

        for (int i = 0; i < rows; i++) 
        {
            // Вычисляем среднее и дисперсию для входа
            double mean = 0.0;
            for (int j = 0; j < cols; j++) {
                mean += input.data[i][j];
            }
            mean /= cols;

            double variance = 0.0;
            for (int j = 0; j < cols; j++) {
                double diff = input.data[i][j] - mean;
                variance += diff * diff;
            }
            variance /= cols;
            double std_dev = sqrt(variance + eps);

            // Градиенты для gamma и beta
            if (gamma.requires_grad) {
                for (int j = 0; j < cols; j++) {
                    gamma.grad[0][j] += grad[i][j] * (input.data[i][j] - mean) / std_dev;
                }
            }
            if (beta.requires_grad) {
                for (int j = 0; j < cols; j++) {
                    beta.grad[0][j] += grad[i][j];
                }
            }

            // Градиент для входа
            if (input.requires_grad) {
                double sum_grad = 0.0;
                double sum_grad_x = 0.0;
                for (int j = 0; j < cols; j++) {
                    double x_norm = (input.data[i][j] - mean) / std_dev;
                    sum_grad += grad[i][j] * gamma.data[0][j];
                    sum_grad_x += grad[i][j] * gamma.data[0][j] * x_norm;
                }
                for (int j = 0; j < cols; j++) {
                    double x_norm = (input.data[i][j] - mean) / std_dev;
                    grad_input.grad[i][j] = gamma.data[0][j] * (grad[i][j] - (sum_grad_x / cols + x_norm * sum_grad / cols)) / std_dev;
                }
            }
        }

        if (input.requires_grad) {
            for (int i = 0; i < input.rows; i++) {
                for (int j = 0; j < input.cols; j++) {
                    input.grad[i][j] += grad_input.grad[i][j];
                }
            }
        }
    }
    else if (operation == "add" && parents.size() == 2)
    {
        backward_add(*this, *parents[0], *parents[1]);
    }
    else if (operation == "matmul" && parents.size() == 2)
    {
        Tensor grad_tensor1(parents[0]->rows, parents[0]->cols);
        Tensor grad_tensor2(parents[1]->rows, parents[1]->cols);
        backward_matmul(*this, *parents[0], *parents[1], grad_tensor1, grad_tensor2);
        if (parents[0]->requires_grad) {
            for (int i = 0; i < grad_tensor1.rows; i++) {
                for (int j = 0; j < grad_tensor1.cols; j++) {
                    parents[0]->grad[i][j] += grad_tensor1.data[i][j];
                }
            }
        }
        if (parents[1]->requires_grad) {
            for (int i = 0; i < grad_tensor2.rows; i++) {
                for (int j = 0; j < grad_tensor2.cols; j++) {
                    parents[1]->grad[i][j] += grad_tensor2.data[i][j];
                }
            }
        }
    }
    else if (operation == "scaled_attention" && parents.size() == 3)
    {
        Tensor grad_Q(parents[0]->rows, parents[0]->cols);
        Tensor grad_K(parents[1]->rows, parents[1]->cols);
        Tensor grad_V(parents[2]->rows, parents[2]->cols);
        MultiHeadAttention::backward_scaled_attention(*this, *parents[0], *parents[1], *parents[2], backward_cache_len, d_k, grad_Q, grad_K, grad_V);
        if (parents[0]->requires_grad) {
            for (int i = 0; i < grad_Q.rows; i++) {
                for (int j = 0; j < grad_Q.cols; j++) {
                    parents[0]->grad[i][j] += grad_Q.data[i][j];
                }
            }
        }
        if (parents[1]->requires_grad) {
            for (int i = 0; i < grad_K.rows; i++) {
                for (int j = 0; j < grad_K.cols; j++) {
                    parents[1]->grad[i][j] += grad_K.data[i][j];
                }
            }
        }
        if (parents[2]->requires_grad) {
            for (int i = 0; i < grad_V.rows; i++) {
                for (int j = 0; j < grad_V.cols; j++) {
                    parents[2]->grad[i][j] += grad_V.data[i][j];
                }
            }
        }

    }
}

// ==== 4 Транспонирование матрицы: нужно для обратного распространения и Attention ====
Tensor Tensor::transpose() const {
    Tensor result(cols, rows, requires_grad);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}
// ===  5 Универсальная функция сложения, по умолчанию false (безопасная), true - быстрее (рисковая)
Tensor Tensor::add(const Tensor& other, bool inplace) const
{
    if (DEBUG == 1) { wcout << L"\n\n===    START  Tensor::add : this shape [" << rows << ", " << cols << "], other shape [" << other.rows << ", " << other.cols << "]   ===" << endl; }

    if (inplace)
    {
        // Inplace версия - модифицируем текущий тензор
        const_cast<Tensor*>(this)->requires_grad = requires_grad || other.requires_grad;

        if (other.rows == 1 && other.cols == cols) {
            // Broadcasting для inplace
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double val = data[i][j] + other.data[0][j];
                    if (!isfinite(val)) {
                        wcout << L"Non-finite value detected in add (inplace): data[" << i << "][" << j << "]=" << data[i][j]  << L", other[0][" << j << "]=" << other.data[0][j] << endl;
                        throw runtime_error("Non-finite value in add (inplace)");
                    }
                    const_cast<Tensor*>(this)->data[i][j] = val;
                }
            }
        }
        else 
        {
            // Обычное сложение (без broadcasting)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double val = data[i][j] + other.data[i][j];
                    if (!isfinite(val)) {
                        wcout << L"Non-finite value detected in add (inplace): data[" << i << "][" << j << "]=" << data[i][j] << L", other[" << i << "][" << j << "]=" << other.data[i][j] << endl;
                        throw runtime_error("Non-finite value in add (inplace)");
                    }
                    const_cast<Tensor*>(this)->data[i][j] = val;
                }
            }
        }

        return *this;
    }
    else {
        // Обычная версия - создаем новый тензор
        Tensor result(rows, cols, requires_grad || other.requires_grad);

        if (other.rows == 1 && other.cols == cols) 
        {
            // Broadcasting
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double val = data[i][j] + other.data[0][j];
                    if (!isfinite(val)) {
                        wcout << L"Non-finite value detected in add: data[" << i << "][" << j << "]=" << data[i][j]  << L", other[0][" << j << "]=" << other.data[0][j] << endl;
                        throw runtime_error("Non-finite value in add");
                    }
                    result.data[i][j] = val;
                }
            }
        }
        else 
        {
            // Обычное сложение
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double val = data[i][j] + other.data[i][j];
                    if (!isfinite(val)) {
                        wcout << L"Non-finite value detected in add: data[" << i << "][" << j << "]=" << data[i][j]  << L", other[" << i << "][" << j << "]=" << other.data[i][j] << endl;
                        throw runtime_error("Non-finite value in add");
                    }
                    result.data[i][j] = val;
                }
            }
        }

        // Сохраняем информацию для backward pass
        if (result.requires_grad) {
            result.parents.push_back(const_cast<Tensor*>(this));
            result.parents.push_back(const_cast<Tensor*>(&other));
            result.operation = "add";
        }

        return result;
    }

    if (DEBUG == 1) { wcout << L"\n\n===    END  Tensor::add  END   ===" << endl; }

}


// ==== 8. Нормализация слоя: стабилизирует обучение, нужен для Transformer ====
Tensor Tensor::layer_norm() const 
{
    Tensor result(rows, cols, requires_grad);
    const double eps = 1e-6;

    // Forward pass
    for (int i = 0; i < rows; i++) {
        // Вычисляем среднее и дисперсию для каждой строки
        double mean = 0.0;
        for (int j = 0; j < cols; j++) {
            mean += data[i][j];
        }
        mean /= cols;

        double variance = 0.0;
        for (int j = 0; j < cols; j++) {
            double diff = data[i][j] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        double std_dev = sqrt(variance + eps);

        // Нормализуем
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = (data[i][j] - mean) / std_dev;
           
        }
    }

    // Сохраняем информацию для backward pass
    if (result.requires_grad) {
        result.parents.push_back(const_cast<Tensor*>(this));
        result.operation = "layer_norm";
    }

    return result;
}






////////////////////////// ===== MULTI-HEAD ATTENTION IMPLEMENTATION ===== //////////////////////////

// === 1. Конструктор: инициализация весов и параметров для Multi-Head Attention ===
MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model(d_model), num_heads(num_heads), d_k(d_model / num_heads),
    W_q(d_model, d_model, true),
    b_q(1, d_model, true),
    W_k(d_model, d_model, true),
    b_k(1, d_model, true),
    W_v(d_model, d_model, true),
    b_v(1, d_model, true),
    W_o(d_model, d_model, true),
    b_o(1, d_model, true)
{
    // Инициализация весов
    W_q.randomize();
    b_q.randomize();
    W_k.randomize();
    b_k.randomize();
    W_v.randomize();
    b_v.randomize();
    W_o.randomize();
    b_o.randomize();
}


// === 2.   Прямой проход с кешем: основной forward вызов Attention ===
Tensor MultiHeadAttention::forward(const Tensor& input, const vector<int>& input_ids, vector<int>& cached_input_ids, KVCache& cache, int layer_idx, bool use_cache)
{
    if (DEBUG == 1) {  wcout << L"\nSTART  ===  MultiHeadAttention::forward   ===  START" << endl; }

    // Проверка размеров весов
    if (b_q.rows != 1 || b_q.cols != d_model) {
        throw runtime_error("Invalid b_q dimensions: [" + to_string(b_q.rows) + ", " + to_string(b_q.cols) + "]");
    }

    // Сохраняем вход для backward
    cached_input = input;
    cached_input_ids = input_ids;

    // Вычисляем Q, K, V
    Tensor Q = input.matmul(W_q).add(b_q, true);
    Tensor K = input.matmul(W_k).add(b_k, true);
    Tensor V = input.matmul(W_v).add(b_v, true);

    // Сохраняем Q, K, V для backward
    cached_Q = Q;
    cached_K = K;
    cached_V = V;

    if (!use_cache)
    {
        // Прямой проход без кэша: используется для обучения
        vector<Tensor> Q_heads = split_heads(Q);
        vector<Tensor> K_heads = split_heads(K);
        vector<Tensor> V_heads = split_heads(V);

        // Attention для каждой головы
        vector<Tensor> attention_outputs;
        for (int h = 0; h < num_heads; h++) {
            Tensor head_output = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], 0);
            head_output.d_k = d_k; // Сохраняем d_k для каждой головы
            attention_outputs.push_back(head_output);
        }

        // Объединяем головы
        Tensor concat_output = merge_heads(attention_outputs);

        // Сохраняем concat_output для backward
        cached_concat_output = concat_output;

        // Выходная проекция
        if (b_o.rows != 1 || b_o.cols != d_model) {
            throw runtime_error("Invalid b_o dimensions: [" + to_string(b_o.rows) + ", " + to_string(b_o.cols) + "]");
        }
        Tensor output = concat_output.matmul(W_o).add(b_o, true);
        output.d_k = d_k; // Сохраняем d_k в финальном выходном тензоре


        if (DEBUG == 1) {  wcout << L"\nEND  ===  MultiHeadAttention::forward   ===  END" << endl; }

        return output;
    }
    else if(use_cache)
    {
        // Прямой проход с кэшем
        int seq_len = input.rows;
        int cache_len = cache.current_length[layer_idx]; // Используем длину для текущего слоя

        // Проверка корректности размеров кэша
        if (cache.keys[layer_idx].rows == 0 || cache.keys[layer_idx].cols == 0) 
        {
            wcout << L"Сбрасываем длину для этого слоя: " << layer_idx << endl;
            cache.keys[layer_idx] = Tensor(MAX_SEQ_LENGTH, d_model, true);
            cache.values[layer_idx] = Tensor(MAX_SEQ_LENGTH, d_model, true);
            cache.current_length[layer_idx] = 0; // Сбрасываем длину для этого слоя
            cache_len = 0;
        }

        // Проверка на переполнение кэша
        if (cache_len + seq_len > MAX_SEQ_LENGTH) 
        {
            wcout << L"ERROR: Cache overflow in MultiHeadAttention::forward, cache_len: " << cache_len << ", seq_len: " << seq_len << ", MAX_SEQ_LENGTH: " << MAX_SEQ_LENGTH << endl;
            throw runtime_error("Cache overflow in MultiHeadAttention::forward");
        }

        // Обновляем кэш
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_model; j++) {
                cache.keys[layer_idx].data[cache_len + i][j] = K.data[i][j];
                cache.values[layer_idx].data[cache_len + i][j] = V.data[i][j];
            }
        }

        // Обновляем current_length для текущего слоя
        cache.current_length[layer_idx] += seq_len;

        // Создаем полные K и V из кэша
        Tensor full_K(cache.current_length[layer_idx], d_model, true);
        Tensor full_V(cache.current_length[layer_idx], d_model, true);
        for (int i = 0; i < cache.current_length[layer_idx]; i++) {
            for (int j = 0; j < d_model; j++) {
                full_K.data[i][j] = cache.keys[layer_idx].data[i][j];
                full_V.data[i][j] = cache.values[layer_idx].data[i][j];
            }
        }

        // Разделяем на головы
        vector<Tensor> Q_heads = split_heads(Q);
        vector<Tensor> K_heads = split_heads(full_K);
        vector<Tensor> V_heads = split_heads(full_V);

        // Attention для каждой головы
        vector<Tensor> attention_outputs;
        for (int h = 0; h < num_heads; h++) {
            Tensor head_output = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], cache.current_length[layer_idx] - seq_len);
            head_output.d_k = d_k; // Сохраняем d_k для каждой головы
            attention_outputs.push_back(head_output);
        }

        // Объединяем головы
        Tensor concat_output = merge_heads(attention_outputs);

        // Сохраняем concat_output для backward
        cached_concat_output = concat_output;

        // Выходная проекция
        if (b_o.rows != 1 || b_o.cols != d_model) {
            throw runtime_error("Invalid b_o dimensions: [" + to_string(b_o.rows) + ", " + to_string(b_o.cols) + "]");
        }
        Tensor output = concat_output.matmul(W_o).add(b_o, true);
        output.d_k = d_k; // Сохраняем d_k в финальном выходном тензоре
        if (DEBUG == 1) {  wcout << L"\nEND === MultiHeadAttention::forward === END" << endl;  }
        return output;
    }
    throw runtime_error("Unexpected code path in MultiHeadAttention::forward");
}

// === 2.1   Scaled Dot-Product Attention с кешем: вычисляет внимание с сохранением прошлых ключей/значений ===
Tensor MultiHeadAttention::scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V, int cache_len)
{
    if (d_k == 0) 
    {
        std::wcout << L"\n\n Error: d_k is zero in scaled_dot_product_attention" << std::endl;
        throw std::runtime_error("d_k is zero");
    }

    Tensor scores = Q.matmul(K.transpose());

    // Масштабирование
    double scale = 1.0 / sqrt(d_k);
    for (int i = 0; i < scores.rows; i++) {
        for (int j = 0; j < scores.cols; j++) {
            scores.data[i][j] *= scale;
        }
    }

    // Каузальная маска (разная логика для кеша и без кеша)
    if (cache_len == 0) 
    {
        // Без кеша: классическая маска
        for (int i = 0; i < scores.rows; i++) {
            for (int j = i + 1; j < scores.cols; j++) {
                scores.data[i][j] = -1e9;
            }
        }
    }
    else {
        // С кешем: токены могут видеть только предыдущие
        for (int i = 0; i < scores.rows; i++) {
            for (int j = cache_len + i + 1; j < scores.cols; j++) {
                scores.data[i][j] = -1e9;
            }
        }
    }

    // Softmax
    for (int i = 0; i < scores.rows; i++) 
    {
        vector<double> softmax_scores = MathUtils::softmax(scores.data[i]);
        scores.data[i] = softmax_scores;
    }

    Tensor result = scores.matmul(V);

    // Сохраняем информацию для backward pass
    if (result.requires_grad) 
    {
        result.parents.push_back(const_cast<Tensor*>(&Q));
        result.parents.push_back(const_cast<Tensor*>(&K));
        result.parents.push_back(const_cast<Tensor*>(&V));
        result.operation = "scaled_attention";
        result.backward_cache_len = cache_len; // Сохраняем cache_len для backward
    }

    return result;
}


// === 2.3. Разделение матрицы на головы: подготовка Q, K, V для multi-head ===
vector<Tensor> MultiHeadAttention::split_heads(const Tensor& input) 
{
    if (DEBUG == 1) {  wcout << L"   ===   split_heads start   ===   " << endl;  }
    if (input.rows == 0 || input.cols == 0)
    {
        wcout << L"input shape: [" << input.rows << ", " << input.cols << "]" << L"   -   num_heads: " << num_heads << L", d_k: " << d_k << endl;
    }

    // Проверка размеров
    if (input.cols != d_model) {
        throw runtime_error("Input cols (" + to_string(input.cols) + ") must equal d_model (" + to_string(d_model) + ")");
    }
    if (input.cols % num_heads != 0) {
        throw runtime_error("Input cols (" + to_string(input.cols) + ") must be divisible by num_heads (" + to_string(num_heads) + ")");
    }
    if (input.data.empty() || input.data[0].empty()) {
        throw runtime_error("Input data is empty or not initialized");
    }

    vector<Tensor> heads;
    int seq_len = input.rows;

    for (int h = 0; h < num_heads; h++) {
        Tensor head(seq_len, d_k);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_k; j++) {
                int col_idx = h * d_k + j;
                if (col_idx >= input.cols) {
                    throw runtime_error("Index out of bounds: col_idx=" + to_string(col_idx) + ", input.cols=" + to_string(input.cols));
                }
                head.data[i][j] = input.data[i][col_idx];
            }
        }
        heads.push_back(head);
    }

    if (DEBUG == 1) {  wcout << L"   ===   Split_heads COMPLETED    heads.size: " << heads.size() << "\n" << endl; }
    return heads;

}

vector<Tensor> MultiHeadAttention::split_heads_grad(const Tensor& input)
{
    if (DEBUG == 1) { wcout << L"   ===   split_heads_grad start   ===   " << endl; }
    if (input.rows == 0 || input.cols == 0) 
    { 
        wcout << L"input shape: [" << input.rows << ", " << input.cols << "]" << L"   -   num_heads: " << num_heads << L", d_k: " << d_k << endl;  
    }

    // Проверка размеров
    if (input.cols != d_model) {
        throw runtime_error("Input cols (" + to_string(input.cols) + ") must equal d_model (" + to_string(d_model) + ")");
    }
    if (input.cols % num_heads != 0) {
        throw runtime_error("Input cols (" + to_string(input.cols) + ") must be divisible by num_heads (" + to_string(num_heads) + ")");
    }
    if (input.grad.empty() || input.grad[0].empty()) {
        throw runtime_error("Input grad is empty or not initialized");
    }

    vector<Tensor> heads;
    int seq_len = input.rows;

    for (int h = 0; h < num_heads; h++) {
        Tensor head(seq_len, d_k, input.requires_grad);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_k; j++) {
                int col_idx = h * d_k + j;
                if (col_idx >= input.cols) {
                    throw runtime_error("Index out of bounds: col_idx=" + to_string(col_idx) + ", input.cols=" + to_string(input.cols));
                }
                head.grad[i][j] = input.grad[i][col_idx];
            }
        }
        heads.push_back(head);
    }

    if (DEBUG == 1) {  wcout << L"   ===   Split_heads_grad COMPLETED    heads.size: " << heads.size() << "\n" << endl; }
    return heads;
}

// === 2.4. Объединение голов в одну матрицу: после attention ===
Tensor MultiHeadAttention::merge_heads(const vector<Tensor>& heads) {
    int seq_len = heads[0].rows;
    Tensor output(seq_len, d_model);

    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < d_k; j++) {
                output.data[i][h * d_k + j] = heads[h].data[i][j];
            }
        }
    }
    return output;
}
// === 4 Обнуление градиентов всех матриц весов === 
void MultiHeadAttention::zero_grad() 
{
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
}




void MultiHeadAttention::add_gradients_to_vector(vector<double>& all_gradients) {
    // Добавляем градиенты W_q
    for (int i = 0; i < W_q.rows; i++) {
        for (int j = 0; j < W_q.cols; j++) {
            all_gradients.push_back(W_q.grad[i][j]);
        }
    }

    // Добавляем градиенты W_k
    for (int i = 0; i < W_k.rows; i++) {
        for (int j = 0; j < W_k.cols; j++) {
            all_gradients.push_back(W_k.grad[i][j]);
        }
    }

    // Добавляем градиенты W_v
    for (int i = 0; i < W_v.rows; i++) {
        for (int j = 0; j < W_v.cols; j++) {
            all_gradients.push_back(W_v.grad[i][j]);
        }
    }

    // Добавляем градиенты W_o
    for (int i = 0; i < W_o.rows; i++) {
        for (int j = 0; j < W_o.cols; j++) {
            all_gradients.push_back(W_o.grad[i][j]);
        }
    }
}
void MultiHeadAttention::scale_gradients(double scale)
{
    // Масштабируем все градиенты
    for (int i = 0; i < W_q.rows; i++) {
        for (int j = 0; j < W_q.cols; j++) {
            W_q.grad[i][j] *= scale;
            W_k.grad[i][j] *= scale;
            W_v.grad[i][j] *= scale;
            W_o.grad[i][j] *= scale;
        }
    }
}
void MultiHeadAttention::apply_weight_decay(double weight_decay) {
    // Применяем weight decay
    for (int i = 0; i < W_q.rows; i++) {
        for (int j = 0; j < W_q.cols; j++) {
            W_q.grad[i][j] += weight_decay * W_q.data[i][j];
            W_k.grad[i][j] += weight_decay * W_k.data[i][j];
            W_v.grad[i][j] += weight_decay * W_v.data[i][j];
            W_o.grad[i][j] += weight_decay * W_o.data[i][j];
        }
    }
}


void MultiHeadAttention::backward(const Tensor& grad_output, Tensor& grad_input)
{
    if (DEBUG == 1) { wcout << L"\n    === START   MultiHeadAttention::backward     START ===  " << endl; }

    // Проверка кешированных данных
    if (cached_concat_output.rows == 0 || cached_concat_output.cols == 0) {
        throw runtime_error("cached_concat_output is not initialized");
    }
    if (cached_input.rows == 0 || cached_input.cols == 0) {
        throw runtime_error("cached_input is not initialized");
    }
    if (cached_Q.rows == 0 || cached_Q.cols == 0 || cached_K.rows == 0 || cached_K.cols == 0 || cached_V.rows == 0 || cached_V.cols == 0) {
        throw runtime_error("cached_Q, cached_K, or cached_V is not initialized");
    }

    // Backward через выходную проекцию W_o и b_o
    Tensor W_o_T = W_o.transpose();
    Tensor grad_concat = grad_output.matmul_grad_data(W_o_T);

    // Обновляем градиенты W_o и b_o
    Tensor grad_W_o = cached_concat_output.transpose().matmul_data_grad(grad_output);
    for (int i = 0; i < W_o.rows; i++) {
        for (int j = 0; j < W_o.cols; j++) {
            W_o.grad[i][j] += grad_W_o.grad[i][j];
        }
    }
    Tensor grad_b_o = grad_output; // Суммирование по оси
    for (int i = 0; i < b_o.rows; i++) {
        for (int j = 0; j < b_o.cols; j++) {
            b_o.grad[i][j] += grad_b_o.grad[i][j];
        }
    }


    // Backward через merge_heads - разделяем обратно на головки
    vector<Tensor> grad_heads = split_heads_grad(grad_concat);

    // Инициализируем градиенты для Q, K, V
    Tensor grad_Q(cached_Q.rows, cached_Q.cols, true);
    Tensor grad_K(cached_K.rows, cached_K.cols, true);
    Tensor grad_V(cached_V.rows, cached_V.cols, true);
    grad_Q.zero_grad();
    grad_K.zero_grad();
    grad_V.zero_grad();

    // Backward через каждую головку attention
    for (int head = 0; head < num_heads; head++) 
    {

        int head_start = head * d_k;
        int head_end = (head + 1) * d_k;

        Tensor head_Q(cached_Q.rows, d_k, true);
        Tensor head_K(cached_K.rows, d_k, true);
        Tensor head_V(cached_V.rows, d_k, true);

        // Копируем данные для головки
        for (int i = 0; i < cached_Q.rows; i++) {
            for (int j = 0; j < d_k; j++) {
                head_Q.data[i][j] = cached_Q.data[i][head_start + j];
                head_K.data[i][j] = cached_K.data[i][head_start + j];
                head_V.data[i][j] = cached_V.data[i][head_start + j];
            }
        }

        // Градиенты для этой головки
        Tensor grad_head_Q(head_Q.rows, head_Q.cols, true);
        Tensor grad_head_K(head_K.rows, head_K.cols, true);
        Tensor grad_head_V(head_V.rows, head_V.cols, true);
        grad_head_Q.zero_grad();
        grad_head_K.zero_grad();
        grad_head_V.zero_grad();

        // Backward через scaled dot-product attention
        backward_scaled_attention(grad_heads[head], head_Q, head_K, head_V,
            cached_input.rows, d_k, grad_head_Q, grad_head_K, grad_head_V);

        // Копируем градиенты обратно
        for (int i = 0; i < cached_Q.rows; i++) {
            for (int j = 0; j < d_k; j++) {
                grad_Q.grad[i][head_start + j] += grad_head_Q.grad[i][j];
                grad_K.grad[i][head_start + j] += grad_head_K.grad[i][j];
                grad_V.grad[i][head_start + j] += grad_head_V.grad[i][j];
            }
        }
    }

    // Backward через проекции Q, K, V
    if (DEBUG == 1) { wcout << L"\n  ==    Backward через проекции Q, K, V   ==  " << endl; }

    // Backward через проекции Q, K, V
    Tensor W_q_T = W_q.transpose();  Tensor grad_input_from_Q = grad_Q.matmul_grad_data(W_q_T);
    Tensor W_k_T = W_k.transpose();  Tensor grad_input_from_K = grad_K.matmul_grad_data(W_k_T);
    Tensor W_v_T = W_v.transpose();  Tensor grad_input_from_V = grad_V.matmul_grad_data(W_v_T);

    // Обновляем градиенты весов
    Tensor cached_input_T = cached_input.transpose();
    Tensor grad_W_q = cached_input_T.matmul_data_grad(grad_Q);
    Tensor grad_W_k = cached_input_T.matmul_data_grad(grad_K);
    Tensor grad_W_v = cached_input_T.matmul_data_grad(grad_V);

    for (int i = 0; i < W_q.rows; i++) {
        for (int j = 0; j < W_q.cols; j++) {
            W_q.grad[i][j] += grad_W_q.grad[i][j];
            W_k.grad[i][j] += grad_W_k.grad[i][j];
            W_v.grad[i][j] += grad_W_v.grad[i][j];
        }
    }
    // Градиенты для b_q, b_k, b_v
    for (int i = 0; i < b_q.rows; i++) {
        for (int j = 0; j < b_q.cols; j++) {
            b_q.grad[i][j] += grad_Q.grad[i][j];
            b_k.grad[i][j] += grad_K.grad[i][j];
            b_v.grad[i][j] += grad_V.grad[i][j];
        }
    }
    // Суммируем градиенты от Q, K, V
    grad_input = Tensor(cached_input.rows, cached_input.cols, true);
    grad_input.zero_grad();
    for (int i = 0; i < grad_input.rows; i++) {
        for (int j = 0; j < grad_input.cols; j++) {
            grad_input.grad[i][j] = grad_input_from_Q.grad[i][j] +
                grad_input_from_K.grad[i][j] +
                grad_input_from_V.grad[i][j];
        }
    }



    
    if (DEBUG == 1) { wcout << L"\n    === END   MultiHeadAttention::backward     END ===  \n\n" << endl; }

}
////////////////////////// ===== FEED FORWARD IMPLEMENTATION ===== //////////////////////////

// === 1. Конструктор: инициализация весов и смещений, случайная инициализация ===
FeedForward::FeedForward(int d_model, int d_ff)
    : W1(d_model, d_ff, true), W2(d_ff, d_model, true),
    b1(1, d_ff, true), b2(1, d_model, true)
{
    b1.randomize();
    b2.randomize();

    W1.randomize();
    W2.randomize();
}

// === 2. Прямой проход (forward): линейное преобразование → GELU → линейное преобразование с bias ===
Tensor FeedForward::forward(const Tensor& input)
{
    if (DEBUG == 1) {
        wcout << L"\n\n ===  FeedForward::forward start  === " << endl;
    }
    // Сохраняем вход для backward
    cached_input = input;

    // Проверка совместимости размеров
    if (input.cols != W1.rows) {
        throw runtime_error("Dimension mismatch in FeedForward::forward: input.cols != W1.rows");
    }
    if (b1.rows != 1 || b1.cols != W1.cols) {
        throw runtime_error("Dimension mismatch in FeedForward::forward: b1 shape incorrect");
    }

    // Первый линейный слой: hidden = input * W1 + b1
    Tensor hidden = input.matmul(W1).add(b1, true);

    // Применяем активацию GeLU
    for (int i = 0; i < hidden.rows; i++) {
        for (int j = 0; j < hidden.cols; j++) {
            hidden.data[i][j] = MathUtils::gelu(hidden.data[i][j]);
        }
    }

    // Сохраняем hidden для backward
    cached_hidden = hidden;


    if (hidden.cols != W2.rows) {
        throw runtime_error("Dimension mismatch in FeedForward::forward: hidden.cols != W2.rows");
    }
    if (b2.rows != 1 || b2.cols != W2.cols) {
        throw runtime_error("Dimension mismatch in FeedForward::forward: b2 shape incorrect");
    }

    Tensor output = hidden.matmul(W2).add(b2, true);
    return output;

    if (DEBUG == 1) {
        wcout << L"\n\n ===  FeedForward::forward end  === " << endl;
    }

}



void FeedForward::add_gradients_to_vector(vector<double>& all_gradients) {
    // Добавляем градиенты W1
    for (int i = 0; i < W1.rows; i++) {
        for (int j = 0; j < W1.cols; j++) {
            all_gradients.push_back(W1.grad[i][j]);
        }
    }

    // Добавляем градиенты W2
    for (int i = 0; i < W2.rows; i++) {
        for (int j = 0; j < W2.cols; j++) {
            all_gradients.push_back(W2.grad[i][j]);
        }
    }

    // Добавляем градиенты b1
    for (int i = 0; i < b1.rows; i++) {
        for (int j = 0; j < b1.cols; j++) {
            all_gradients.push_back(b1.grad[i][j]);
        }
    }

    // Добавляем градиенты b2
    for (int i = 0; i < b2.rows; i++) {
        for (int j = 0; j < b2.cols; j++) {
            all_gradients.push_back(b2.grad[i][j]);
        }
    }
}
void FeedForward::scale_gradients(double scale) {
    // Масштабируем градиенты W1
    for (int i = 0; i < W1.rows; i++) {
        for (int j = 0; j < W1.cols; j++) {
            W1.grad[i][j] *= scale;
        }
    }

    // Масштабируем градиенты W2
    for (int i = 0; i < W2.rows; i++) {
        for (int j = 0; j < W2.cols; j++) {
            W2.grad[i][j] *= scale;
        }
    }

    // Масштабируем градиенты b1
    for (int i = 0; i < b1.rows; i++) {
        for (int j = 0; j < b1.cols; j++) {
            b1.grad[i][j] *= scale;
        }
    }

    // Масштабируем градиенты b2
    for (int i = 0; i < b2.rows; i++) {
        for (int j = 0; j < b2.cols; j++) {
            b2.grad[i][j] *= scale;
        }
    }
}
void FeedForward::apply_weight_decay(double weight_decay) {
    // Применяем weight decay к W1
    for (int i = 0; i < W1.rows; i++) {
        for (int j = 0; j < W1.cols; j++) {
            W1.grad[i][j] += weight_decay * W1.data[i][j];
        }
    }

    // Применяем weight decay к W2
    for (int i = 0; i < W2.rows; i++) {
        for (int j = 0; j < W2.cols; j++) {
            W2.grad[i][j] += weight_decay * W2.data[i][j];
        }
    }

    // Применяем weight decay к b1
    for (int i = 0; i < b1.rows; i++) {
        for (int j = 0; j < b1.cols; j++) {
            b1.grad[i][j] += weight_decay * b1.data[i][j];
        }
    }

    // Применяем weight decay к b2
    for (int i = 0; i < b2.rows; i++) {
        for (int j = 0; j < b2.cols; j++) {
            b2.grad[i][j] += weight_decay * b2.data[i][j];
        }
    }
}
void FeedForward::zero_grad()
{
    W1.zero_grad();
    W2.zero_grad();
    b1.zero_grad();
    b2.zero_grad();
}


void FeedForward::backward(const Tensor& grad_output, Tensor& grad_input)
{
    if (DEBUG == 1) { wcout << L"\n\n>>> START    FeedForward::backward     START <<<" << endl; }

    // Инициализация градиентов
    grad_input = Tensor(cached_input.rows, cached_input.cols, true);
    grad_input.zero_grad();
    W2.zero_grad();
    b2.zero_grad();
    W1.zero_grad();
    b1.zero_grad();

    // Градиент через второй линейный слой: grad_hidden = grad_output * W2^T
    Tensor grad_hidden(cached_hidden.rows, cached_hidden.cols, true);
    grad_hidden.zero_grad();
    for (int i = 0; i < grad_output.rows; i++) {
        for (int j = 0; j < grad_output.cols; j++) {
            for (int k = 0; k < W2.rows; k++) {
                grad_hidden.grad[i][k] += grad_output.grad[i][j] * W2.data[k][j];
                W2.grad[k][j] += grad_output.grad[i][j] * cached_hidden.data[i][k];
            }
            b2.grad[0][j] += grad_output.grad[i][j];
        }
    }
    // Градиент через GeLU
    for (int i = 0; i < grad_hidden.rows; i++) {
        for (int j = 0; j < grad_hidden.cols; j++) {
            double x = cached_hidden.data[i][j];
            double gelu_deriv = MathUtils::gelu_derivative(x);
            grad_hidden.grad[i][j] *= gelu_deriv;
            if (i == 0 && j == 0 && (x == 0 || gelu_deriv == 0)) //вывод отладочной информации только первым элементом тензора ([0][0])  ну и когда равно 0.
            {
                wcout << L"GeLU deriv at [0][0]: x=" << x << L", deriv=" << gelu_deriv << endl;
            }
        }
    }

    // Градиент через первый линейный слой: grad_input = grad_hidden * W1^T
    for (int i = 0; i < grad_hidden.rows; i++) {
        for (int j = 0; j < W1.rows; j++) {
            for (int k = 0; k < grad_hidden.cols; k++) {
                grad_input.grad[i][j] += grad_hidden.grad[i][k] * W1.data[j][k];
                W1.grad[j][k] += grad_hidden.grad[i][k] * cached_input.data[i][j];
            }
            b1.grad[0][j] += grad_hidden.grad[i][j];
        }
    }

    if (DEBUG == 1) { wcout << L">>> END    FeedForward::backward     END <<<" << endl; }

}




////////////////////////// ===== TRANSFORMER LAYER IMPLEMENTATION ===== //////////////////////////

// === 1. Конструктор: инициализация Attention, FeedForward и параметров layer norm ===
// Конструктор по умолчанию для TransformerLayer
TransformerLayer::TransformerLayer() :
    attention(EMBEDDING_DIM, NUM_HEADS),
    feed_forward(EMBEDDING_DIM, FF_DIM),
    ln1_gamma(1, EMBEDDING_DIM, true),
    ln1_beta(1, EMBEDDING_DIM, true),
    ln2_gamma(1, EMBEDDING_DIM, true),
    ln2_beta(1, EMBEDDING_DIM, true)
{
    ln1_gamma.randomize();
    ln1_beta.zero();
    ln2_gamma.randomize();
    ln2_beta.zero();
}

// Конструктор с параметрами (без изменений)
TransformerLayer::TransformerLayer(int d_model, int num_heads, int d_ff) :
    attention(d_model, num_heads),
    feed_forward(d_model, d_ff),
    ln1_gamma(1, d_model, true),
    ln1_beta(1, d_model, true),
    ln2_gamma(1, d_model, true),
    ln2_beta(1, d_model, true)
{
    ln1_gamma.randomize();
    ln1_beta.zero();
    ln2_gamma.randomize();
    ln2_beta.zero();
}


// === 2. Прямой проход с кешем: полный слой (LayerNorm → Attention → Add → LayerNorm → FF → Add) ===
Tensor TransformerLayer::forward(const Tensor& input, const vector<int>& input_ids, vector<int>& cached_input_ids, KVCache& cache, int layer_idx, bool use_cache)
{
    if (DEBUG == 1) {  wcout << L"\n\n\n==== TransformerLayer forward (layer " << layer_idx << ")   bool cache = (" << use_cache << ")\n " ;  }

    if (!use_cache)
    {
        // Проверяем входные данные
        if (input.rows == 0 || input.cols == 0) {
            wcout << L"Input shape: [" << input.rows << ", " << input.cols << "]" << L" - sample value [0][0]: " << input.data[0][0] << endl;
            wcout << L"input_ids size: " << input_ids.size() << endl;
        }

        // Сохраняем вход для backward
        cached_input = input;

        // Pre-layer norm с применением ln1_gamma и ln1_beta
        Tensor norm1 = input.layer_norm();
        for (int i = 0; i < norm1.rows; i++) {
            for (int j = 0; j < norm1.cols; j++) {
                norm1.data[i][j] = norm1.data[i][j] * ln1_gamma.data[0][j] + ln1_beta.data[0][j];
            }
        }
        if (norm1.requires_grad) {
            norm1.parents.push_back(const_cast<Tensor*>(&input));
            norm1.parents.push_back(&ln1_gamma);
            norm1.parents.push_back(&ln1_beta);
            norm1.operation = "layer_norm_with_params";
        }
        cached_attn_input = norm1;

        // Self-attention без кеша
        Tensor attn_output = attention.forward(norm1, input_ids, cached_input_ids, cache, layer_idx, use_cache);
        attn_output.d_k = attention.d_k;
        cached_attn_output = attn_output;

        // Residual connection
        Tensor residual1 = input.add(attn_output, false);

        // Feed forward с применением ln2_gamma и ln2_beta
        Tensor norm2 = residual1.layer_norm();
        for (int i = 0; i < norm2.rows; i++) {
            for (int j = 0; j < norm2.cols; j++) {
                norm2.data[i][j] = norm2.data[i][j] * ln2_gamma.data[0][j] + ln2_beta.data[0][j];
            }
        }
        if (norm2.requires_grad) {
            norm2.parents.push_back(const_cast<Tensor*>(&residual1));
            norm2.parents.push_back(&ln2_gamma);
            norm2.parents.push_back(&ln2_beta);
            norm2.operation = "layer_norm_with_params";
        }
        cached_ff_input = norm2;

        Tensor ff_output = feed_forward.forward(norm2);

        // Second residual connection
        Tensor residual2 = residual1.add(ff_output, true);


        return residual2;


    }
    else
    {

        // Случай с кешем
        if (input.rows == 0 || input.cols == 0) {
            wcout << L"Input shape (cached): [" << input.rows << ", " << input.cols << "]" << L" - sample value [0][0]: " << input.data[0][0] << endl;
            wcout << L"input_ids size (cached): " << input_ids.size() << endl;
        }

        cached_input = input;
        Tensor norm1 = input.layer_norm();
        for (int i = 0; i < norm1.rows; i++) {
            for (int j = 0; j < norm1.cols; j++) {
                norm1.data[i][j] = norm1.data[i][j] * ln1_gamma.data[0][j] + ln1_beta.data[0][j];
            }
        }
        if (norm1.requires_grad) {
            norm1.parents.push_back(const_cast<Tensor*>(&input));
            norm1.parents.push_back(&ln1_gamma);
            norm1.parents.push_back(&ln1_beta);
            norm1.operation = "layer_norm_with_params";
        }
        cached_attn_input = norm1;

        Tensor attn_output = attention.forward(norm1, input_ids, cached_input_ids, cache, layer_idx, use_cache);
        attn_output.d_k = attention.d_k;
        cached_attn_output = attn_output;

        Tensor residual1 = input.add(attn_output, false);

        Tensor norm2 = residual1.layer_norm();
        for (int i = 0; i < norm2.rows; i++) {
            for (int j = 0; j < norm2.cols; j++) {
                norm2.data[i][j] = norm2.data[i][j] * ln2_gamma.data[0][j] + ln2_beta.data[0][j];
            }
        }
        if (norm2.requires_grad) {
            norm2.parents.push_back(const_cast<Tensor*>(&residual1));
            norm2.parents.push_back(&ln2_gamma);
            norm2.parents.push_back(&ln2_beta);
            norm2.operation = "layer_norm_with_params";
        }
        cached_ff_input = norm2;

        Tensor ff_output = feed_forward.forward(norm2);

        Tensor residual2 = residual1.add(ff_output, true);

        return residual2;
    }
}


void TransformerLayer::add_gradients_to_vector(vector<double>& all_gradients) {
    // Добавляем градиенты attention слоя
    attention.add_gradients_to_vector(all_gradients);

    // Добавляем градиенты feed_forward слоя
    feed_forward.add_gradients_to_vector(all_gradients);

    // Добавляем градиенты ln1_gamma
    for (int i = 0; i < ln1_gamma.rows; i++) {
        for (int j = 0; j < ln1_gamma.cols; j++) {
            all_gradients.push_back(ln1_gamma.grad[i][j]);
        }
    }

    // Добавляем градиенты ln1_beta
    for (int i = 0; i < ln1_beta.rows; i++) {
        for (int j = 0; j < ln1_beta.cols; j++) {
            all_gradients.push_back(ln1_beta.grad[i][j]);
        }
    }

    // Добавляем градиенты ln2_gamma
    for (int i = 0; i < ln2_gamma.rows; i++) {
        for (int j = 0; j < ln2_gamma.cols; j++) {
            all_gradients.push_back(ln2_gamma.grad[i][j]);
        }
    }

    // Добавляем градиенты ln2_beta
    for (int i = 0; i < ln2_beta.rows; i++) {
        for (int j = 0; j < ln2_beta.cols; j++) {
            all_gradients.push_back(ln2_beta.grad[i][j]);
        }
    }
}
void TransformerLayer::scale_gradients(double scale) {
    // Масштабируем градиенты attention слоя
    attention.scale_gradients(scale);

    // Масштабируем градиенты feed_forward слоя
    feed_forward.scale_gradients(scale);

    // Масштабируем градиенты ln1_gamma
    for (int i = 0; i < ln1_gamma.rows; i++) {
        for (int j = 0; j < ln1_gamma.cols; j++) {
            ln1_gamma.grad[i][j] *= scale;
        }
    }

    // Масштабируем градиенты ln1_beta
    for (int i = 0; i < ln1_beta.rows; i++) {
        for (int j = 0; j < ln1_beta.cols; j++) {
            ln1_beta.grad[i][j] *= scale;
        }
    }

    // Масштабируем градиенты ln2_gamma
    for (int i = 0; i < ln2_gamma.rows; i++) {
        for (int j = 0; j < ln2_gamma.cols; j++) {
            ln2_gamma.grad[i][j] *= scale;
        }
    }

    // Масштабируем градиенты ln2_beta
    for (int i = 0; i < ln2_beta.rows; i++) {
        for (int j = 0; j < ln2_beta.cols; j++) {
            ln2_beta.grad[i][j] *= scale;
        }
    }
}
void TransformerLayer::apply_weight_decay(double weight_decay) {
    // Применяем weight decay к attention слою
    attention.apply_weight_decay(weight_decay);

    // Применяем weight decay к feed_forward слою
    feed_forward.apply_weight_decay(weight_decay);

    // Применяем weight decay к ln1_gamma
    for (int i = 0; i < ln1_gamma.rows; i++) {
        for (int j = 0; j < ln1_gamma.cols; j++) {
            ln1_gamma.grad[i][j] += weight_decay * ln1_gamma.data[i][j];
        }
    }

    // Применяем weight decay к ln1_beta
    for (int i = 0; i < ln1_beta.rows; i++) {
        for (int j = 0; j < ln1_beta.cols; j++) {
            ln1_beta.grad[i][j] += weight_decay * ln1_beta.data[i][j];
        }
    }

    // Применяем weight decay к ln2_gamma
    for (int i = 0; i < ln2_gamma.rows; i++) {
        for (int j = 0; j < ln2_gamma.cols; j++) {
            ln2_gamma.grad[i][j] += weight_decay * ln2_gamma.data[i][j];
        }
    }

    // Применяем weight decay к ln2_beta
    for (int i = 0; i < ln2_beta.rows; i++) {
        for (int j = 0; j < ln2_beta.cols; j++) {
            ln2_beta.grad[i][j] += weight_decay * ln2_beta.data[i][j];
        }
    }
}


Tensor TransformerLayer::backward(const Tensor& grad_output)
{
    if (DEBUG == 1) { wcout << L"\n\n-->>>   TransformerLayer backward start   <<<--" << endl; }
    if (grad_output.rows == 0 || grad_output.cols == 0) {
        wcout << L"grad_output shape: [" << grad_output.rows << ", " << grad_output.cols << "]" << L" - sample value [0][0]: " << grad_output.grad[0][0] << endl;
    }

    // Градиент через второй residual connection
    Tensor grad_ff_output(grad_output.rows, grad_output.cols, true);
    for (int i = 0; i < grad_output.rows; i++) {
        for (int j = 0; j < grad_output.cols; j++) {
            grad_ff_output.grad[i][j] = grad_output.grad[i][j];
        }
    }
    Tensor grad_residual1(grad_output.rows, grad_output.cols, true);
    for (int i = 0; i < grad_output.rows; i++) {
        for (int j = 0; j < grad_output.cols; j++) {
            grad_residual1.grad[i][j] = grad_output.grad[i][j];
        }
    }

    // Backward через второй layer norm с gamma и beta
    Tensor grad_ff_input(cached_ff_input.rows, cached_ff_input.cols, true);
    Tensor::backward_layer_norm_with_params(grad_ff_output, cached_ff_input, ln2_gamma, ln2_beta,
        grad_ff_input, ln2_gamma, ln2_beta);

    // Backward через feed forward
    Tensor grad_after_attn(cached_attn_output.rows, cached_attn_output.cols, true);
    feed_forward.backward(grad_ff_input, grad_after_attn);

    // Складываем градиенты от residual connection
    for (int i = 0; i < grad_after_attn.rows; i++) {
        for (int j = 0; j < grad_after_attn.cols; j++) {
            grad_after_attn.grad[i][j] += grad_residual1.grad[i][j];
        }
    }

    // Градиент через первый residual connection
    Tensor grad_attn_output = grad_after_attn;
    Tensor grad_input_residual = grad_after_attn;

    // Backward через первый layer norm с gamma и beta
    Tensor grad_attn_input(cached_attn_input.rows, cached_attn_input.cols, true);
    Tensor::backward_layer_norm_with_params(grad_attn_output, cached_attn_input, ln1_gamma, ln1_beta,
        grad_attn_input, ln1_gamma, ln1_beta);

    // Backward через attention
    Tensor grad_input_from_attn(cached_input.rows, cached_input.cols, true);
    attention.backward(grad_attn_input, grad_input_from_attn);

    // Складываем градиенты от первого residual connection
    Tensor grad_input(cached_input.rows, cached_input.cols, true);
    for (int i = 0; i < grad_input.rows; i++) {
        for (int j = 0; j < grad_input.cols; j++) {
            grad_input.grad[i][j] = grad_input_from_attn.grad[i][j] + grad_input_residual.grad[i][j];
        }
    }


    if (DEBUG == 1) { wcout << L"-->>>   TransformerLayer backward end   <<<--\n" << endl; }
    return grad_input;
}





////////////////////////// ===== POSITIONAL ENCODING IMPLEMENTATION ===== //////////////////////////

// === 1. Конструктор: вычисление и сохранение позиционных энкодингов (синусы и косинусы) ===
PositionalEncoding::PositionalEncoding(int max_len, int d_model) : encoding(max_len, d_model) {
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            if (i % 2 == 0) {
                encoding.data[pos][i] = sin(pos / pow(10000.0, 2.0 * i / d_model));
            }
            else {
                encoding.data[pos][i] = cos(pos / pow(10000.0, 2.0 * (i - 1) / d_model));
            }
        }
    }
}
// === 2. Добавление позиционного энкодинга к эмбеддингам ===
Tensor PositionalEncoding::add_positional_encoding(const Tensor& embeddings) {
    Tensor result = embeddings;
    for (int i = 0; i < min(embeddings.rows, encoding.rows); i++) {
        for (int j = 0; j < embeddings.cols; j++) {
            result.data[i][j] += encoding.data[i][j];
        }
    }
    return result;
}





////////////////////////// ===== ADVANCED TRANSFORMER MODEL IMPLEMENTATION ===== //////////////////////////

// === 1. Конструктор: инициализация эмбеддингов, позиционного кодирования, слоев, оптимизатора Adam ===
TransformerModel::TransformerModel() :
    pos_encoding(MAX_SEQ_LENGTH, EMBEDDING_DIM),
    output_projection(EMBEDDING_DIM, 1, true),   // Будет переинициализировано
    learning_rate(0.0001),
    beta1(0.9),
    beta2(0.999),
    epsilon(1e-8),
    step_count(0)
{

    tokenizer = make_unique<AdvancedTokenizer>();
    for (int i = 0; i < NUM_LAYERS; i++) {
        layers.emplace_back(EMBEDDING_DIM, NUM_HEADS, FF_DIM); // Используем emplace_back вместо resize
    }
    generation_cache.resize_for_layers(NUM_LAYERS, MAX_SEQ_LENGTH, EMBEDDING_DIM);




    wcout << L" Создание продвинутой Transformer модели..." << endl;
    wcout << L" Параметры:" << endl;
    wcout << L"   - Размерность эмбеддингов: " << EMBEDDING_DIM << endl;
    wcout << L"   - Количество голов внимания: " << NUM_HEADS << endl;
    wcout << L"   - Размерность FF слоя: " << FF_DIM << endl;
    wcout << L"   - Количество слоев: " << NUM_LAYERS << endl;
    wcout << L"   - Максимальная длина: " << MAX_SEQ_LENGTH << endl;
}


                   /*  ДЛЯ ОБУЧЕНИЕ   */
// === 1. Валидация модели на тексте (вычисление средней потери) для основного файла тренировки ===
double TransformerModel::validate(const vector<string>& val_texts)
{
    if (DEBUG == 1) { wcout << L"\n\n--> >>   start   TransformerModel::validate   start   << <--" << endl; }


    double total_loss = 0.0;
    int valid_examples = 0;

    for (const string& text : val_texts) {
        vector<int> tokens = tokenizer->encode(text);
        if (tokens.size() < 3) continue;

        vector<int> input_ids(tokens.begin(), tokens.end() - 1);
        vector<int> target_ids(tokens.begin() + 1, tokens.end());

        Tensor logits = forward(input_ids);

        double loss = 0.0;
        for (int i = 0; i < min((int)target_ids.size(), logits.rows); i++) {
            if (target_ids[i] < logits.cols) {
                vector<double> logit_row = logits.data[i];
                vector<double> probs = MathUtils::softmax(logit_row);
                loss -= log(max(probs[target_ids[i]], 1e-10));
            }
        }

        loss /= min((int)target_ids.size(), logits.rows);
        total_loss += loss;
        valid_examples++;
    }

    if (DEBUG == 1) { wcout << L"\n\n--> >>   end   TransformerModel::validate   end   << <--" << endl; }

    return valid_examples > 0 ? total_loss / valid_examples : 1e6;
}

// === 2. Обучающий шаг для основного файла тренировки ===
void TransformerModel::train_step(const vector<string>& batch_texts)
{
    wcout << L"- - - Тренировочный шаг на "; cout << batch_texts.size(); wcout << L" примерах - - - \n" << endl;

    double total_loss = 0.0;
    int valid_examples = 0;

    // Обнуляем градиенты ПЕРЕД forward pass
    embeddings.zero_grad();
    output_projection.zero_grad();
    for (auto& layer : layers) 
    {
        layer.ln1_gamma.zero_grad();
        layer.ln1_beta.zero_grad();
        layer.ln2_gamma.zero_grad();
        layer.ln2_beta.zero_grad();
    }
    for (auto& layer : layers)
    {
        layer.attention.zero_grad();
        layer.feed_forward.zero_grad();
    }


    for (const string& text : batch_texts)
    {
        vector<int> tokens = tokenizer->encode(text);

        // Prepare input and target
        vector<int> input_ids(tokens.begin(), tokens.end() - 1);
        vector<int> target_ids(tokens.begin() + 1, tokens.end());
        if (DEBUG == 1) {  wcout << L"\nTarget size: "; cout << target_ids.size() << endl;  }
        // Forward pass
        Tensor logits = forward(input_ids);

        if (DEBUG == 1) 
        {
            cout << "\n ------------- \nLogits[0]:\n ";
            for (int i = 0; i < logits.rows; ++i)
            {
                cout << "Row " << i << ": ";
                for (int j = 0; j < min(10, logits.cols); ++j)
                {
                    cout << logits.data[i][j] << " ";
                }
                cout << endl;
            }
        }


        // Calculate loss
        double loss = calculate_loss_and_gradients(logits, target_ids);
        if (loss > 0)
        { // Проверяем что loss валидный
            total_loss += loss;
            valid_examples++;
        }

    }



    if (valid_examples > 0) 
    {
        total_loss /= valid_examples;
        wcout << L"TOTAL LOSS :::::: "; cout << fixed << setprecision(6) << total_loss << endl;

        // Нормализуем градиенты на размер батча
        normalize_gradients(valid_examples);

        // Увеличиваем step_count перед обновлением параметров
        step_count++;
        wcout << L"--------------------Параметры обновляются (шаг " << step_count << ")--------------" << endl;

        if (DEBUG == 1) {
            wcout << L"\n     update_with_adam(*param):   " << endl;
        }

        for (auto& layer : layers) {
            // Собираем все параметры слоя
            vector<Tensor*> layer_params;
            layer_params.push_back(&layer.attention.W_q);
            layer_params.push_back(&layer.attention.W_k);
            layer_params.push_back(&layer.attention.W_v);
            layer_params.push_back(&layer.attention.W_o);
            layer_params.push_back(&layer.attention.b_q);
            layer_params.push_back(&layer.attention.b_k);
            layer_params.push_back(&layer.attention.b_v);
            layer_params.push_back(&layer.attention.b_o);
            layer_params.push_back(&layer.feed_forward.W1);
            layer_params.push_back(&layer.feed_forward.W2);
            layer_params.push_back(&layer.feed_forward.b1);
            layer_params.push_back(&layer.feed_forward.b2);
            layer_params.push_back(&layer.ln1_gamma);
            layer_params.push_back(&layer.ln1_beta);
            layer_params.push_back(&layer.ln2_gamma);
            layer_params.push_back(&layer.ln2_beta);

            if (DEBUG == 1)
            {
                // Отладочный вывод градиентов
                wcout << L"\n------------Проверка градиентов для слоя - " << &layer - &layers[0] << L"-----------" << endl;
                for (Tensor* param : layer_params) {
                    if (param->requires_grad) {
                        double grad_sum = 0.0;
                        double grad_max = 0.0;
                        for (int i = 0; i < param->rows; i++) {
                            for (int j = 0; j < param->cols; j++) {
                                double grad = param->grad[i][j];
                                grad_sum += abs(grad);
                                grad_max = max(grad_max, abs(grad));
                            }
                        }
                        wcout << L"  Параметр: " << param << L", requires_grad: " << param->requires_grad << L", grad_sum: " << grad_sum << L", grad_max: " << grad_max << endl;
                    }
                    else
                    {
                        wcout << L"  Параметр: " << param << L", requires_grad: false (градиенты не вычисляются)" << endl;
                    }
                }
            }

            // Обновление параметров
            for (Tensor* param : layer_params) {
                if (param->requires_grad) {
                    update_with_adam(*param);
                }
            }
        }

        // Обновление embeddings и output_projection
        if (DEBUG == 1) { wcout << L"\n     update_with_adam(embeddings):   " << endl; }
        update_with_adam(embeddings);
        if (DEBUG == 1) { wcout << L"\n     update_with_adam(output_projection):   " << endl; }
        update_with_adam(output_projection);
    }
    else
    {
        wcout << L"\nПредупреждение: нет валидных примеров в батче!" << endl;
    }

}


// === 2.1     Вычисление loss и градиентов для логитов (calculate_loss_and_gradients) ===
double TransformerModel::calculate_loss_and_gradients(Tensor& logits, const vector<int>& target_ids)
{

    if (DEBUG == 1) {  wcout << L"\n___ calculate_loss_and_gradients ___\n " << endl;  }


    double total_loss = 0.0;
    int valid_positions = 0;
    // Вычисляем loss и градиенты для каждой позиции
    for (int i = 0; i < min((int)target_ids.size(), logits.rows); i++)
    {
        if (target_ids[i] >= 0 && target_ids[i] < logits.cols)
        {
            // Получаем logits для текущей позиции
            vector<double> logit_row(logits.cols);
            for (int j = 0; j < logits.cols; j++) {
                logit_row[j] = logits.data[i][j];
            }

            // Вычисляем softmax
            vector<double> probs = MathUtils::softmax(logit_row);


            // Cross-entropy loss
            double pos_loss = -log(max(probs[target_ids[i]], 1e-10));
            total_loss += pos_loss;
            valid_positions++;

            // Вычисляем градиенты для logits (∂L/∂logits)
            for (int j = 0; j < logits.cols; j++) {
                double target = (j == target_ids[i]) ? 1.0 : 0.0;
                logits.grad[i][j] = probs[j] - target;
            }
        }
    }


    if (DEBUG == 1)
    {
        // После цикла вычисления градиентов
        double grad_sum = 0.0;
        for (int i = 0; i < logits.rows; i++)
        {
            for (int j = 0; j < logits.cols; j++)
            {
                grad_sum += abs(logits.grad[i][j]);
            }
        }
        wcout << L"___ Logits grad sum after calculate_loss_and_gradients: " << grad_sum << L"Logits grad[0][0]: " << logits.grad[0][0] << endl;
    }

    if (valid_positions > 0) 
    {
        total_loss /= valid_positions;

        // Propagate градиенты обратно к параметрам
        backward_propagate_gradients(logits);
    }

    return total_loss;
}

// === 2.1.1   Распространение градиентов назад через output_projection и embeddings ===
void TransformerModel::backward_propagate_gradients(const Tensor& logits) {
    if (DEBUG == 1) {
        wcout << L"\n\n-------------------------------------------------" << endl;
        wcout << L">>> START   backward_propagate_gradients   START <<<" << endl;
    }

    // // // // // DEBUG // // 
    {
        // Проверка градиентов logits
        double logits_grad_sum = 0.0;
        for (int i = 0; i < logits.rows; i++) {
            for (int j = 0; j < logits.cols; j++) {
                logits_grad_sum += abs(logits.grad[i][j]);
            }
        }
        if (logits_grad_sum == 0)
        {
            wcout << L"Logits grad sum: " << logits_grad_sum << endl;
        }
    }


    // Градиенты для output_projection
    Tensor logits_grad_tensor(logits.rows, logits.cols, true);
    for (int i = 0; i < logits.rows; i++) {
        for (int j = 0; j < logits.cols; j++) {
            logits_grad_tensor.grad[i][j] = logits.grad[i][j];
        }
    }

    Tensor hidden_T = last_hidden_states.transpose();
    Tensor dW_out = hidden_T.matmul_data_grad(logits_grad_tensor);
    for (int i = 0; i < output_projection.rows; i++) {
        for (int j = 0; j < output_projection.cols; j++) {
            output_projection.grad[i][j] += dW_out.grad[i][j];
        }
    }

    Tensor W_out_T = output_projection.transpose();
    Tensor dHidden = logits_grad_tensor.matmul_grad_data(W_out_T);

    // // // // // DEBUG // // 
    {
        // Логирование суммы градиентов dHidden
        double dHidden_grad_sum = 0.0;
        for (int i = 0; i < dHidden.rows; i++) {
            for (int j = 0; j < dHidden.cols; j++) {
                dHidden_grad_sum += abs(dHidden.grad[i][j]);
            }
        }

        if (dHidden_grad_sum == 0) {
            wcout << L"dHidden grad sum: " << dHidden_grad_sum << endl;
        }
    }


    // Обратный проход через все слои Transformer
    Tensor grad_input = dHidden;
    for (int layer_idx = NUM_LAYERS - 1; layer_idx >= 0; layer_idx--) 
    {
        if (DEBUG == 1)  { wcout << L"Backward through layer " << layer_idx << endl;  }
        grad_input = layers[layer_idx].backward(grad_input);

        // // // // // DEBUG // // 
        {
            // Логирование суммы градиентов после каждого слоя
            double layer_grad_sum = 0.0;
            for (int i = 0; i < grad_input.rows; i++) {
                for (int j = 0; j < grad_input.cols; j++) {
                    layer_grad_sum += abs(grad_input.grad[i][j]);
                }
            }

            if (layer_grad_sum == 0) {
                wcout << L"Layer " << layer_idx << L" grad_input sum: " << layer_grad_sum << endl;
            }
        }
    }

    // Градиенты для embeddings
    for (int i = 0; i < grad_input.rows; i++) {
        int token_id = cached_input_ids[i];
        if (token_id >= 0 && token_id < embeddings.rows) {
            for (int j = 0; j < embeddings.cols; j++) {
                embeddings.grad[token_id][j] += grad_input.grad[i][j];
            }
        }
    }

    // Градиенты для positional encoding (если requires_grad=true)
    if (pos_encoding.encoding.requires_grad) {
        for (int i = 0; i < grad_input.rows; i++) {
            for (int j = 0; j < grad_input.cols; j++) {
                pos_encoding.encoding.grad[i][j] += grad_input.grad[i][j];
            }
        }
    }

    // // // // // DEBUG // // 
    {
        // Логирование градиентов для проверки
        double embeddings_grad_sum = 0.0;
        for (int i = 0; i < embeddings.rows; i++) {
            for (int j = 0; j < embeddings.cols; j++) {
                embeddings_grad_sum += abs(embeddings.grad[i][j]);
            }
        }
        if (embeddings_grad_sum == 0) {
            wcout << L"Embeddings grad sum: " << embeddings_grad_sum << endl;
        }
    }

    if (DEBUG == 1) { wcout << L"\nEND    Backward propagation   END\n\n" << endl; }
}

// === 2.2. Обновление параметров Adam-оптимизатором (update_with_adam) ===
void TransformerModel::update_with_adam(Tensor& param) {

    if (DEBUG == 1) { wcout << L"[]     update_with_adam     []" << endl; }

    // Проверка requires_grad
    if (!param.requires_grad) 
    {
        if (DEBUG == 1) { wcout << L"     []   ПРОПУЩЕН   []" << endl; }
        return; // Пропускаем, если градиенты не нужны
    }


    // // // // // DEBUG // // 
    {
        // Вычисление суммы градиентов для отладки
        double grad_sum = 0.0;
        for (int i = 0; i < param.rows; i++) {
            for (int j = 0; j < param.cols; j++) {
                if (!isfinite(param.grad[i][j])) {
                    wcout << L"NaN detected in grad at [" << i << "][" << j << "] = " << param.grad[i][j] << endl;
                    throw runtime_error("NaN in input gradient");
                }
                grad_sum += abs(param.grad[i][j]);
            }
        }
        if (grad_sum == 0) {
            wcout << L"Param grad sum before Adam update: " << grad_sum << endl;
        }
    

      // Проверка step_count
      if (step_count <= 0) 
    {
        wcout << L"Invalid step_count: " << step_count << endl;
        throw runtime_error("Invalid step_count in Adam");
    }

    }

    // Инициализация состояния Adam
    if (m_embeddings.find(&param) == m_embeddings.end()) 
    {
        m_embeddings[&param] = Tensor(param.rows, param.cols);
        v_embeddings[&param] = Tensor(param.rows, param.cols);
        m_embeddings[&param].zero(); // Инициализация нулями
        v_embeddings[&param].zero(); // Инициализация нулями
    }

    Tensor& m = m_embeddings[&param];
    Tensor& v = v_embeddings[&param];

    // Проверка размеров тензоров
    if (param.rows != m.rows || param.cols != m.cols || param.rows != v.rows || param.cols != v.cols) {
        throw runtime_error("Mismatched tensor dimensions in Adam update");
    }

    // Adam update
    for (int i = 0; i < param.rows; i++) {
        for (int j = 0; j < param.cols; j++) {
            // Обновление m и v
            m.data[i][j] = beta1 * m.data[i][j] + (1 - beta1) * param.grad[i][j];
            v.data[i][j] = beta2 * v.data[i][j] + (1 - beta2) * param.grad[i][j] * param.grad[i][j];

            //DEBUG   Проверка m и v 
            if (!isfinite(m.data[i][j]) || !isfinite(v.data[i][j])) {
                wcout << L"NaN detected after m/v update: m=" << m.data[i][j] << ", v=" << v.data[i][j] << endl;
                throw runtime_error("NaN in m or v");
            }

            // Коррекция смещения
            double m_hat = m.data[i][j] / (1 - pow(beta1, step_count));
            double v_hat = v.data[i][j] / (1 - pow(beta2, step_count));

            //DEBUG   Проверка на отрицательное v_hat
            if (v_hat < 0) {
                wcout << L"Negative v_hat detected: v_hat=" << v_hat << endl;
                throw runtime_error("Negative v_hat in Adam");
            }

            double denom = sqrt(v_hat) + epsilon;

            //DEBUG   Проверка на NaN
            if (!isfinite(m_hat) || !isfinite(v_hat) || !isfinite(denom)) {
                wcout << L"NaN detected in Adam: m_hat=" << m_hat << ", v_hat=" << v_hat << ", denom=" << denom << endl;
                throw runtime_error("NaN in Adam computation");
            }

            // Обновление параметра
            param.data[i][j] -= learning_rate * m_hat / denom;

            //DEBUG   Проверка параметра после обновления
            if (!isfinite(param.data[i][j])) {
                wcout << L"NaN detected in param after update at [" << i << "][" << j << "]" << endl;
                throw runtime_error("NaN in param after Adam update");
            }
        }
    }

    if (DEBUG == 1) { wcout << L"---------\n" << endl; }

}



// === 2.3. Нормализация градиентов на размер батча (normalize_gradients) ===
void TransformerModel::normalize_gradients(int batch_size)
{
    if (DEBUG == 1) { wcout << L" \n\n  === START   TransformerModel::normalize_gradients  START ===" << endl; }
    double scale = 1.0 / batch_size; // Масштабирование по размеру батча

    // Масштабируем градиенты для каждого слоя
    for (auto& layer : layers) {
        layer.scale_gradients(scale);
    }

    // Масштабируем градиенты эмбеддингов
    if (embeddings.requires_grad) {
        for (int i = 0; i < embeddings.rows; i++) {
            for (int j = 0; j < embeddings.cols; j++) {
                embeddings.grad[i][j] *= scale;
            }
        }
    }

    // Масштабируем градиенты выходной проекции
    if (output_projection.requires_grad) {
        for (int i = 0; i < output_projection.rows; i++) {
            for (int j = 0; j < output_projection.cols; j++) {
                output_projection.grad[i][j] *= scale;
            }
        }
    }

    // Дополнительно: реализация градиентного клиппинга (опционально)
    double max_norm = max_gradient_norm; // Предел нормы градиента из параметров модели
    double total_norm = 0.0;

    // Вычисляем норму градиентов (L2 норма)
    for (auto& layer : layers) {
        vector<double> all_gradients;
        layer.add_gradients_to_vector(all_gradients);
        for (double grad : all_gradients) {
            total_norm += grad * grad;
        }
    }
    for (int i = 0; i < embeddings.rows; i++) {
        for (int j = 0; j < embeddings.cols; j++) {
            total_norm += embeddings.grad[i][j] * embeddings.grad[i][j];
        }
    }
    for (int i = 0; i < output_projection.rows; i++) {
        for (int j = 0; j < output_projection.cols; j++) {
            total_norm += output_projection.grad[i][j] * output_projection.grad[i][j];
        }
    }
    total_norm = sqrt(total_norm);

    // Если норма превышает max_gradient_norm, масштабируем все градиенты
    if (total_norm > max_norm) 
    {
        double clip_scale = max_norm / (total_norm + 1e-6); // Избегаем деления на 0
        if (DEBUG == 1) 
        {
            wcout << L"Применяется клиппинг с масштабом: " << clip_scale << endl;

        }

        for (auto& layer : layers) {
            layer.scale_gradients(clip_scale);
        }
        if (embeddings.requires_grad) {
            for (int i = 0; i < embeddings.rows; i++) {
                for (int j = 0; j < embeddings.cols; j++) {
                    embeddings.grad[i][j] *= clip_scale;
                }
            }
        }
        if (output_projection.requires_grad) {
            for (int i = 0; i < output_projection.rows; i++) {
                for (int j = 0; j < output_projection.cols; j++) {
                    output_projection.grad[i][j] *= clip_scale;
                }
            }
        }
    }

    if (DEBUG == 1) {
        wcout << L"Градиенты нормализованы с масштабом: " << scale << L", норма градиентов: " << total_norm << L", порог: " << max_gradient_norm << endl;
    }


}



// === 2.4. Forward без кеша (полный проход по слоям)   и  Forward с кешем для инкрементальной генерации ===
Tensor TransformerModel::forward(const vector<int>& input_ids)
{
    if (DEBUG == 1) {  wcout << L"\n=== TransformerModel::forward (no cache) start ===" << endl;  }

    int seq_len = static_cast<int>(input_ids.size()); // input_ids.size() надеюсь не превысит 2,147,483,647, иначе произойдёт переполнение.
    if(seq_len >= 2000000000){    wcout << L"\n\n\n=== !!!!!!!     seq_len скоро привысит 2млн 147тыс.    !!!!!!! ===\n\n\n";  }

    if (embeddings.rows == 0 || embeddings.cols == 0 || pos_encoding.encoding.rows == 0 || pos_encoding.encoding.cols == 0)
    {
        wcout << L"embeddings rows: " << embeddings.rows << L", cols: " << embeddings.cols << endl;
        wcout << L"pos_encoding.encoding rows: " << pos_encoding.encoding.rows << L", cols: " << pos_encoding.encoding.cols << endl;

    }
    if (seq_len == 0) {
        wcout << L"Warning: input_ids is empty" << endl;
        return Tensor(0, embeddings.cols, false);
    }
    if (seq_len > MAX_SEQ_LENGTH) {
        throw runtime_error("Sequence length " + to_string(seq_len) +
            " exceeds MAX_SEQ_LENGTH " + to_string(MAX_SEQ_LENGTH));
    }


    // Сохраняем input_ids
    cached_input_ids = input_ids;
    if (cached_input_ids.size() == 0)
    {
        wcout << L"Saved cached_input_ids size: " << cached_input_ids.size() << endl;
    }


    // Получаем эмбеддинги токенов
    Tensor token_embeddings(seq_len, EMBEDDING_DIM);
    for (int i = 0; i < seq_len; i++) {
        if (input_ids[i] >= 0 && input_ids[i] < embeddings.rows) {
            for (int j = 0; j < EMBEDDING_DIM; j++) {
                token_embeddings.data[i][j] = embeddings.data[input_ids[i]][j];
            }
        }
        else 
        {
            wcout << L"Invalid token_id[" << i << "]: " << input_ids[i] << endl;
            throw runtime_error("Invalid token_id: " + to_string(input_ids[i]));
        }
    }

    // Добавляем позиционное кодирование
    Tensor input_embeddings = pos_encoding.add_positional_encoding(token_embeddings);

    // Проходим через слои трансформера
    Tensor hidden_states = input_embeddings;
    KVCache temp_cache;
    temp_cache.resize_for_layers(NUM_LAYERS, MAX_SEQ_LENGTH, EMBEDDING_DIM);
    for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++)
    {
        hidden_states = layers[layer_idx].forward(hidden_states, input_ids, cached_input_ids, temp_cache, layer_idx, false);
    }

    // Сохраняем для backward pass
    last_hidden_states = hidden_states;
    if (last_hidden_states.rows == 0 || last_hidden_states.cols == 0) 
    {
        wcout << L"Last hidden states shape: [" << last_hidden_states.rows << ", " << last_hidden_states.cols << "] - sample value [0][0]: " << last_hidden_states.data[0][0] << endl;
    }
    // Применяем layer norm перед выходом
    hidden_states = hidden_states.layer_norm();
    if (hidden_states.rows == 0 || hidden_states.cols == 0)
    {
        wcout << L"After final layer norm shape: [" << hidden_states.rows << ", " << hidden_states.cols << "] - sample value [0][0]: " << hidden_states.data[0][0] << endl;
    }

    // Финальная проекция
    Tensor output = hidden_states.matmul(output_projection);
    if (output.rows == 0 || output.cols == 0)
    {
        wcout << L"Output shape: [" << output.rows << ", " << output.cols << "] - sample value [0][0]: " << output.data[0][0] << endl;
    }

    if (DEBUG == 1) {  wcout << L"=== TransformerModel::forward (no cache) end ===" << endl;  }
    return output;
}
Tensor TransformerModel::forward(const vector<int>& input_ids, KVCache& cache)
{
    if (DEBUG == 1) { wcout << L"\n=== TransformerModel::forward (with cache) start ===" << endl; }

    int seq_len = static_cast<int>(input_ids.size()); // input_ids.size() надеюсь не превысит 2,147,483,647, иначе произойдёт переполнение.
    if (seq_len >= 2000000000) { wcout << L"\n\n\n=== !!!!!!!     seq_len скоро привысит 2млн 147тыс.    !!!!!!! ===\n\n\n"; }


    if (embeddings.rows == 0 || embeddings.cols == 0 || pos_encoding.encoding.rows == 0 || pos_encoding.encoding.cols == 0 || cache.current_length.size() == 0)
    {
        wcout << L"embeddings rows: " << embeddings.rows << L", cols: " << embeddings.cols << endl;
        wcout << L"pos_encoding.encoding rows: " << pos_encoding.encoding.rows << L", cols: " << pos_encoding.encoding.cols << endl;
        wcout << L"cache.current_length size: " << cache.current_length.size() << endl;
    }
    if (seq_len == 0) {
        wcout << L"Warning: input_ids is empty" << endl;
        return Tensor(0, embeddings.cols, false);
    }

    // // // DEBUG // Проверяем длину последовательности для каждого слоя
    for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++) {
        if (layer_idx >= static_cast<int>(cache.current_length.size())) {
            wcout << L"Error: layer_idx " << layer_idx << L" exceeds cache.current_length size " << cache.current_length.size() << endl;
            throw runtime_error("layer_idx exceeds cache.current_length size");
        }
        if (seq_len + cache.current_length[layer_idx] > MAX_SEQ_LENGTH) {
            throw runtime_error("Sequence length " + to_string(seq_len + cache.current_length[layer_idx]) +  " exceeds MAX_SEQ_LENGTH " + to_string(MAX_SEQ_LENGTH) + " for layer " + to_string(layer_idx));
        }
    }

    // Сохраняем input_ids
    cached_input_ids = input_ids;
    if (cached_input_ids.size() == 0) {
        wcout << L"Saved cached_input_ids size: " << cached_input_ids.size() << endl;
    }


    // Получаем эмбеддинги токенов
    Tensor token_embeddings(seq_len, EMBEDDING_DIM);
    for (int i = 0; i < seq_len; i++) {
        if (input_ids[i] >= 0 && input_ids[i] < embeddings.rows) {
            for (int j = 0; j < EMBEDDING_DIM; j++) {
                token_embeddings.data[i][j] = embeddings.data[input_ids[i]][j];
            }
        }
        else {
            wcout << L"Invalid token_id[" << i << "]: " << input_ids[i] << endl;
            throw runtime_error("Invalid token_id: " + to_string(input_ids[i]));
        }
    }

    // Позиционное кодирование с учетом текущей позиции в кеше
    Tensor input_embeddings(seq_len, EMBEDDING_DIM);
    for (int i = 0; i < seq_len; i++) {
        int pos = cache.current_length[0] + i;
        if (pos >= pos_encoding.encoding.rows) {
            wcout << L"Error: pos " << pos << L" exceeds pos_encoding.encoding.rows " << pos_encoding.encoding.rows << endl;
            throw runtime_error("Position " + to_string(pos) + " exceeds pos_encoding.encoding.rows");
        }
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            input_embeddings.data[i][j] = token_embeddings.data[i][j] + pos_encoding.encoding.data[pos][j];
        }
    }

    // Проходим через слои с использованием кеша
    Tensor hidden_states = input_embeddings;
    for (int layer_idx = 0; layer_idx < NUM_LAYERS; layer_idx++) {
        hidden_states = layers[layer_idx].forward(hidden_states, input_ids, cached_input_ids, cache, layer_idx, true);
    }

    // Сохраняем для backward pass
    last_hidden_states = hidden_states;
    if (last_hidden_states.rows == 0 || last_hidden_states.cols == 0)
    {
        wcout << L"Last hidden states shape: [" << last_hidden_states.rows << ", " << last_hidden_states.cols << "] - sample value [0][0]: " << last_hidden_states.data[0][0] << endl;
    }

    // Применяем layer norm перед выходом
    hidden_states = hidden_states.layer_norm();
    if (hidden_states.rows == 0 || hidden_states.cols == 0)
    {
        wcout << L"After final layer norm shape: [" << hidden_states.rows << ", " << hidden_states.cols << "] - sample value [0][0]: " << hidden_states.data[0][0] << endl;
    }
    // Финальная проекция
    Tensor output = hidden_states.matmul(output_projection);
    if (output.rows == 0 || output.cols == 0)
    {
        wcout << L"Output shape: [" << output.rows << ", " << output.cols << "] - sample value [0][0]: " << output.data[0][0] << endl;
    }

    if (DEBUG == 1) { wcout << L"=== TransformerModel::forward (with cache) end ===" << endl; }

    return output;
}

// === 3 Установка токенов с другого места
void TransformerModel::setTokenizer(const AdvancedTokenizer& tok) {
    wcout << L"\nSTART  ===  setTokenizer ===  START" << endl;
    tokenizer = std::make_unique<AdvancedTokenizer>(tok); // Копируем токенизатор

    VOCAB_SIZE = tokenizer->getVocabSize();
    wcout << L"Setting tokenizer with VOCAB_SIZE: " << VOCAB_SIZE << endl;

    if (VOCAB_SIZE <= 0) {
        wcout << L"Error: VOCAB_SIZE is " << VOCAB_SIZE << L", cannot initialize embeddings!" << endl;
        throw runtime_error("Invalid VOCAB_SIZE: " + to_string(VOCAB_SIZE));
    }
    if (!tokenizer) {
        wcout << L"Error: tokenizer is null after copying!" << endl;
        throw runtime_error("Tokenizer pointer is null.");
    }

    // Загружаем fastText эмбеддинги
    embeddings = Tensor(VOCAB_SIZE, EMBEDDING_DIM, true);
    for (int i = 0; i < VOCAB_SIZE; i++) 
    {
        string token = tokenizer->decode({ i });

        if (token.empty()) 
        {
            wcout << L"[WARNING] Token at index " << i << L" is empty!" << endl;
            continue; // Пропускаем или можно выбросить ошибку
        }

        vector<double> emb = tokenizer->get_word_embedding(token, EMBEDDING_DIM);

        if (emb.size() != static_cast<size_t>(EMBEDDING_DIM)) 
        {
            wcout << L"[ERROR] Invalid embedding size for token '" << wstring(token.begin(), token.end()) << L"' (index " << i << L")" << endl;
            throw runtime_error("Invalid embedding vector size at index " + to_string(i) + " for token: " + token);
        }


        for (int j = 0; j < EMBEDDING_DIM; j++) 
        {
            embeddings.data[i][j] = emb[j];
        }
    }




    output_projection = Tensor(EMBEDDING_DIM, VOCAB_SIZE, true);
    output_projection.randomize();
    wcout << L"Токенизатор установлен. Размер словаря: " << VOCAB_SIZE << endl;
    wcout << L"Embeddings shape: [" << embeddings.rows << ", " << embeddings.cols << "]" << endl;
    wcout << L"Output projection shape: [" << output_projection.rows << ", " << output_projection.cols << "]" << endl;
    wcout << L"END  ===  setTokenizer ===  END\n" << endl;
}



// === 5. Генерация ответа Кварка по текстовому запросу с семплированием токенов ===
string TransformerModel::generate_response(const string& input_text, int max_length)
{
    if (DEBUG == 1)   {  wcout << L"\n=========     START generate_response  START     =========" << endl; }

    wcout << L" == \nАнализ входного текста в generate_response с помощью токенов: ";
    analyze_tokenization(input_text); wcout << L" ==  ";

    // Токенизация входного текста
    vector<int> input_ids = tokenizer->encode(input_text);

    wcout << L"\nВходные токены: " << input_ids.size() << L", values: ";  for (int id : input_ids) wcout << id << L" " << endl;


    // Проверка корректности input_ids
    for (int id : input_ids) 
    {
        if (id < 0 || id >= tokenizer->getVocabSize()) {
            wcout << L"ERROR: Invalid token_id " << id << L" in input_ids" << endl;
            return "Error: Invalid token_id";
        }
    }


    // Инициализация кеша
    wcout << L"Initializing generation_cache for " << NUM_LAYERS << L" layers, MAX_SEQ_LENGTH: " << MAX_SEQ_LENGTH << L", EMBEDDING_DIM: " << EMBEDDING_DIM << endl;

    generation_cache.resize_for_layers(NUM_LAYERS, MAX_SEQ_LENGTH, EMBEDDING_DIM);

    // Первый проход - обрабатываем весь input сразу
    if (DEBUG == 1) 
    {
        wcout << L"\n >>> generate_response <<<   выполняется forward " << endl;
        wcout << L"After resize_for_layers: generation_cache.current_length size: " << generation_cache.current_length.size() << endl;
    }

    Tensor logits = forward(input_ids, generation_cache); 
    if (generation_cache.current_length.size() == 0)
    {
        wcout << L"==========  Initialized generation_cache for " << NUM_LAYERS << L" layers, generation_cache.current_length size: " << generation_cache.current_length.size() << endl;
    }

    // Проверка логитов на NaN/inf
    bool has_invalid_logits = false;
    for (int i = 0; i < logits.rows; i++) {
        for (int j = 0; j < logits.cols; j++) {
            if (!isfinite(logits.data[i][j])) {
                wcout << L"ERROR: Invalid logit value (NaN/inf) at position [" << i << L"][" << j << L"]: " << logits.data[i][j] << endl;
                has_invalid_logits = true;
            }
        }
    }
    if (has_invalid_logits) {
        return "Error: Invalid logits detected";
    }





    vector<int> generated = input_ids;

    for (int step = 0; step < max_length; step++) 
    {
        wcout << L" Generation step " << step << endl;

        // Получаем логиты для последнего токена
        vector<double> last_logits;
        int last_pos = logits.rows - 1;
        for (int i = 0; i < logits.cols; i++) {
            last_logits.push_back(logits.data[last_pos][i]);
        }

        // Проверка last_logits на NaN/inf
        for (double logit : last_logits) {
            if (!isfinite(logit)) {
                wcout << L"ERROR: Invalid logit value (NaN/inf) in last_logits" << endl;
                return "Error: Invalid logits in sampling";
            }
        }


        // Семплируем следующий токен
        if (DEBUG == 1) { wcout << L" Sampling next token ..." << endl; }
        int next_token = sample_token(last_logits, 1.2, 15);
        wcout << L" Sampled next_token: " << next_token << "\n";


        // Проверка корректности next_token
        if (next_token < 0 || next_token >= tokenizer->getVocabSize())
        {
            wcout << L"ERROR: Invalid next_token " << next_token << L" (VOCAB_SIZE: " << tokenizer->getVocabSize() << L")" << endl;
            return "Error: Invalid sampled token";
        }
        if (next_token == tokenizer->getEosTokenId()) {
            wcout << L" EOS token reached, stopping generation" << endl;
            break;
        }


        generated.push_back(next_token);

        // Следующий проход с кешем
        if (DEBUG == 1) {  wcout << L" Запускаем forward для следующего тоекна: " << next_token << endl;  }

        logits = forward({ next_token }, generation_cache);

        // Проверка 
        if (logits.rows == 0 || logits.cols == 0) 
        {
            wcout << L" Forward pass completed, logits shape: [" << logits.rows << L", " << logits.cols << L"]" << endl;
        }
    }

    if (DEBUG == 1) { wcout << L" Generation completed, decoding generated tokens" << endl; }
    string response = tokenizer->decode(generated);
    if (DEBUG == 1) 
    {
        wcout << L"\n\nОтвет до передачи Generated response : " << response.c_str() << endl;
        wcout << L"\n=========     END generate_response  END     =========" << endl;
    }
    return response;
}

// === 6. Семплирование токена из logits с temperature и top-k фильтрацией ===
int TransformerModel::sample_token(const vector<double>& logits, double temperature, int top_k)
{
    // Temperature scaling
    vector<double> scaled_logits = logits;
    for (double& logit : scaled_logits) {
        logit /= temperature;
    }

    // Top-k filtering
    vector<pair<double, int>> logit_pairs;
    for (int i = 0; i < scaled_logits.size(); i++) {
        logit_pairs.push_back({ scaled_logits[i], i });
    }

    // Сортируем по убыванию
    sort(logit_pairs.begin(), logit_pairs.end(), greater<pair<double, int>>());

    // Берем только top-k токенов
    vector<double> top_logits;
    vector<int> top_indices;
    for (int i = 0; i < min(top_k, (int)logit_pairs.size()); i++) {
        top_logits.push_back(logit_pairs[i].first);
        top_indices.push_back(logit_pairs[i].second);
    }

    // Softmax для top-k
    vector<double> probs = MathUtils::softmax(top_logits);

    // Sampling
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);

    double r = dis(gen);
    double cumsum = 0.0;

    for (int i = 0; i < probs.size(); i++) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return top_indices[i];
        }
    }

    // Fallback
    return top_indices[0];
}



//-----------------------------------------------------------------------------------------------------
// отладка      //    просмотр информации  // служебные методы
void TransformerModel::analyze_tokenization(const string& text) 
{

    vector<int> tokens = tokenizer->encode(text);
    wcout << L"\n Количество токенов: "; cout << tokens.size() << endl;
    wcout << L"Токены: ";
    for (int token : tokens) 
    {
        cout << token << " ";
    }
    cout << endl;

    string decoded = tokenizer->decode(tokens);
    wcout << L"Декодированный текст: "; cout << decoded << endl;
    wcout << L"Соответствие: "; cout << (text == decoded ? "Yeas" : "No") << endl;
}




void TransformerModel::save_tensor(json& j, const std::string& key, Tensor& tensor)
{
    std::wcout << L"Сохранение тензора: " << key.c_str() << L", размеры: ["  << tensor.rows << L", " << tensor.cols << L"]" << std::endl;

    if (tensor.data.empty() || tensor.data[0].empty()) {
        std::wcout << L"Ошибка: Тензор " << key.c_str() << L" пустой!" << std::endl;
        throw std::runtime_error("Пустой тензор: " + key);
    }

    // Логируем первые несколько элементов для отладки
    if (key == "W_q") 
    {
        std::wcout << L"Значения W_q[0][0:5]: ";
        for (size_t j = 0; j < std::min(static_cast<size_t>(5), tensor.data[0].size()); ++j) {
            std::wcout << tensor.data[0][j] << L" ";
        }
        std::wcout << std::endl;
    }

    // Дополнительная проверка размеров и валидности
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        if (tensor.data[i].size() != static_cast<size_t>(tensor.cols)) {
            std::wcout << L"Ошибка: Несоответствие размеров в строке " << i   << L" тензора " << key.c_str() << std::endl;
            throw std::runtime_error("Несоответствие размеров в тензоре: " + key);
        }
    }

    j[key] = tensor.data;
}

// Улучшенная версия Save_File_Communication_RUS
void TransformerModel::Save_File_Communication_RUS() {
    try {
        std::wcout << L"Начинаем сохранение модели..." << std::endl;

        json j;

        // Добавляем версию для совместимости
        j["version"] = "1.0";

        // Сохраняем метаданные
        j["embedding_dim"] = EMBEDDING_DIM;
        j["num_heads"] = NUM_HEADS;
        j["ff_dim"] = FF_DIM;
        j["num_layers"] = NUM_LAYERS;
        j["VOCAB_SIZE"] = VOCAB_SIZE;
        j["step_count"] = step_count;

        // Сохраняем embeddings
        save_tensor(j, "embeddings", embeddings);

        // Сохраняем output_projection
        save_tensor(j, "output_projection", output_projection);

        // Сохраняем веса всех слоев
        json j_layers = json::array();
        for (size_t i = 0; i < layers.size(); ++i) {
            json j_layer;

            // Создаем структуру для attention
            j_layer["attention"] = json::object();
            save_tensor(j_layer["attention"], "W_q", layers[i].attention.W_q);
            save_tensor(j_layer["attention"], "W_k", layers[i].attention.W_k);
            save_tensor(j_layer["attention"], "W_v", layers[i].attention.W_v);
            save_tensor(j_layer["attention"], "W_o", layers[i].attention.W_o);
            save_tensor(j_layer["attention"], "b_q", layers[i].attention.b_q);
            save_tensor(j_layer["attention"], "b_k", layers[i].attention.b_k);
            save_tensor(j_layer["attention"], "b_v", layers[i].attention.b_v);
            save_tensor(j_layer["attention"], "b_o", layers[i].attention.b_o);

            // Создаем структуру для feedforward
            j_layer["feedforward"] = json::object();
            save_tensor(j_layer["feedforward"], "W1", layers[i].feed_forward.W1);
            save_tensor(j_layer["feedforward"], "W2", layers[i].feed_forward.W2);
            save_tensor(j_layer["feedforward"], "b1", layers[i].feed_forward.b1);
            save_tensor(j_layer["feedforward"], "b2", layers[i].feed_forward.b2);

            // Создаем структуры для layer norm
            j_layer["ln1"] = json::object();
            save_tensor(j_layer["ln1"], "gamma", layers[i].ln1_gamma);
            save_tensor(j_layer["ln1"], "beta", layers[i].ln1_beta);

            j_layer["ln2"] = json::object();
            save_tensor(j_layer["ln2"], "gamma", layers[i].ln2_gamma);
            save_tensor(j_layer["ln2"], "beta", layers[i].ln2_beta);

            j_layers.push_back(j_layer);
        }
        j["layers"] = j_layers;

        // Сохраняем в файл
        std::ofstream file(Path_directory_Quark + "Brain\\Communication_RUS.json");
        if (!file.is_open()) 
        {
            wcout << L"Не удалось открыть файл для сохранения\n";
        }

        file << j.dump(4) << std::endl;
        file.close();

        std::wcout << L"Модель успешно сохранена в Communication_RUS.json" << std::endl;
    }
    catch (const std::exception& e) {
        std::wcout << L"Ошибка при сохранении модели: " << e.what() << std::endl;
    }
}

void TransformerModel::load_tensor(Tensor& tensor, const json& j, const std::string& key, int rows, int cols) {
    if (!j.contains(key)) {
        throw std::runtime_error("Отсутствует ключ '" + key + "' в JSON");
    }
    tensor = Tensor(rows, cols, true);
    tensor.data = j[key].get<std::vector<std::vector<double>>>();
    if (tensor.data.size() != rows || (rows > 0 && tensor.data[0].size() != cols)) {
        throw std::runtime_error("Несовместимые размеры для тензора '" + key + "'");
    }
}
void TransformerModel::Load_File_Communication_RUS()
{
    try {
        std::ifstream file(Path_directory_Quark + "Brain\\Communication_RUS.json");
        if (!file.is_open()) {
            throw std::runtime_error("Не удалось открыть файл Communication_RUS.json");
        }

        json j;
        file >> j;
        file.close();

        // Проверяем метаданные
        if (!j.contains("version") || !j.contains("embedding_dim") || !j.contains("num_heads") ||
            !j.contains("ff_dim") || !j.contains("num_layers") || !j.contains("VOCAB_SIZE") ||
            !j.contains("step_count") || !j.contains("embeddings") || !j.contains("output_projection") ||
            !j.contains("layers")) {
            throw std::runtime_error("В JSON отсутствуют необходимые ключи");
        }

        // Проверяем совместимость параметров
        if (j["embedding_dim"].get<int>() != EMBEDDING_DIM ||
            j["num_heads"].get<int>() != NUM_HEADS ||
            j["ff_dim"].get<int>() != FF_DIM ||
            j["num_layers"].get<int>() != NUM_LAYERS ||
            j["VOCAB_SIZE"].get<int>() != VOCAB_SIZE) {
            throw std::runtime_error("Несовместимые параметры модели");
        }

        // Загружаем step_count
        step_count = j["step_count"].get<int>();

        // Загружаем embeddings
        load_tensor(embeddings, j, "embeddings", VOCAB_SIZE, EMBEDDING_DIM);

        // Загружаем output_projection
        load_tensor(output_projection, j, "output_projection", EMBEDDING_DIM, VOCAB_SIZE);

        // Загружаем веса слоев
        if (j["layers"].size() != NUM_LAYERS) {
            throw std::runtime_error("Несоответствие количества слоев");
        }

        // Очищаем существующий вектор layers и создаем новые слои
        layers.clear();
        layers.reserve(NUM_LAYERS); // Резервируем место для эффективности
        for (size_t i = 0; i < NUM_LAYERS; ++i) {
            // Создаем новый слой с нужными параметрами
            layers.emplace_back(EMBEDDING_DIM, NUM_HEADS, FF_DIM);
            auto& j_layer = j["layers"][i];

            // Загружаем MultiHeadAttention
            load_tensor(layers[i].attention.W_q, j_layer["attention"], "W_q", EMBEDDING_DIM, EMBEDDING_DIM);
            load_tensor(layers[i].attention.W_k, j_layer["attention"], "W_k", EMBEDDING_DIM, EMBEDDING_DIM);
            load_tensor(layers[i].attention.W_v, j_layer["attention"], "W_v", EMBEDDING_DIM, EMBEDDING_DIM);
            load_tensor(layers[i].attention.W_o, j_layer["attention"], "W_o", EMBEDDING_DIM, EMBEDDING_DIM);
            load_tensor(layers[i].attention.b_q, j_layer["attention"], "b_q", 1, EMBEDDING_DIM);
            load_tensor(layers[i].attention.b_k, j_layer["attention"], "b_k", 1, EMBEDDING_DIM);
            load_tensor(layers[i].attention.b_v, j_layer["attention"], "b_v", 1, EMBEDDING_DIM);
            load_tensor(layers[i].attention.b_o, j_layer["attention"], "b_o", 1, EMBEDDING_DIM);

            // Загружаем FeedForward
            load_tensor(layers[i].feed_forward.W1, j_layer["feedforward"], "W1", EMBEDDING_DIM, FF_DIM);
            load_tensor(layers[i].feed_forward.W2, j_layer["feedforward"], "W2", FF_DIM, EMBEDDING_DIM);
            load_tensor(layers[i].feed_forward.b1, j_layer["feedforward"], "b1", 1, FF_DIM);
            load_tensor(layers[i].feed_forward.b2, j_layer["feedforward"], "b2", 1, EMBEDDING_DIM);

            // Загружаем LayerNorm параметры
            load_tensor(layers[i].ln1_gamma, j_layer["ln1"], "gamma", 1, EMBEDDING_DIM);
            load_tensor(layers[i].ln1_beta, j_layer["ln1"], "beta", 1, EMBEDDING_DIM);
            load_tensor(layers[i].ln2_gamma, j_layer["ln2"], "gamma", 1, EMBEDDING_DIM);
            load_tensor(layers[i].ln2_beta, j_layer["ln2"], "beta", 1, EMBEDDING_DIM);
        }

        std::wcout << L"Модель успешно загружена из Communication_RUS.json" << std::endl;
    }
    catch (const std::exception& e) {
        std::wcout << L"Ошибка при загрузке модели: " << e.what() << std::endl;
    }
}
