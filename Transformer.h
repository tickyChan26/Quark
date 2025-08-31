/*
Transformer.h
ТОЛЬКО ОБЪЯВЛЕНИЯ КЛАССОВ - реализация в .cpp файле
*/

#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <set>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <openblas/cblas.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

// Внешние зависимости
extern string Path_directory_Quark; extern int DEBUG;

#include "Math.h"
#include "Tokenizer.h"

// ===== FORWARD DECLARATIONS =====
class Tensor;
class FeedForward;
class PositionalEncoding;
class MultiHeadAttention;
class TransformerLayer;
class TransformerModel;

// ===== TENSOR WITH AUTOMATIC DIFFERENTIATION =====
class Tensor
{
public:

	int d_k; // Добавляем поле для хранения d_k
	vector<vector<double>> data;
	vector<vector<double>> grad;
	int rows, cols;
	bool requires_grad;
	int backward_cache_len = 0;

	// Для автоградиента
	std::vector<Tensor*> parents;           // Родительские тензоры
	std::string operation;                  // Название операции ("matmul", "add", "relu" и т.д.)

	// Конструкторы
	Tensor() : rows(0), cols(0), requires_grad(true) {}
	Tensor(int r, int c, bool requires_grad_param = true);

	// Основные операции
	void randomize();
	void zero_grad();
	void zero();

	// Матричные операции
	vector<double> to_flat_array(const std::vector<std::vector<double>>& matrix) const;
	void from_flat_array(const std::vector<double>& flat, std::vector<std::vector<double>>& matrix, int rows, int cols) const;

	Tensor matmul(const Tensor& other) const;
	Tensor matmul_grad_data(const Tensor& other) const;
	Tensor matmul_data_grad(const Tensor& other) const;

	Tensor transpose() const;
	Tensor add(const Tensor& other, bool inplace = false) const;

	// Нормализация
	Tensor layer_norm() const;

	void backward();

	// Статические методы для backward pass
	static void backward_matmul(const Tensor& grad_output, const Tensor& tensor1, const Tensor& tensor2, Tensor& grad_tensor1, Tensor& grad_tensor2) {
		// grad_tensor1 = grad_output * tensor2^T
		if (grad_tensor1.requires_grad) {
			Tensor tensor2_T = tensor2.transpose();
			Tensor grad_tensor1_contrib = grad_output.matmul(tensor2_T);
			// Аккумулируем градиенты
			for (int i = 0; i < grad_tensor1.rows; i++) {
				for (int j = 0; j < grad_tensor1.cols; j++) {
					grad_tensor1.grad[i][j] += grad_tensor1_contrib.data[i][j];
				}
			}
		}
		// grad_tensor2 = tensor1^T * grad_output

		if (grad_tensor2.requires_grad) {
			Tensor tensor1_T = tensor1.transpose();
			Tensor grad_tensor2_contrib = tensor1_T.matmul(grad_output);
			// Аккумулируем градиенты
			for (int i = 0; i < grad_tensor2.rows; i++) {
				for (int j = 0; j < grad_tensor2.cols; j++) {
					grad_tensor2.grad[i][j] += grad_tensor2_contrib.data[i][j];
				}
			}
		}
	}
	static void backward_add(const Tensor& grad_output, Tensor& grad_tensor1, Tensor& grad_tensor2) {
		if (grad_tensor1.requires_grad) {
			for (int i = 0; i < grad_tensor1.rows; i++) {
				for (int j = 0; j < grad_tensor1.cols; j++) {
					grad_tensor1.grad[i][j] += grad_output.data[i][j];
				}
			}
		}
		if (grad_tensor2.requires_grad) {
			for (int i = 0; i < grad_tensor2.rows; i++) {
				for (int j = 0; j < grad_tensor2.cols; j++) {
					grad_tensor2.grad[i][j] += grad_output.data[i][j];
				}
			}
		}
	}
	static void backward_layer_norm(const Tensor& grad_output, const Tensor& input, Tensor& grad_input)
	{
		if (DEBUG == 1) { wcout << L"\n  ==  ВХОДИМ В backward_layer_norm  (.h)  == " << endl; }

		const double eps = 1e-6;

		if (input.rows == 0 || input.cols == 0 || grad_output.rows == 0 || grad_output.cols == 0) {
			wcout << L"backward_layer_norm: input shape [" << input.rows << ", " << input.cols << "]" << L" -  grad_output shape [" << grad_output.rows << ", " << grad_output.cols << "]" << endl;
		}
		// Проверка размеров
		if (grad_output.rows != input.rows || grad_output.cols != input.cols) {
			throw runtime_error("Dimension mismatch in backward_layer_norm");
		}
		if (grad_input.rows != input.rows || grad_input.cols != input.cols) {
			throw runtime_error("grad_input dimension mismatch in backward_layer_norm");
		}

		// Инициализация grad_input.grad
		if (grad_input.requires_grad)
		{
			grad_input.zero_grad();
		}
		else
		{
			wcout << L"- grad_input  уже и так пустое  !! skipping !!" << endl;
			return;
		}

		for (int i = 0; i < input.rows; i++)
		{
			// Вычисляем mean и variance
			double mean = 0.0;
			for (int j = 0; j < input.cols; j++) {
				mean += input.data[i][j];
			}
			mean /= input.cols;

			double variance = 0.0;
			for (int j = 0; j < input.cols; j++) {
				double diff = input.data[i][j] - mean;
				variance += diff * diff;
			}
			variance /= input.cols;
			double std_dev = sqrt(variance + eps);

			// Проверка на числовую стабильность
			if (std_dev < 1e-10) {
				wcout << L"Warning: std_dev too small: " << std_dev << endl;
				throw runtime_error("Standard deviation too small in backward_layer_norm");
			}

			// Вычисляем градиенты
			double grad_mean = 0.0;
			double grad_var = 0.0;
			for (int j = 0; j < input.cols; j++) {
				double normalized = (input.data[i][j] - mean) / std_dev;
				grad_mean += grad_output.grad[i][j]; // Используй grad
				grad_var += grad_output.grad[i][j] * normalized; // Используй grad
			}
			grad_mean /= input.cols;
			grad_var *= -0.5 * pow(variance + eps, -1.5);

			// Градиенты для входа
			for (int j = 0; j < input.cols; j++) {
				grad_input.grad[i][j] = (grad_output.grad[i][j] / std_dev) + // Используй grad
					(grad_var * 2.0 * (input.data[i][j] - mean) / input.cols) -
					(grad_mean / (input.cols * std_dev));
			}
		}

		if (DEBUG == 1) { wcout << L"  ==  ВЫХОД из backward_layer_norm  (.h)  ==      grad_input[0][0]: " << grad_input.grad[0][0] << "\n" << endl; }
	}
	static void backward_layer_norm_with_params(const Tensor& grad_output, const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor& grad_input, Tensor& grad_gamma, Tensor& grad_beta)
	{
		if (DEBUG == 1) { wcout << L"\n  ==  ВХОДИМ В backward_layer_norm_with_params (.h)  == " << endl; }

		const double eps = 1e-6;
		if (input.rows == 0 || input.cols == 0 || grad_output.rows == 0 || grad_output.cols == 0 || gamma.rows == 0 || gamma.cols == 0 || beta.rows == 0 || beta.cols == 0)
		{
			wcout << L"backward_layer_norm_with_params: input shape [" << input.rows << ", " << input.cols << "]"
				<< L" - grad_output shape [" << grad_output.rows << ", " << grad_output.cols << "]"
				<< L" - gamma shape [" << gamma.rows << ", " << gamma.cols << "]"
				<< L" - beta shape [" << beta.rows << ", " << beta.cols << "]" << endl;
		}

		// Проверка размеров
		if (grad_output.rows != input.rows || grad_output.cols != input.cols) {
			throw runtime_error("Dimension mismatch in backward_layer_norm_with_params: grad_output vs input");
		}
		if (grad_input.rows != input.rows || grad_input.cols != input.cols) {
			throw runtime_error("grad_input dimension mismatch in backward_layer_norm_with_params");
		}
		if (gamma.rows != 1 || gamma.cols != input.cols || beta.rows != 1 || beta.cols != input.cols) {
			throw runtime_error("gamma or beta dimension mismatch in backward_layer_norm_with_params");
		}

		// Инициализация градиентов
		if (grad_input.requires_grad) {
			grad_input.zero_grad();
		}
		else
		{
			wcout << L"- grad_input does not require grad, skipping" << endl;
			return;
		}

		if (grad_gamma.requires_grad)
		{
			grad_gamma.zero_grad();
		}
		else
		{
			wcout << L"- grad_gamma does not require grad, skipping" << endl;
			return;
		}

		if (grad_beta.requires_grad) {
			grad_beta.zero_grad();
		}
		else
		{
			wcout << L"- grad_beta does not require grad, skipping" << endl;
			return;
		}

		for (int i = 0; i < input.rows; i++) {
			// Вычисляем mean и variance
			double mean = 0.0;
			for (int j = 0; j < input.cols; j++) {
				mean += input.data[i][j];
			}
			mean /= input.cols;

			double variance = 0.0;
			for (int j = 0; j < input.cols; j++) {
				double diff = input.data[i][j] - mean;
				variance += diff * diff;
			}
			variance /= input.cols;
			double std_dev = sqrt(variance + eps);

			// Проверка на числовую стабильность
			if (std_dev < 1e-10) {
				wcout << L"Warning: std_dev too small: " << std_dev << endl;
				throw runtime_error("Standard deviation too small in backward_layer_norm_with_params");
			}

			// Вычисляем нормализованный вход
			vector<double> normalized(input.cols);
			for (int j = 0; j < input.cols; j++) {
				normalized[j] = (input.data[i][j] - mean) / std_dev;
			}

			// Градиенты для gamma и beta
			if (grad_gamma.requires_grad || grad_beta.requires_grad) {
				for (int j = 0; j < input.cols; j++) {
					if (grad_gamma.requires_grad) {
						grad_gamma.grad[0][j] += grad_output.grad[i][j] * normalized[j];
					}
					if (grad_beta.requires_grad) {
						grad_beta.grad[0][j] += grad_output.grad[i][j];
					}
				}
			}

			// Градиенты для входа
			double grad_mean = 0.0;
			double grad_var = 0.0;
			for (int j = 0; j < input.cols; j++) {
				double grad_scaled = grad_output.grad[i][j] * gamma.data[0][j]; // Учитываем gamma
				grad_mean += grad_scaled;
				grad_var += grad_scaled * normalized[j];
			}
			grad_mean /= input.cols;
			grad_var *= -0.5 * pow(variance + eps, -1.5);

			for (int j = 0; j < input.cols; j++) {
				grad_input.grad[i][j] = (grad_output.grad[i][j] * gamma.data[0][j] / std_dev) +
					(grad_var * 2.0 * (input.data[i][j] - mean) / input.cols) -
					(grad_mean / (input.cols * std_dev));
			}
		}

		if (grad_input.grad[0][0] == 0 || grad_gamma.grad[0][0] == 0 || grad_beta.grad[0][0] == 0)
		{
			wcout << L" grad_input[0][0]: " << grad_input.grad[0][0] << L" grad_gamma[0][0]: " << grad_gamma.grad[0][0] << L" grad_beta[0][0]: " << grad_beta.grad[0][0] << endl;
		}

		if (DEBUG == 1) { wcout << L"  ==  ВЫХОД из backward_layer_norm_with_params  (.h)  == " << endl; }

	}

};

// ===== СТРУКТУРА ДЛЯ KV-КЕША =====
struct KVCache {
	vector<Tensor> keys;    // кеш ключей для каждого слоя
	vector<Tensor> values;  // кеш значений для каждого слоя
	vector<int> current_length; // Длина кеша для каждого слоя

	KVCache() {
		// По умолчанию пустой кеш
		keys.clear();
		values.clear();
		current_length.clear();
	}

	void clear() {
		// Сбрасываем кеш для всех слоев
		for (size_t i = 0; i < keys.size(); i++) {
			keys[i].zero(); // Явно обнуляем тензоры
			values[i].zero();
			current_length[i] = 0; // Сбрасываем длину для каждого слоя
		}
	}

	void resize_for_layers(int num_layers, int max_len, int d_model) {
		keys.resize(num_layers);
		values.resize(num_layers);
		current_length.resize(num_layers, 0); // Инициализируем длину нулями для каждого слоя
		for (int i = 0; i < num_layers; i++) {
			keys[i] = Tensor(max_len, d_model, false);
			values[i] = Tensor(max_len, d_model, false);
			keys[i].zero(); // Явно обнуляем тензоры
			values[i].zero();
		}
	}

};

// ===== MULTI-HEAD ATTENTION =====
class MultiHeadAttention {
private:

	// Приватные методы
	vector<Tensor> split_heads(const Tensor& input);
	vector<Tensor> split_heads_grad(const Tensor& input);

	Tensor merge_heads(const vector<Tensor>& heads);

	// Кеш для backward pass
	Tensor cached_input;
	Tensor cached_norm_input;
	vector<Tensor> cached_attention_weights;
	Tensor cached_concat_output;
	Tensor cached_Q, cached_K, cached_V;

public:
	Tensor b_q, b_k, b_v, b_o;
	Tensor W_q, W_k, W_v, W_o;
	int d_model;
	int num_heads;
	int d_k; // Добавляем d_k как член класса
	double attention_dropout;

	MultiHeadAttention(int model_dim, int heads);

	Tensor forward(const Tensor& input, const vector<int>& input_ids, vector<int>& cached_input_ids, KVCache& cache, int layer_idx, bool use_cache);
	Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V, int cache_len);

	static void backward_scaled_attention(const Tensor& grad_output, const Tensor& Q, const Tensor& K, const Tensor& V, int cache_len, int d_k, Tensor& grad_Q, Tensor& grad_K, Tensor& grad_V)
	{
		if (DEBUG == 1) { wcout << L"\n    === START   backward_scaled_attention  (.h)   START ===  " << endl; }

		// Пересчитываем attention scores и weights
		Tensor scores = Q.matmul(K.transpose());
		double scale = 1.0 / sqrt(d_k);

		// Применяем маску и softmax
		for (int i = 0; i < scores.rows; i++) {
			for (int j = 0; j < scores.cols; j++) {
				scores.data[i][j] *= scale;
				bool masked = (cache_len == 0) ? (j > i) : (j > cache_len + i);
				if (masked) scores.data[i][j] = -1e9;
			}
		}

		// Softmax для каждой строки
		vector<vector<double>> attention_weights(scores.rows, vector<double>(scores.cols));
		for (int i = 0; i < scores.rows; i++) {
			vector<double> softmax_scores = MathUtils::softmax(scores.data[i]);
			attention_weights[i] = softmax_scores;
		}

		// Backward pass:   // 1. grad_V = attention_weights^T * grad_output
		Tensor attn_weights_tensor(scores.rows, scores.cols);
		for (int i = 0; i < scores.rows; i++) {
			for (int j = 0; j < scores.cols; j++) {
				attn_weights_tensor.data[i][j] = attention_weights[i][j];
			}
		}
		grad_V = attn_weights_tensor.transpose().matmul_data_grad(grad_output);

		// 2. grad_attention_weights = grad_output * V^T
		Tensor grad_attn_weights = grad_output.matmul_grad_data(V.transpose());

		// 3. Backward через softmax
		Tensor grad_scores(scores.rows, scores.cols);
		for (int i = 0; i < scores.rows; i++) {
			for (int j = 0; j < scores.cols; j++) {
				double softmax_grad = 0.0;
				for (int k = 0; k < scores.cols; k++) {
					double delta_jk = (j == k) ? 1.0 : 0.0;
					softmax_grad += grad_attn_weights.grad[i][k] * attention_weights[i][j] * (delta_jk - attention_weights[i][k]);
				}
				grad_scores.grad[i][j] = softmax_grad * scale; // Записываем в grad вместо data
			}
		}

		// 4. Backward через Q * K^T
		grad_Q = grad_scores.matmul_grad_data(K); // grad_scores.grad * K.data
		// Создаём временный тензор для grad_scores.grad
		Tensor grad_scores_data(grad_scores.rows, grad_scores.cols);
		for (int i = 0; i < grad_scores.rows; i++) {
			for (int j = 0; j < grad_scores.cols; j++) {
				grad_scores_data.data[i][j] = grad_scores.grad[i][j];
			}
		}
		// Транспонируем и умножаем
		Tensor grad_K_temp = grad_scores_data.transpose().matmul(Q); // grad_scores.grad^T * Q.data
		grad_K = Tensor(grad_K_temp.rows, grad_K_temp.cols, true);
		for (int i = 0; i < grad_K_temp.rows; i++) {
			for (int j = 0; j < grad_K_temp.cols; j++) {
				grad_K.grad[i][j] = grad_K_temp.data[i][j]; // Переносим из data в grad
			}
		}

		if (DEBUG == 1) { wcout << L"\n    === END   backward_scaled_attention  (.h)   END ===  " << endl; }
	}

	void backward(const Tensor& grad_output, Tensor& grad_input);
	void zero_grad();
	void add_gradients_to_vector(vector<double>& all_gradients);
	void scale_gradients(double scale);
	void apply_weight_decay(double weight_decay);
};

// ===== FEED FORWARD =====
class FeedForward {
private:
	double dropout_rate;

	// Кеш для backward pass
	Tensor cached_input;
	Tensor cached_hidden;
	Tensor cached_gelu_input;

public:
	Tensor W1, W2;
	Tensor b1, b2;

	FeedForward(int d_model, int d_ff);
	Tensor forward(const Tensor& input);
	void backward(const Tensor& grad_output, Tensor& grad_input);
	void zero_grad();

	void add_gradients_to_vector(vector<double>& all_gradients);
	void scale_gradients(double scale);
	void apply_weight_decay(double weight_decay);
};

// ===== TRANSFORMER LAYER =====
class TransformerLayer {
private:

	// Кеш для backward pass
	Tensor cached_input;
	Tensor cached_norm1;
	Tensor cached_attn_output;
	Tensor cached_residual1;
	Tensor cached_norm2;
	Tensor cached_ff_output;
	Tensor cached_attn_input;
	Tensor cached_ff_input;

public:

	MultiHeadAttention attention;
	FeedForward feed_forward;
	Tensor ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;

	TransformerLayer();
	TransformerLayer(int d_model, int num_heads, int d_ff);
	Tensor forward(const Tensor& input, const vector<int>& input_ids, vector<int>& cached_input_ids, KVCache& cache, int layer_idx, bool use_cache);

	// Методы для backward pass
	Tensor backward(const Tensor& grad_output);

	void zero_grad();

	// Публичные методы для градиентов (нужны для TransformerModel)
	void add_gradients_to_vector(vector<double>& all_gradients);
	void scale_gradients(double scale);
	void apply_weight_decay(double weight_decay);
};

// ===== POSITIONAL ENCODING =====
class PositionalEncoding {
public:
	Tensor encoding;

	PositionalEncoding(int max_len, int d_model);
	Tensor add_positional_encoding(const Tensor& embeddings);
};

// ===== ADVANCED TRANSFORMER MODEL =====
class TransformerModel
{
private:
	KVCache generation_cache;

	// Приватные методы
	void backward_pass();
	void update_with_adam(Tensor& param);
	int sample_token(const vector<double>& logits, double temperature = 1.0, int top_k = 50);
	void train_Weights(const vector<string>& texts);

	// Tensor pool для оптимизации
	vector<Tensor> tensor_pool;
	int pool_index = 0;

	Tensor get_pooled_tensor(int rows, int cols) {
		if (pool_index >= tensor_pool.size()) {
			tensor_pool.emplace_back(rows, cols, false);
		}
		Tensor& t = tensor_pool[pool_index++];
		if (t.rows != rows || t.cols != cols) {
			t = Tensor(rows, cols, false);
		}
		pool_index = (pool_index + 1) % 10;
		return t;
	}

public:
	// Кеш для backward pass
	vector<Tensor> cached_layer_inputs;
	vector<Tensor> cached_layer_outputs;
	Tensor cached_token_embeddings;
	Tensor cached_input_embeddings;

	vector<int> cached_input_ids;

	// Параметры модели
	Tensor grad_through_layers; // Добавляем поле для градиентов

	Tensor embeddings;              // Embedding матрица
	Tensor output_projection;       // Выходная проекция
	Tensor last_hidden_states;      // Сохраненные состояния для backward pass

	vector<TransformerLayer> layers;
	PositionalEncoding pos_encoding;
	unique_ptr<AdvancedTokenizer> tokenizer; // Изменено на unique_ptr

	// Параметры обучения
	double learning_rate;
	double beta1, beta2;
	double epsilon;
	int step_count;
	double max_gradient_norm = 1.0;
	double weight_decay = 0.01;
	bool regularize_positional_embeddings = false;

	// Состояние Adam оптимизатора
	map<Tensor*, Tensor> m_embeddings, v_embeddings;
	map<Tensor*, Tensor> m_output, v_output;

	// Конструктор
	TransformerModel();

	// Методы для работы с токенизатором
	void setTokenizer(const AdvancedTokenizer& tok);
	AdvancedTokenizer& getTokenizer() { return *tokenizer; }
	const AdvancedTokenizer& getTokenizer() const { return *tokenizer; }

	// Основные методы
	Tensor forward(const vector<int>& input_ids);
	Tensor forward(const vector<int>& input_ids, KVCache& cache);

	// Обучение
	void backward_propagate_gradients(const Tensor& logits);
	double calculate_loss_and_gradients(Tensor& logits, const vector<int>& target_ids);
	void normalize_gradients(int batch_size);
	void zero_all_gradients();
	void train_step(const vector<string>& batch_texts);
	double validate(const vector<string>& val_texts);

	// Генерация
	string generate_response(const string& input_text, int max_length = 100);

	// Утилиты
	void analyze_tokenization(const string& text);
	void interactive_chat();

	// Совместимость со старым интерфейсом
	void save_tensor(json& j, const std::string& key, Tensor& tensor);
	void load_tensor(Tensor& tensor, const json& j, const std::string& key, int rows, int cols);

	void Save_File_Communication_RUS();
	void Load_File_Communication_RUS();
};

// Архитектурные константы
static const int EMBEDDING_DIM = 300;
static const int NUM_HEADS = 8;
static const int FF_DIM = 1024;
static const int NUM_LAYERS = 6;
static const int MAX_SEQ_LENGTH = 128;

static int VOCAB_SIZE;