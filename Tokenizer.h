// файл Tokenizer.h ////

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <codecvt>
#include <locale>
#include <nlohmann/json.hpp>
#include <fasttext.h>
#include <memory> // Для unique_ptr

using namespace std;
extern string Path_directory_Quark;


class AdvancedTokenizer 
{
private:
    unordered_map<string, int> token_to_id;
    unordered_map<int, string> id_to_token;
    int next_id = 0;
    unique_ptr<fasttext::FastText> ft_model; // Используем unique_ptr вместо объекта

    static const string PAD_TOKEN;
    static const string UNK_TOKEN;
    static const string BOS_TOKEN;
    static const string EOS_TOKEN;
    static const string USER_TOKEN; 
    static const string BOT_TOKEN; 
  

public:

    AdvancedTokenizer();
    AdvancedTokenizer(const AdvancedTokenizer& other); // Конструктор копирования
    AdvancedTokenizer& operator=(const AdvancedTokenizer& other); // Оператор присваивания

    vector<string> getUTF8Characters(const string& text);

    void addToken(const string& token);

    void train_tokenizer(const vector<string>& texts);


    vector<int> encode(const string& text);
    string decode(const vector<int>& tokens) const;

    int getVocabSize() const;
    int getPadTokenId() const;
    int getBosTokenId() const;
    int getEosTokenId() const;


    void Save_File_Tokenizer_Safe();
    void Debug_Print_Tokenizer();

    void Load_File_Tokenizer();

    // Методы для fastText
    void load_fasttext_model(const string& model_path);
    vector<double> get_word_embedding(const string& word, int embedding_dim);


    vector<string> split(const string& text);
    double cosine_similarity(const fasttext::Vector& a, const fasttext::Vector& b);
    int find_closest_token(const fasttext::Vector& vec);



    
};