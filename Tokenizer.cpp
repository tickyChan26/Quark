// файл Tokenizer.cpp классы обьявлены в Tokenizer.h ////

#include "Tokenizer.h"

#include <set>
#include <algorithm>
#include <string>
#include <vector>
#include <filesystem>
 
#include <iostream>
#include <fstream>
#include <iomanip>
#include <codecvt>
#include <locale>

#include <nlohmann/json.hpp>


 
using namespace std;
using json = nlohmann::json;

// Определение статических констант
const string AdvancedTokenizer::PAD_TOKEN = "<PAD>";
const string AdvancedTokenizer::UNK_TOKEN = "<UNK>";
const string AdvancedTokenizer::BOS_TOKEN = "<BOS>";
const string AdvancedTokenizer::EOS_TOKEN = "<EOS>";

const string AdvancedTokenizer::USER_TOKEN = "<USER>"; 
const string AdvancedTokenizer::BOT_TOKEN = "<BOT>";   

// Конструктор - добавляем спецтокены
AdvancedTokenizer::AdvancedTokenizer()
{
    addToken(PAD_TOKEN);
    addToken(UNK_TOKEN);
    addToken(BOS_TOKEN);
    addToken(EOS_TOKEN);

    addToken(USER_TOKEN); 
    addToken(BOT_TOKEN);

    ft_model = make_unique<fasttext::FastText>(); // Инициализируем unique_ptr
}


AdvancedTokenizer::AdvancedTokenizer(const AdvancedTokenizer& other)
    : token_to_id(other.token_to_id),
    id_to_token(other.id_to_token),
    next_id(other.next_id),
    ft_model(make_unique<fasttext::FastText>()) {
    // ft_model заново создаётся пустой; загрузка модели должна быть вызвана явно
}

AdvancedTokenizer& AdvancedTokenizer::operator=(const AdvancedTokenizer& other) {
    if (this != &other) {
        token_to_id = other.token_to_id;
        id_to_token = other.id_to_token;
        next_id = other.next_id;
        ft_model = make_unique<fasttext::FastText>();
        // ft_model заново создаётся пустой; загрузка модели должна быть вызвана явно
    }
    return *this;
}


// Загрузка fastText модели
void AdvancedTokenizer::load_fasttext_model(const string& model_path)
{
    wcout << L" === Загрузка fastText модели: " << model_path.c_str() << endl;
    try 
    {
        ft_model->loadModel(model_path);
        wcout << L" --- Модель fastText успешно загружена" << endl;
    }
    catch (const exception& e) 
    {
        wcout << L"Ошибка загрузки fastText: " << e.what() << endl;
        throw;
    }
}

// Получение эмбеддинга для слова
vector<double> AdvancedTokenizer::get_word_embedding(const string& word, int embedding_dim) {
    fasttext::Vector vec(embedding_dim);
    ft_model->getWordVector(vec, word);
    vector<double> embedding(vec.data(), vec.data() + vec.size());
    if (embedding.empty() || all_of(embedding.begin(), embedding.end(), [](double x) { return x == 0.0; })) {
        wcout << L"Эмбеддинг пустой (возможно, спецтокен), возвращаем нулевой вектор" << endl;
        return vector<double>(embedding_dim, 0.0);
    }
    return embedding;
}


//  очень полезная функция которая даёт возможность работать с двобайтовым UTF-8
vector<string> AdvancedTokenizer::getUTF8Characters(const string& text)
{
    vector<string> characters;
    size_t i = 0;

    while (i < text.size())
    {
        int len = 1;
        unsigned char c = text[i];

        if ((c & 0x80) == 0) {
            len = 1;
        }
        else if ((c & 0xE0) == 0xC0) {
            len = 2;
        }
        else if ((c & 0xF0) == 0xE0) {
            len = 3;
        }
        else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }

        if (i + len <= text.size()) {
            characters.push_back(text.substr(i, len));
        }
        i += len;
    }

    return characters;
}

void AdvancedTokenizer::train_tokenizer(const vector<string>& texts) {
    cout << "Строим словарь из " << texts.size() << " текстов..." << endl;

    set<string> unique_chars;
    for (const auto& text : texts) {
        vector<string> chars = getUTF8Characters(text);
        for (const auto& ch : chars) {
            unique_chars.insert(ch);
        }
    }

    for (const auto& ch : unique_chars) {
        addToken(ch);
    }

    unordered_map<string, int> bigram_counts;
    for (const auto& text : texts) {
        vector<string> chars = getUTF8Characters(text);
        for (size_t i = 0; i + 1 < chars.size(); i++) {
            string bigram = chars[i] + chars[i + 1];
            bigram_counts[bigram]++;
        }
    }

    vector<pair<string, int>> sorted_bigrams(bigram_counts.begin(), bigram_counts.end());
    sort(sorted_bigrams.begin(), sorted_bigrams.end(),
        [](const pair<string, int>& a, const pair<string, int>& b) {
            return a.second > b.second;
        });

    size_t limit = min<size_t>(1000, sorted_bigrams.size());
    for (size_t i = 0; i < limit; i++) {
        addToken(sorted_bigrams[i].first);
    }

    cout << "Словарь создан. Размер: " << next_id << " токенов" << endl;


}
void AdvancedTokenizer::addToken(const string& token) 
{
    if (token_to_id.find(token) == token_to_id.end()) 
    {
        token_to_id[token] = next_id;
        id_to_token[next_id] = token;
        next_id++;
    }
}


double AdvancedTokenizer::cosine_similarity(const fasttext::Vector& a, const fasttext::Vector& b) 
{
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < a.size(); ++i)
    {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    double norm = sqrt(norm_a) * sqrt(norm_b);
    return norm > 0.0 ? dot / norm : 0.0;
}
int AdvancedTokenizer::find_closest_token(const fasttext::Vector& vec) 
{
    double max_sim = -1.0;
    int best_token = token_to_id[UNK_TOKEN];
    for (const auto& [word, id] : token_to_id) {
        fasttext::Vector word_vec(300); // Предполагаем 300-мерные эмбеддинги
        ft_model->getWordVector(word_vec, word);
        if (!all_of(word_vec.data(), word_vec.data() + word_vec.size(), [](float x) { return x == 0.0; })) {
            double sim = cosine_similarity(vec, word_vec);
            if (sim > max_sim) {
                max_sim = sim;
                best_token = id;
            }
        }
    }
    return best_token;
}

// Обновлённый encode
vector<int> AdvancedTokenizer::encode(const string& text) {
    vector<int> result;
    result.push_back(token_to_id[BOS_TOKEN]);
    wcout << L"\nТекст в encode: ";
    cout << text << endl;
    vector<string> utf8_chars = getUTF8Characters(text);
    for (size_t i = 0; i < utf8_chars.size(); i++) {
        bool found_bigram = false;
        if (i + 1 < utf8_chars.size()) {
            string bigram = utf8_chars[i] + utf8_chars[i + 1];
            if (token_to_id.find(bigram) != token_to_id.end()) {
                result.push_back(token_to_id[bigram]);
                i++;
                found_bigram = true;
            }
        }
        if (!found_bigram) {
            if (token_to_id.find(utf8_chars[i]) != token_to_id.end()) {
                result.push_back(token_to_id[utf8_chars[i]]);
            }
            else if (ft_model) {
                fasttext::Vector vec(300); // Предполагаем 300-мерные эмбеддинги
                ft_model->getWordVector(vec, utf8_chars[i]);
                if (!all_of(vec.data(), vec.data() + vec.size(), [](float x) { return x == 0.0; })) {
                    int closest_token = find_closest_token(vec);
                    result.push_back(closest_token);
                }
                else {
                    result.push_back(token_to_id[UNK_TOKEN]);
                }
            }
            else {
                result.push_back(token_to_id[UNK_TOKEN]);
            }
        }
    }
    result.push_back(token_to_id[EOS_TOKEN]);
    return result;
}
string AdvancedTokenizer::decode(const vector<int>& tokens) const {
    string result;
    for (int token_id : tokens) {
        if (id_to_token.find(token_id) != id_to_token.end()) {
            string token = id_to_token.at(token_id); // Используем at() вместо []
            if (token == BOS_TOKEN || token == EOS_TOKEN || token == PAD_TOKEN)
                continue;
            // Добавляем только печатные символы и пробелы
            bool valid = true;
            for (char c : token) {
                if ((unsigned char)c < 0x20 && c != '\n') {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                result += token;
            }
        }
        else {
            result += '?';
        }
    }
    return result;
}


int AdvancedTokenizer::getVocabSize() const {
    return next_id;
}
int AdvancedTokenizer::getPadTokenId() const {
    return token_to_id.at(PAD_TOKEN);
}
int AdvancedTokenizer::getBosTokenId() const {
    return token_to_id.at(BOS_TOKEN);
}
int AdvancedTokenizer::getEosTokenId() const {
    return token_to_id.at(EOS_TOKEN);
}



// Функция для отладки - показывает содержимое токенизатора
void AdvancedTokenizer::Debug_Print_Tokenizer()
{
    wcout << L"=== ОТЛАДКА ТОКЕНИЗАТОРА ===" << endl;
    wcout << L"Размер token_to_id: "; cout << token_to_id.size() << endl;
    wcout << L"Размер id_to_token: "; cout << id_to_token.size() << endl;
    wcout << L"Следующий ID: "; cout << next_id << endl;

    for (int i = 0; i < next_id; ++i)
    {
        auto it = id_to_token.find(i);
        if (it != id_to_token.end())
        {
            cout << "  ID " << i << " -> \"" << it->second << "\"" << endl;
        }
        else
        {
            cout << "  ID " << i; wcout << L" -> [пусто или отсутствует]" << endl;
        }
    }

    // Проверяем на некорректные данные
    bool has_issues = false;
    for (const auto& pair : token_to_id)
    {
        if (pair.first.empty())
        {
            wcout << L"\nПРОБЛЕМА: Найден пустой токен с ID "; cout << pair.second << endl;
            has_issues = true;
        }
        if (pair.second < 0)
        {
            wcout << L"\nПРОБЛЕМА: Найден негативный ID "; cout << pair.second; wcout << L" для токена \""; cout << pair.first << "\"" << endl;
            has_issues = true;
        }
    }

    if (!has_issues)
    {
        wcout << L"\nДанные токенизатора выглядят корректно" << endl;
    }

    wcout << L"==============================" << endl;
}


void AdvancedTokenizer::Save_File_Tokenizer_Safe()
{
    wcout << L"\nСохраняем токены" << endl;

    try
    {
        // Создаем временные контейнеры для проверенных данных
        map<string, int> safe_token_to_id;
        map<int, string> safe_id_to_token;

        // Фильтруем данные
        for (const auto& pair : token_to_id)
        {
            if (!pair.first.empty() && pair.second >= 0)
            {
                safe_token_to_id[pair.first] = pair.second;
            }
        }
        for (const auto& pair : id_to_token)
        {
            if (!pair.second.empty() && pair.first >= 0)
            {
                safe_id_to_token[pair.first] = pair.second;
            }
        }

        json j;
        j["token_to_id"] = safe_token_to_id;
        j["id_to_token"] = safe_id_to_token;
        j["next_id"] = next_id;

        string filepath = Path_directory_Quark + "Brain\\Tokenizer.json";
        ofstream file(filepath);

        if (!file.is_open())
        {
            cerr << "Не удалось открыть файл для записи: " << filepath << endl;
            return;
        }

        file << j.dump(4);
        file.close();

        wcout << L"Токены сохранены безопасно. Количество: "; cout << safe_token_to_id.size() << endl;
    }
    catch (const exception& e)
    {
        wcout << L"Ошибка в безопасном сохранении: "; cout << e.what() << endl;
    }

}

void AdvancedTokenizer::Load_File_Tokenizer() 
{
    wcout << L"\nЗагружем токены" << endl;
    ifstream file(Path_directory_Quark + "Brain\\Tokenizer.json");
    if (!file.is_open()) {
        wcout << L"\nНе удалось открыть готовый файл токенов " << endl;
        return;
    }

    
    json j;
    file >> j;
    
    token_to_id = j["token_to_id"].get<unordered_map<string, int>>();
    
    id_to_token = j["id_to_token"].get<unordered_map<int, string>>();
    
    next_id = j["next_id"].get<int>();

    wcout << L"\nЗагружено: "; cout << token_to_id.size();  wcout << L" токенов " << endl;



}

