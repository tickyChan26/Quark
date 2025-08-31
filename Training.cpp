#include "Training.h"
#include "Transformer.h"  // для TransformerModel

#include <conio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>

using namespace std;

// Внешняя переменная с путём к директории
extern string Path_directory_Quark;


//Загрузка данных для обучения
void TrainingSystem::loadTrainingData(const string& filename)
{
    ifstream file(filename);

    if (!file.is_open()) {
        wcout << L"Не удалось открыть файл для обучения\n ";
    }
    else 
    {
        training_texts.clear();
        string line;
        while (getline(file, line)) 
        {
            if (!line.empty()) 
            {
                training_texts.push_back(line);
            }
        }
        file.close();
    }


   


    wcout << L"\n\nЗагружено "; cout << training_texts.size(); wcout << L" примеров для обучения" << endl;
}

void TrainingSystem::enhanced_training(TransformerModel& model, int epochs)
{
    wcout << L" ЗАПУСК ПРОДВИНУТОГО ОБУЧЕНИЯ " << endl;
    wcout << L"====================================" << endl;
    auto start_time = chrono::high_resolution_clock::now();

    const vector<string>& train_texts = training_texts; 

    if (train_texts.empty()) 
    {
        wcout << L"Ошибка: нет данных для обучения!" << endl;
        return;
    }

   


    // Разделяем на train и validation (10%)
    int val_size = static_cast<int>(train_texts.size() * 0.1);
    int train_size = static_cast<int>(train_texts.size()) - val_size;

    vector<string> train_set(train_texts.begin(), train_texts.begin() + train_size);
    vector<string> val_set(train_texts.begin() + train_size, train_texts.end());

    wcout << L" Размер обучающей выборки: "; cout << train_size << endl;
    wcout << L" Размер валидационной выборки: "; cout << val_size << endl;


    double best_val_loss = 1e9;
    int patience_counter = 0;
    const int PATIENCE = 3;

    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        auto epoch_start = chrono::high_resolution_clock::now();

        wcout << L"\n ЭПОХА "; cout << (epoch + 1) << "/" << epochs << endl;
        wcout << L"========================" << endl;


        const int batch_size = 8;
        for (int i = 0; i < train_size; i += batch_size) 
        {
            vector<string> batch;
            for (int j = i; j < min(i + batch_size, train_size); j++) 
            {
                batch.push_back(train_set[j]);
            }

            model.train_step(batch);

            // Вывод прогресса
            if (((i / batch_size) % 10) == 0) 
            {
                wcout << L" Обработано батчей: ";
                cout << (i / batch_size + 1) << "/" << ((train_size + batch_size - 1) / batch_size) << endl;
            }
        }


        // Валидация
        if (!val_set.empty()) 
        {
            wcout << L" Валидация..." << endl;
            double val_loss = model.validate(val_set);  // Предполагается метод validate
            cout << " Validation Loss: " << fixed << setprecision(6) << val_loss << endl;

            if (val_loss < best_val_loss) 
            {
                best_val_loss = val_loss;
                patience_counter = 0;
                wcout << L"- - Новый лучший результат! Сохраняю модель." << endl;
                model.Save_File_Communication_RUS();
            }
            else 
            {
                patience_counter++;
                wcout << L" Валидационная потеря не улучшилась. Patience: " << patience_counter << "/" << PATIENCE << endl;
              
                if (patience_counter >= PATIENCE) 
                {
                    wcout << L" Early stopping! Загружаю лучшую модель..." << endl;
                    model.Load_File_Communication_RUS();
                    break;
                }
            }
        }



        auto epoch_end = chrono::high_resolution_clock::now();
        auto epoch_duration = chrono::duration_cast<chrono::seconds>(epoch_end - epoch_start);
        wcout << L" Эпоха завершена за "; cout << epoch_duration.count(); wcout << L" секунд" << endl;

        // остановить обучение
        if (_kbhit()) {
            int ch = _getch();
            wcout << L"\nОбучение остановлено пользователем" << endl;
            break;
        }


    }

    // технический блок (просчитка времени)
    {

    wcout << L"\n ОБУЧЕНИЕ ЗАВЕРШЕНО! " << endl;
    wcout << L"=========================" << endl;
  
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);
    int64_t total_seconds = duration.count();  // безопасно
    int minutes = static_cast<int>(total_seconds / 60);
    int seconds = static_cast<int>(total_seconds % 60);

        wcout << L"Обучение завершено за:   "; cout << minutes << " min " << seconds << " sec" << endl;
    }

}

// Обновление счётчика эпох
void UpdateEpochCount()
{
    int current_epoch = 0;

    ifstream infile(Path_directory_Quark + "Brain\\Epochs.txt");
    if (infile.is_open()) {
        string label;
        infile >> label >> current_epoch;  // ожидается "Epochs N"
        infile.close();
    }

    current_epoch++;

    ofstream outfile(Path_directory_Quark + "Brain\\Epochs.txt");
    if (outfile.is_open()) {
        outfile << "Epochs " << current_epoch << endl;
        outfile.close();
    }
    else {
        wcout << L"Не удалось открыть файл для записи количества эпох" << endl;
    }
}




