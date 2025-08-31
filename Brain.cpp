/*
   Блок Barin.cpp (Main)
*/

// Библиотеки
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <cmath>
#include <algorithm>
#include <locale>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <set>
#include <codecvt>
#include <cwchar> 
#include <filesystem>

#include <io.h>
#include <fcntl.h>
#include <conio.h>  // для kbhit() и getch()
#include <windows.h>



// Подключаем модули
#include "Math.h"
#include "Tokenizer.h"
#include "Transformer.h"
#include "Training.h"

#pragma comment(lib, "Shell32.lib")

 
using namespace std;
string Path_directory_Quark; //Глобальный путь в директорию
int DEBUG = 1;

// Чтение по символам через _getch() для понимания UTF-8
string read_UTF_8() 
{
    string text;  char c;
    while ((c = _getch()) != '\r') { // '\r' это Enter
        if (c == '\b') { // Backspace
            if (!text.empty()) {
                text.pop_back();
                cout << "\b \b"; // стираем символ на экране
            }
        }
        else {
            text += c;
            cout << c; // выводим символ на экран
        }
    }
    cout << endl;
    return text;
}



int main(int argc, char** argv)
{
    {
        setlocale(LC_ALL, ".UTF8");      // setlocale(LC_ALL, "Russian"); 
        SetConsoleCP(CP_UTF8);           // CP_UTF8 == 65001
        SetConsoleOutputCP(CP_UTF8);

        //Path_directory_Quark
        filesystem::path Path_directory = filesystem::canonical(filesystem::path(argv[0])).remove_filename();
        Path_directory_Quark = Path_directory.string(); // Записываем путь
        wcout << L"\n [ путь в директорию Quark ] "; cout << Path_directory_Quark << "\n\n";
    }

    wcout << L"=== QUARK TRANSFORMER NEURAL NETWORK ===" << endl;
    wcout << L"Создание живой нейросети на базе Transformer архитектуры" << endl;

    // Создаем модель
    TransformerModel model;
    TrainingSystem trainer;
    AdvancedTokenizer tokanizer;

    wcout << L"\nВключить общую отладку? (1 - да/ 2 - нет) [- Не влияет на отладку ошибок -] \nВвод: ";
    cin >> DEBUG;
    if(DEBUG == 1){ wcout << L"\n Отладка включена!\n"; }else if (DEBUG != 1) { wcout << L"\n Отладка отключена.\n"; }


    wcout << L"\nЗагружать файлы обученной модели? (1 - да/ 2 - нет) \n[Выполнять в том случае если есть уже обученная модель] \n[ ОСТОРОЖНО (НЕТ) ПЕРЕЗАПИШЕТ УЖЕ СУЩЕСТВУЮЩИЙ ФАЙЛ ] \nВвод: ";

    int Load; cin >> Load;
    if (Load == 1)
    {
        model.Load_File_Communication_RUS();
    }
    else
    {
        wcout << L"\nУверены что хотите создать новый файл и стереть уже существующий? (1 - да/ 2 - нет)\nВвод: ";
        int Load; cin >> Load;
        if (Load == 1)
        {
            
        }
        else 
        { 
            model.Load_File_Communication_RUS();
        }
       
    }


    wcout << L"\nЗагрузить существующие токены? (1 - да/ 2 - нет) [Выполнять в том случае если есть обученые токены] \nВвод: ";
    int Load_Token; cin >> Load_Token;
    if (Load_Token == 1)
    {
        wcout << L"\nЗагружаем существующие токены ";

        tokanizer.Load_File_Tokenizer();  //загружаем существующие токены
        tokanizer.load_fasttext_model(Path_directory_Quark + "Brain\\Embeddings_RU.bin");  //загружаем эмбеддинги
        
        model.setTokenizer(tokanizer); //передаём токены в модель
        tokanizer.Debug_Print_Tokenizer();
    }
    else
    {
        wcout << L"\nНачать обучение токенов? (1 - да/ 2 - нет) [ Перезаписывает текущий файл токенов на новый] \nВвод: ";
        int Train_Token;  cin >> Train_Token;

        if (Train_Token == 1)
        {
            ifstream file(Path_directory_Quark + "Training_materials\\Tokenizer_Text.txt");
            vector<string> lines;

            if (!file.is_open())
            {
                wcout << L"Не удалось открыть файл Tokenizer_Text.txt" << endl;
            }
            else 
            {
                string line;
                while (getline(file, line)) 
                {
                    if (!line.empty()) 
                    {
                        lines.push_back(line);
                    }
                }
                file.close();
            }


            tokanizer.train_tokenizer(lines);

          
            tokanizer.Save_File_Tokenizer_Safe();  //сохраняем новые токены

            tokanizer.Load_File_Tokenizer();  //загружаем существующие токены
            tokanizer.load_fasttext_model(Path_directory_Quark + "Brain\\Embeddings_RU.bin");  //загружаем эмбеддинги

            model.setTokenizer(tokanizer); //передаём токены в модель

            tokanizer.Debug_Print_Tokenizer();
        }
    }




    wcout << L"\n\nЗапустить обучение? (1 - да/ 2 - нет): ";
    int Train; cin >> Train;
    if (Train == 1)
    {
        // Загружаем данные для обучения
        trainer.loadTrainingData(Path_directory_Quark + "Training_materials\\Russian_word(0k).txt");

        wcout << L"Количество эпох: ";
        int epochs;
        cin >> epochs;

        trainer.enhanced_training(model, epochs);

        model.Save_File_Communication_RUS();

    }
    else 
    { 
        wcout << L"\n Обучение пропущенно \n";
    }

    cin >> Train;  system("cls");

    // Режим общения
    wcout << L"\n=== РЕЖИМ ОБЩЕНИЯ С НЕЙРОСЕТЬЮ ===" << endl;
    cin.ignore(); // Очищаем буфер

    string user_input;
    while (true)
    {

        wcout << L"\n Вы: ";
        user_input = read_UTF_8();

        //wcout << L"\n == Анализ текста до generate_response : \n";
         //model.analyze_tokenization(user_input);



            if (!user_input.empty())
            {

                auto start_time = chrono::high_resolution_clock::now();
                string response = model.generate_response(user_input, 50);
                auto end_time = chrono::high_resolution_clock::now();

                auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
                wcout << L" Кварк: "; cout << response << endl;
                // Время
                {
                    // Выводим в миллисекундах
                    wcout << L"\n (Ответ сгенерирован за " << duration.count() << L" мс)" << endl;

                    // Вычисляем минуты и оставшиеся миллисекунды
                    long long total_ms = duration.count();
                    int minutes = static_cast<int>(total_ms / 60000);
                    int remaining_ms = static_cast<int>(total_ms % 60000);


                    // Выводим в формате "X минут Y мс"
                    wcout << L"\n (Ответ сгенерирован за "  << minutes << L" минут " << remaining_ms << L" мс)" << endl;
                }


            }
        
    }

    return 0;
}

