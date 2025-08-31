#pragma once
#include <string>
#include <vector>

using namespace std;


// Forward declaration
class TransformerModel;

class TrainingSystem
{
private:
    vector<string> training_texts;

public:
    void loadTrainingData(const string& filename);

    // �������� ������ (����� ������)
    void enhanced_training(TransformerModel& model, int epochs);

    
};

// ��������������� ������� ��� ���������� �������� ����
void UpdateEpochCount();
