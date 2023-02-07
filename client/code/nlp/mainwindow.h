#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include <QLabel>
#include <QMovie>

#include "NetManager.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QWidget
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void RecvStr(QString);
    void RecvScore(QString);

    void ScoreStr();
    void QueryStr();

    void ClearLabel();
    void PlayGif();
    void TerminateGif(int);

signals:
    void SendQuery(QString);
    void SendScore(QString, QString);
    void StartPlay();

private:
    Ui::MainWindow *ui;
    NetManager* mainNet;

    QLabel* labelList[20] = {NULL};
    QMovie* gifList[20] = {NULL};
    int movieNum = 0;
    bool isCheck;
    int nowPlay = 0;
};
#endif // MAINWINDOW_H
