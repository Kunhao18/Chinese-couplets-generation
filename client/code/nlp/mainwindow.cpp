#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>


MainWindow::MainWindow(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    mainNet = new NetManager();
    connect(mainNet, &NetManager::ReturnResult, this, &MainWindow::RecvStr);
    connect(mainNet, &NetManager::ReturnScore, this, &MainWindow::RecvScore);
    connect(this, &MainWindow::SendQuery, mainNet, &NetManager::QuerySentence);
    connect(this, &MainWindow::SendScore, mainNet, &NetManager::ScoreSentence);
    connect(ui->queryButton, &QPushButton::clicked, this, &MainWindow::QueryStr);
    connect(ui->scoreButton, &QPushButton::clicked, this, &MainWindow::ScoreStr);
    connect(this, &MainWindow::StartPlay, this, &MainWindow::PlayGif);
    ui->back->setScaledContents(true);
    ui->back->setPixmap(QPixmap(":/new/prefix1/back.jpg"));
    ui->score->setText("");
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::QueryStr() {
    if (isCheck == true)
        return;
    isCheck = true;
    QString query = ui->query->text();
    if (query.size() < 1) {
        isCheck = false;
        QMessageBox::StandardButton msg = QMessageBox::information(NULL, "Error", "Length zero.");
        return;
    }
    emit SendQuery(query);
}


void MainWindow::ScoreStr() {
    if (isCheck == true)
        return;
    isCheck = true;
    QString pre = ui->query->text();
    QString pro = ui->answer->text();
    if (pre.size() != pro.size()) {
        isCheck = false;
        QMessageBox::StandardButton msg = QMessageBox::information(NULL, "Error", "Length mismatch.");
        return;
    }
    emit SendScore(pre, pro);
}


void MainWindow::ClearLabel() {
    for (int i = 0; i < 20; i ++) {
        if (labelList[i]) {
            delete labelList[i];
            labelList[i] = NULL;
        }
        if (gifList[i]) {
            delete gifList[i];
            gifList[i] = NULL;
        }
    }
}


void MainWindow::PlayGif() {
    int idx = nowPlay;
    if (idx == movieNum) {
        isCheck = false;
        return;
    }
    if (gifList[idx] == NULL) {
        nowPlay++;
        emit PlayGif();
        return;
    }
    //qDebug() << idx;
    connect(gifList[idx], &QMovie::frameChanged, this, &MainWindow::TerminateGif);
    gifList[idx]->setSpeed(800);
    gifList[idx]->start();
}


void MainWindow::TerminateGif(int frame) {
    QMovie* tmp = (QMovie*)sender();
    if (frame == tmp->frameCount() - 1) {
        tmp->stop();
        nowPlay++;
        emit PlayGif();
    }
}


void MainWindow::RecvStr(QString _result) {
    //_result = QString::fromLocal8Bit("小井不节大");
    ui->answer->setText(_result);
    int len = _result.size();
    qDebug() << len;
    ClearLabel();
    double offset = 640 - (double)len / 2 * 100;
    movieNum = len;
    for (int i = 0; i < len; i ++) {

        QString filePath = QString::asprintf("D:/Program/Python/grabber/bihua/%s.gif", _result.mid(i).left(1).toStdString().c_str());
        qDebug() << filePath;
        gifList[i] = new QMovie(filePath);
        if (!gifList[i]->isValid())
            gifList[i] = NULL;
        labelList[i] = new QLabel(this);
        labelList[i]->setGeometry(offset + i * 100, 280, 100, 100);
        labelList[i]->setScaledContents(true);
        labelList[i]->setMovie(gifList[i]);
        labelList[i]->show();
        if (gifList[i]) {
            gifList[i]->start();
            gifList[i]->stop();
        }
    }
    nowPlay = 0;
    emit StartPlay();
}


void MainWindow::RecvScore(QString _result) {
    //_result = QString::fromLocal8Bit("小井不节大");
    ClearLabel();
    ui->score->setText(_result);
    isCheck = false;
}
