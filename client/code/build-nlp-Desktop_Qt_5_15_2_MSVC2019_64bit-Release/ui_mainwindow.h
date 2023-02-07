/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QLineEdit *query;
    QLineEdit *answer;
    QPushButton *queryButton;
    QLabel *back;
    QLabel *score;
    QPushButton *scoreButton;

    void setupUi(QWidget *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1280, 530);
        MainWindow->setMinimumSize(QSize(1280, 530));
        MainWindow->setMaximumSize(QSize(1280, 530));
        MainWindow->setStyleSheet(QString::fromUtf8(""));
        query = new QLineEdit(MainWindow);
        query->setObjectName(QString::fromUtf8("query"));
        query->setGeometry(QRect(210, 30, 851, 91));
        query->setStyleSheet(QString::fromUtf8("font: 18pt \"\345\215\216\346\226\207\344\270\255\345\256\213\";"));
        query->setAlignment(Qt::AlignCenter);
        answer = new QLineEdit(MainWindow);
        answer->setObjectName(QString::fromUtf8("answer"));
        answer->setGeometry(QRect(210, 150, 851, 91));
        answer->setStyleSheet(QString::fromUtf8("font: 18pt \"\345\215\216\346\226\207\344\270\255\345\256\213\";"));
        answer->setAlignment(Qt::AlignCenter);
        queryButton = new QPushButton(MainWindow);
        queryButton->setObjectName(QString::fromUtf8("queryButton"));
        queryButton->setGeometry(QRect(410, 420, 221, 81));
        queryButton->setStyleSheet(QString::fromUtf8("font: 18pt \"\345\215\216\346\226\207\344\270\255\345\256\213\";"));
        back = new QLabel(MainWindow);
        back->setObjectName(QString::fromUtf8("back"));
        back->setGeometry(QRect(0, 0, 1281, 531));
        back->setAutoFillBackground(false);
        back->setStyleSheet(QString::fromUtf8(""));
        back->setFrameShape(QFrame::Box);
        back->setLineWidth(3);
        back->setScaledContents(true);
        score = new QLabel(MainWindow);
        score->setObjectName(QString::fromUtf8("score"));
        score->setGeometry(QRect(540, 290, 201, 81));
        score->setStyleSheet(QString::fromUtf8("font: 75 20pt \"Times New Roman\";"));
        score->setAlignment(Qt::AlignCenter);
        scoreButton = new QPushButton(MainWindow);
        scoreButton->setObjectName(QString::fromUtf8("scoreButton"));
        scoreButton->setGeometry(QRect(660, 420, 221, 81));
        scoreButton->setStyleSheet(QString::fromUtf8("font: 18pt \"\345\215\216\346\226\207\344\270\255\345\256\213\";"));
        back->raise();
        query->raise();
        answer->raise();
        queryButton->raise();
        score->raise();
        scoreButton->raise();

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QWidget *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        query->setText(QString());
        queryButton->setText(QCoreApplication::translate("MainWindow", "\347\224\237\346\210\220", nullptr));
        back->setText(QString());
        score->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
        scoreButton->setText(QCoreApplication::translate("MainWindow", "\350\257\204\345\210\206", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
