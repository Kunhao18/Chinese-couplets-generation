#ifndef NETMANAGER_H
#define NETMANAGER_H

#include <QObject>
#include <QString>
#include <winsock.h>


class NetManager: public QObject {
    Q_OBJECT
public:
    NetManager();
    void QuerySentence(QString);
    void ScoreSentence(QString, QString);

signals:
    void ReturnResult(QString);
    void ReturnScore(QString);

private:
    SOCKET* mySocket;
};


#endif // NETMANAGER_H
