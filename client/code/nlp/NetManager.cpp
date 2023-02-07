#include "NetManager.h"
#include <QDebug>


const char* SERVER_IP = "127.0.0.1";
const int SERVER_PORT = 5234;


//初始化套接字库
void WinSockInitialization() {
    //初始化套接字库
    WORD w_req = MAKEWORD(2, 2);//版本号
    WSADATA wsadata;
    int err;
    err = WSAStartup(w_req, &wsadata);
    if (err != 0) {
        qDebug() << QString::fromLocal8Bit("初始化套接字库失败！");
    }
    else {
        qDebug() << QString::fromLocal8Bit("初始化套接字库成功！");
    }
    //检测版本号
    if (LOBYTE(wsadata.wVersion) != 2 || HIBYTE(wsadata.wHighVersion) != 2) {
        qDebug() << QString::fromLocal8Bit("套接字库版本号不符！");
        WSACleanup();
    }
    else {
        qDebug() << QString::fromLocal8Bit("套接字库版本正确！");
    }
    //填充服务端地址信息
}


//构造
NetManager::NetManager() {
    WinSockInitialization();
    mySocket = new SOCKET;

    SOCKADDR_IN server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.S_un.S_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(SERVER_PORT);

    *mySocket = socket(AF_INET, SOCK_STREAM, 0);
    if (::connect(*mySocket, (SOCKADDR *)&server_addr, sizeof(SOCKADDR)) == SOCKET_ERROR) {
        qDebug() << QString::fromLocal8Bit("服务器连接失败！");
        WSACleanup();
    }
    else {
        qDebug() << QString::fromLocal8Bit("服务器连接成功！");
    }
}


void NetManager::ScoreSentence(QString _pre, QString _pro) {
    char sendBuf[256] = {0};
    int i = _pre.size();
    int j = _pro.size();
    int opCode = 0;
    QByteArray bytes_pre = _pre.toUtf8();
    QByteArray bytes_pro = _pro.toUtf8();
    int offset = 0;
    memcpy(sendBuf, &opCode, sizeof(int));
    offset += sizeof(int);
    memcpy(sendBuf + offset, &i, sizeof(int));
    offset += sizeof(int);
    memcpy(sendBuf + offset, &bytes_pre.data()[0], i * 3);
    offset += i * 3;
    memcpy(sendBuf + offset, &j, sizeof(int));
    offset += sizeof(int);
    memcpy(sendBuf + offset, &bytes_pro.data()[0], j * 3);
    send(*mySocket, sendBuf, 256, 0);

    char recvBuf[256] = {0};
    recv(*mySocket, recvBuf, 256, 0);
    QByteArray bytes = QByteArray(recvBuf);
    QString str = QString(bytes);
    emit ReturnScore(str);
}


void NetManager::QuerySentence(QString _query) {
    char sendBuf[256] = {0};
    int i = _query.size();
    int opCode = 1;
    QByteArray bytes = _query.toUtf8();
    qDebug() << bytes.size();
    int offset = 0;
    memcpy(sendBuf, &opCode, sizeof(int));
    offset += sizeof(int);
    memcpy(sendBuf + offset, &i, sizeof(int));
    offset += sizeof(int);
    memcpy(sendBuf + offset, &bytes.data()[0], i * 3);
    send(*mySocket, sendBuf, 256, 0);

    char recvBuf[256] = {0};
    recv(*mySocket, recvBuf, 256, 0);
    bytes = QByteArray(recvBuf);
    QString str = QString(bytes);
    emit ReturnResult(str);
}
