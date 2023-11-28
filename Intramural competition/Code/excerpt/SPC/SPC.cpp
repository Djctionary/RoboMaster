#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>
using namespace std;

int fd;

/**
 * openUart
 * @param  comport 想要打开的串口号
 * return  失败返回-1
 */
int openUart( int comport)
{
    const char *dev[]  = {"/dev/ttyUSB0"};
    //瑞泰科技只留出来两路串口，UART0为调试口，UART1为普通串口，所以咱们使用UART1
    if(comport == 0)
    {
        fd = open(dev[0], O_RDWR | O_NOCTTY | O_NDELAY);
        if(-1 == fd)
        {
            perror("Can't Open Serial Port");
            return (-1);
        }
    }
    printf("fd-open=%d\n", fd);
    return fd;
}


/**
 * uartInit
 * @param  nSpeed 波特率  nBits 数据位 nEvent 奇偶校验位 nStop 停止位
 * @return  返回-1为初始化失败
 */
int uartInit(int nSpeed, int nBits, char nEvent, int nStop)
{
    struct termios newtio, oldtio;
    /*保存测试现有串口参数设置，在这里如果串口号等出错，会有相关的出错信息*/
    if  ( tcgetattr( fd, &oldtio)  !=  0) {
        perror("SetupSerial 1");
        printf("tcgetattr( fd,&oldtio) -> %d\n", tcgetattr( fd, &oldtio));
        return -1;
    }
    bzero( &newtio, sizeof( newtio ) );
    /*步骤一，设置字符大小*/
    newtio.c_cflag  |=  CLOCAL | CREAD;
    newtio.c_cflag &= ~CSIZE;
    /*设置停止位*/
    switch( nBits )
    {
    case 7:
        newtio.c_cflag |= CS7;
        break;
    case 8:
        newtio.c_cflag |= CS8;
        break;
    }
    /*设置奇偶校验位*/
    switch( nEvent )
    {
    case 'o':
    case 'O': //奇数
        newtio.c_cflag |= PARENB;
        newtio.c_cflag |= PARODD;
        newtio.c_iflag |= (INPCK | ISTRIP);
        break;
    case 'e':
    case 'E': //偶数
        newtio.c_iflag |= (INPCK | ISTRIP);
        newtio.c_cflag |= PARENB;
        newtio.c_cflag &= ~PARODD;
        break;
    case 'n':
    case 'N':  //无奇偶校验位
        newtio.c_cflag &= ~PARENB;
        break;
    default:
        break;
    }
    /*设置波特率*/
    switch( nSpeed )
    {
    case 2400:
        cfsetispeed(&newtio, B2400);
        cfsetospeed(&newtio, B2400);
        break;
    case 4800:
        cfsetispeed(&newtio, B4800);
        cfsetospeed(&newtio, B4800);
        break;
    case 9600:
        cfsetispeed(&newtio, B9600);
        cfsetospeed(&newtio, B9600);
        break;
    case 115200:
        cfsetispeed(&newtio, B115200);
        cfsetospeed(&newtio, B115200);
        break;
    case 460800:
        cfsetispeed(&newtio, B460800);
        cfsetospeed(&newtio, B460800);
        break;
    default:
        cfsetispeed(&newtio, B9600);
        cfsetospeed(&newtio, B9600);
        break;
    }
    /*设置停止位*/
    if( nStop == 1 )
        newtio.c_cflag &=  ~CSTOPB;
    else if ( nStop == 2 )
        newtio.c_cflag |=  CSTOPB;
    /*设置等待时间和最小接收字符*/
    newtio.c_cc[VTIME]  = 0;
    newtio.c_cc[VMIN] = 0;
    /*处理未接收字符*/
    tcflush(fd, TCIFLUSH);
    /*激活新配置*/
    if((tcsetattr(fd, TCSANOW, &newtio)) != 0)
    {
        perror("com set error");
        return -1;
    }
    printf("set done!\n");
    return 0;
}


/**
 *uartSend
 *@param send_buf 要发送的数据 length 发送的数据长度
 */
void uartSend(uint8_t send_buf[], int length)
{
    int w;
    w = write(fd, send_buf, length);
    if(w == -1)
    {
        printf("Send failed!\n");
    }
    else
    {
        printf("Send success!\n");
    }
}
/**
 *uartSend
 *@param send_buf 要接收的数据 length 接收的数据长度
 */
void uartRead(char receive_buf[], int length)
{
    int r;
    r = read(fd, receive_buf, strlen(receive_buf));
    for(int i = 0; i < r; i++)
    {
        printf("%c", receive_buf[i]);
    }

}

int main(int argc , char **argv) {
    fd = openUart(0);              //打开串口
    //uint8_t test1[] = {0xBF,0xa1,0xFB};       //用来测试
    uint8_t test1[3]; 
    uint8_t test2[] = {0xBF,0xa2,0xFB}; 
    uartInit(115200, 8, 'n' , 1) ; //波特率为115200，无奇偶校验,波特率需与下位机相匹配
    while(1)
    {
	for(int i = 1; i > 0; i--) {
        test1[0] = 0xBF;
        test1[1] = 0xa1;
	test1[2] = 0xFB;   
        uartSend(test1, 3);
    	}
        sleep(4);
    	for(int i = 1; i > 0; i--) {
        uartSend(test2, 3);
    	}
 	sleep(4);
	
    }
    
    close(fd);
    return 0;
}

