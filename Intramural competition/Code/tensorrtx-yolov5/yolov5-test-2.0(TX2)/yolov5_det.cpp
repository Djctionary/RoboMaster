#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <thread>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
int fd;
uint8_t test[3];
int flag,right,first=0;
static int number=1;
static double pid;

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw) {
  if (std::string(argv[1]) == "-d" && argc == 3) {
    engine = std::string(argv[2]);
  } else {
    return false;
  }
  return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}



void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

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
void SPC(double pid)
{
    //pid is + means left
    //pid is - means right

    //fd = openUart(0);              //打开串口	
    //uint8_t test[3]; 
    if(-1<=pid&&pid<=1)
    {
        test[0] = 0xBF;
        test[1] = 0xa1;
	test[2] = 0xFB;  
    }
    else if(pid>1&&pid<4)
    {
        test[0] = 0xBF;
        test[1] = 0xa2;
	test[2] = 0xFB;   
    }
    else if(pid<-1&&pid>-4)
    {
        test[0] = 0xBF;
        test[1] = 0xa3;
	test[2] = 0xFB;   
    }
    else if(pid>=4&&pid<=7)
    {
        test[0] = 0xBF;
        test[1] = 0xa4;
	test[2] = 0xFB;   
    }
    else if(pid<=-4&&pid>=-7)
    {
        test[0] = 0xBF;
        test[1] = 0xa5;
	test[2] = 0xFB;   
    }
    else if(pid>7&&pid<=100)
    {
        test[0] = 0xBF;
        test[1] = 0xa6;
	test[2] = 0xFB;   
    }
    else if(pid<-7&&pid>=-100)
    {
        test[0] = 0xBF;
        test[1] = 0xa7;
	test[2] = 0xFB;   
    }
    else if(pid==999)
    {
        test[0] = 0xBF;
        test[1] = 0xa8;
	test[2] = 0xFB;   
    }
    else if(pid==666)
    {
        test[0] = 0xBF;
        test[1] = 0xa9;
	test[2] = 0xFB;   
    }


    uartSend(test, 3);


    std::ofstream file("test.txt", std::ios::app);
    if(test[1]==0xa0)
    {
	file<<test[0]<<" "<<0xa0<<" "<<test[2]<<std::endl;
    }
    else if(test[1]==0xa1)
    {
	file<<test[0]<<" "<<0xa1<<" "<<test[2]<<std::endl;
    }
    else if(test[1]==0xa2)
    {
	file<<test[0]<<" "<<0xa2<<" "<<test[2]<<std::endl;
    }
    else if(test[1]==0xa3)
    {
	file<<test[0]<<" "<<0xa3<<" "<<test[2]<<std::endl;
    }
    file.close();

   // close(fd);

}

/*************************************************************************************************/

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);

  std::string wts_name = "";
  std::string engine_name = "";
  bool is_p6 = false;
  float gd = 0.0f, gw = 0.0f;

  if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov5_det -d [.engine]  // deserialize plan file and run inference" << std::endl;
    return -1;
  }


  // Deserialize the engine from file
  IRuntime* runtime = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  deserialize_engine(engine_name, &runtime, &engine, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Init CUDA preprocessing
  cuda_preprocess_init(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float* gpu_buffers[2];
  float* cpu_output_buffer = nullptr;
  prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

  // Read images from camara
    cv::VideoCapture capture(0); // æåŒUSBæåæºïŒåæ°0è¡šç€ºäœ¿çšç¬?äžäž?æåå€?

    if (!capture.isOpened()) {
        std::cerr << "Failed to open camera!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_NORMAL);


    //serialport

    fd = openUart(0);              //打开串口	
    uartInit(115200, 8, 'n' , 1) ; //波特率为115200，无奇偶校验,波特率需与下位机相匹配
    
  // batch predict
  while (true) {

    static long long int name=1;
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    capture.read(frame); // è¯»åæåæºæè·çåž?

        if (frame.empty()) {
            std::cerr << "No frame captured!" << std::endl;
            break;
        }

        cv::imshow("Camera", frame); // æŸç€ºåŸå
	
        // æäž'q'é?æ¶éåºåŸªç?
        if (cv::waitKey(1) == 'q') {
            break;
        }
        
    img_batch.push_back(frame);
    img_name_batch.push_back(std::to_string(name));
    

    // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);


    for (size_t i = 0; i < img_batch.size(); i++) {
    auto& res = res_batch[i];
    int exist = 1;
    for (size_t j = 0; j < res.size(); j++) {
    if((int)res[j].class_id==1) flag=0;
    else if((int)res[j].class_id==2) flag++;
    if((int)res[j].class_id==3&&exist==1) right++;
    else	right=0;

    std::ofstream file("class_id.txt", std::ios::app);
    file<<(int)res[j].class_id<<std::endl;
    file.close();
	}
        exist=0;
    }

    
    if(flag>=10)
    {

	if(right>=10)
	{
	    SPC(999);
	}

	// Draw bounding boxes
	else
	{
    		pid = draw_bbox(img_batch, res_batch, number);//ÃÃºÂ³ÃÃÃÃÃÂµÃ£ÃÃžÂ±ÃªÅœÃŠÃÃ«Â±ÅžÂµÃÃÃÂ±ÅžÃÃ
    		number++;
    
    		std::ofstream file("pid.txt", std::ios::app);
    		file<<pid<<std::endl;
    		file.close();


    		SPC(pid);
    //std::string save_path = "../output_images";
    //cv::imwrite(save_path + "/" + std::to_string(name) + ".png", img_batch[0]);
    		name++;
	}

    }
    else if(flag<10) SPC(666);

    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::imshow("Output", img_batch[0]); // æŸç€ºåŸå
       
}
  close(fd);
  // Release stream and buffers
  capture.release(); // éæŸæåæ?
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_output_buffer;
  cuda_preprocess_destroy();
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  // Print histogram of the output distribution
  // std::cout << "\nOutput:\n\n";
  // for (unsigned int i = 0; i < kOutputSize; i++) {
  //   std::cout << prob[i] << ", ";
  //   if (i % 10 == 0) std::cout << std::endl;
  // }
  // std::cout << std::endl;

  return 0;
}

