#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

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
    cv::VideoCapture capture(0); // 打开USB摄像机，参数0表示使用�?一�?摄像�?

    if (!capture.isOpened()) {
        std::cerr << "Failed to open camera!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_NORMAL);

  // batch predict
  while (true) {

    static long long int name=1;
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    capture.read(frame); // 读取摄像机捕获的�?

        if (frame.empty()) {
            std::cerr << "No frame captured!" << std::endl;
            break;
        }

        cv::imshow("Camera", frame); // 显示图像
	
        // 按下'q'�?时退出循�?
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

    // Draw bounding boxes
    static int number=1;
    double angle = draw_bbox(img_batch, res_batch, number);//Éú³ÉÖÐÐÄµã×ø±êŽæÈë±ŸµØÎÄ±ŸÖÐ
    number++;

    int flag = 0;//left 0 right 1
    if(angle<0) flag = 1;
    std::string save_path = "../output_images";
    cv::imwrite(save_path + "/" + std::to_string(name) + ".png", img_batch[0]);
      
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::imshow("Output", img_batch[0]); // 显示图像
    name++;
  }

  // Release stream and buffers
  capture.release(); // 释放摄像�?
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

