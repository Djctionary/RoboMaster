# yolov5-test-1.1

***2023/10/29***

## 成果

成功在实验室orin nx上部署测试代码，成功编译，成功生成engine文件，成功推理出目标框中心值并存入txt文本。

## 问题

1. 未测试输出坐标值是否准确
2. 存入txt文本的数据没有标识和解释

## 改进目标

改进**postprocess.cpp**代码在输出的坐标值处画圆，并把框也画出来，测试输出是否准确。

改进输出到文本的数据格式，便于他人阅读和理解。

## 注意事项

**config.h**中需要修改**kNumClass**的值为需要识别物体类别的数量

> // Detection model and Segmentation model' number of classes
>
> constexpr static int kNumClass = 1;