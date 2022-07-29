#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS

# include<stdio.h>
# include<stdlib.h>
# include<math.h>
# include<time.h>
# include<string.h>

# define TRAIN_NUM  60000  // 训练数据量
# define TEST_NUM  10000  // 测试数据量
# define LEARNING_RATE 0.1  // 初始学习率
# define LAYERS_NUM 5  // 神经网络层数
# define INPUT_SIZE 784  // 输入层神经元个数——>数据为28×28像素的图片（28×28=784）
# define HIDDEN_SIZE1 50  // 隐含层一神经元个数
# define HIDDEN_SIZE2 16  // 隐含层二神经元个数
# define HIDDEN_SIZE3 16  // 隐含层三神经元个数
// # define HIDDEN_SIZE4 256  // 隐含层四神经元个数
# define OUTPUT_SIZE 10  // 输出层神经元个数
# define EPOCHS 50 // 训练次数

double learning_rate = LEARNING_RATE;

// 激活函数
double Activation_Function(double x)
{
	double function_result;
	
	/*
	// Elu函数
	if (x < 0)
	{
		function_result = exp(x) - 1;
	}
	else
	{
		function_result = x;
	}  
	*/
	
	// Sigmoid函数
	function_result = 1 / (1 + exp(-x));
	
	// Tanh函数
	// function_result = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	
	return function_result;
}

// 随机取值函数
double Random_W_B() // 为权重W和偏置B随机赋值
{
	/*
	// 总结： 正态分布的随机数不如直接生成随机数

	// 基于Box-Muller方法生成正态分布的随机数
	static double U1, U2;
	static double PI = 3.14;
	double temp, random_num;

	// U1和U2取值范围为(0,1]——>生成[m, n]的随机数：x = rand() % (n - m + 1) + m
	U1 = (double)(rand() % (1 - 0 + 1));
    U2 = (double)(rand() % (1 - 0 + 1));
	temp = sqrt(2.0 * fabs(log(U1 + 0.5)))* sin(2.0 * PI * U2);
	random_num = (-1.0) + temp * 2;
	*/

	double random_num = (-1) + rand() % (1 - (-1) + 1);
	return random_num;
}

// 定义神经元
typedef struct Neuron
{
	double b; // 偏置
	double a; // 激活值
	double z; // 加权和
	double* w; // 指向权重的指针
	double pd; // 损失函数对当前神经元偏置的偏导（梯度）
} Neuron;

// 定义神经层
typedef struct Layer
{
	int Neuron_Num;
	Neuron* neuron;
} Layer;

// 定义神经网络
typedef struct NeuronNet
{
	int Layer_Num;
	Layer* layer;
}NeuronNet;

// 初始化神经元
void InitialNeuronNet(NeuronNet* neuronnet, int layer_num, int* Array_Of_Neuron_Num_Each_Layer)
// 传参：神经元（结构体）、神经层数目（整型）、每层神经元个数（数组）
{
	// 为神经层开辟动态空间
	neuronnet->Layer_Num = layer_num;
	neuronnet->layer = (Layer*)malloc(sizeof(Layer) * layer_num);

	// 为每层神经元分配动态内存
	for (int i = 0; i < layer_num; i++)
	{
		neuronnet->layer[i].Neuron_Num = Array_Of_Neuron_Num_Each_Layer[i];
		neuronnet->layer[i].neuron = (Neuron*)malloc(sizeof(Neuron) * Array_Of_Neuron_Num_Each_Layer[i]);
	}

	// 初始化神经元参数（从第二层开始）
	for (int i = 1; i < neuronnet->Layer_Num; i++)
	{
		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{

			// 开辟空间（权重个数为上一神经层的神经元个数）
			neuronnet->layer[i].neuron[j].w = (double*)malloc(sizeof(double) * neuronnet->layer[i - 1].Neuron_Num);

			// 初始化偏置
			neuronnet->layer[i].neuron[j].b = Random_W_B();

			// 初始化权重: 对权重向量中的每一个权重赋随机值
			for (int k = 0; k < neuronnet->layer[i - 1].Neuron_Num; k++)
			{
				double weight = Random_W_B();
				neuronnet->layer[i].neuron[j].w[k] = weight;
			}
		}
	}
}

// 正向传播
void Forward_Propagation(NeuronNet* neuronnet, double* inputs)
{
	for (int i = 0; i < neuronnet->layer[0].Neuron_Num; i++)
	{
		// 输入层各神经元激活值初始化
		neuronnet->layer[0].neuron[i].a = inputs[i];
	}

	// 初始化第二层
	for (int i = 1; i < neuronnet->Layer_Num; i++)
	{
		/*
		// 用来归一化的数组
		double max[10000];
		double min[10000];
		*/

		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{
			// 加权和
			double z = 0;
			for (int k = 0; k < neuronnet->layer[i - 1].Neuron_Num; k++)
			{
				double weight = neuronnet->layer[i].neuron[j].w[k];
				z += neuronnet->layer[i - 1].neuron[k].a * weight;
			}
			neuronnet->layer[i].neuron[j].z = z + neuronnet->layer[i].neuron[j].b;
			neuronnet->layer[i].neuron[j].a = Activation_Function(neuronnet->layer[i].neuron[j].z);
		}

		/*
		// 归一化（对Elu函数）：（（X-min）/ （max - min）+ A）
		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{
			max[j] = min[j] = neuronnet->layer[i].neuron[0].a;
			max[j] = max[j] > neuronnet->layer[i].neuron[j].a ? max[j] : neuronnet->layer[i].neuron[j].a;
			min[j] = min[j] < neuronnet->layer[i].neuron[j].a ? min[j] : neuronnet->layer[i].neuron[j].a;
		}

		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{
			neuronnet->layer[i].neuron[j].a = ((neuronnet->layer[i].neuron[j].a - min[j]) / (max[j] - min[j] + 1.5));
		}

		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{
			max[j] = 0;
			min[j] = 0;
		}
		*/

	}
}

// 反向传播
void Back_Propagation(NeuronNet* neuronnet, double* targets)
{
	// Neuron_Num_Of_The_Last_Layer为最后一层神经元个数
	int Neuron_Num_Of_The_Last_Layer = neuronnet->layer[neuronnet->Layer_Num - 1].Neuron_Num;
	// i为最后一层神经元的下标
	for (int i = 0; i < Neuron_Num_Of_The_Last_Layer; i++)
	{
		// A为最后一层神经元的激活值
		double A = neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].a;
		// 求最后一层每个神经元的梯度（损失函数对偏置求偏导）
	    
		// Sigmoid函数
		neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = A * (1 - A) * (targets[i] - A) / OUTPUT_SIZE;
		
		// Tanh函数
		// neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = (1 - pow(A, 2)) * (targets[i] - A) / OUTPUT_SIZE;  
		
		/*
		// Elu函数
		if(A > 0)
		{
		    neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = (targets[i] - A) / OUTPUT_SIZE;
		}
		else
		{
			neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = (double)((exp(A) - 1) * (targets[i] - A) / OUTPUT_SIZE);
		}
		*/
		
		// 更新权值
		neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].b += learning_rate * neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd;
	}

	// 开始反向传播
	for (int i = neuronnet->Layer_Num - 1; i > 0; i--)
	{
		// 从倒数第二层神经元开始，j为前一层的神经元数
		for (int j = 0; j < neuronnet->layer[i - 1].Neuron_Num; j++)
		{
			// 对前一层激活值的偏导数（初始化为0）
			double PD_SUM = 0;
			// k为当前层神经元数
			for (int k = 0; k < neuronnet->layer[i].Neuron_Num; k++)
			{
				// 对前一层激活值的偏导数
				PD_SUM += neuronnet->layer[i].neuron[k].w[j] * neuronnet->layer[i].neuron[k].pd;
				/*
				   更新当前层权重的偏导数（两个公式如下）：
				   1、对当前层权重的偏导数 = 对偏置的偏导数 * 前一层的激活值
				   2、更新的权重 = 学习率 * Σ（对当前层权重的偏导数）
				*/
				neuronnet->layer[i].neuron[k].w[j] += learning_rate * neuronnet->layer[i].neuron[k].pd * neuronnet->layer[i - 1].neuron[j].a;
			}
			// 前一层神经元的激活值
			double activation = neuronnet->layer[i - 1].neuron[j].a;
			// 前一层神经元对偏置的导数的更新
			
			// Sigmoid函数
			neuronnet->layer[i - 1].neuron[j].pd = activation * (1 - activation) * PD_SUM; 
			
			// Tanh函数
			// neuronnet->layer[i - 1].neuron[j].pd = (1 - pow(activation, 2)) * PD_SUM;  

			/*
			// Elu函数
			if (activation > 0)
			{
				neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = PD_SUM;
			}
			else
			{
				neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = (double)((exp(activation) - 1) * PD_SUM);
			}
			*/

			neuronnet->layer[i - 1].neuron[j].b += learning_rate * neuronnet->layer[i - 1].neuron[j].pd;
		}
	}
}

// 存储模型数据(二进制）
void Save_Database(NeuronNet* neuronnet, FILE* fpModel)
{
	fopen_s(&fpModel, "./model.dat", "w+b");

	/* 
	    记录每一神经层的每一神经元的权重及偏置
	    输入层没有权重，故层数从1开始而非从0开始
	*/
	for (int i = 1; i < neuronnet->Layer_Num; i++)
	{
		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{
			for (int k = 0; k < neuronnet->layer[i - 1].Neuron_Num; k++)
			{
				fwrite(&(neuronnet->layer[i].neuron[j].w[k]), sizeof(double), 1, fpModel);
			}
			fwrite(&(neuronnet->layer[i].neuron[j].b), sizeof(double), 1, fpModel);
		}
	}
	fclose(fpModel);
}

// 读取模型数据
void Read_Database(NeuronNet* neuronnet, FILE* fpModel)
{
	fopen_s(&fpModel, "./model.dat", "rb");

	/*
		读取每一神经层的每一神经元的权重及偏置
		输入层没有权重，故层数从1开始而非从0开始
	*/
	for (int i = 1; i < neuronnet->Layer_Num; i++)
	{
		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{
			for (int k = 0; k < neuronnet->layer[i - 1].Neuron_Num; k++)
			{
				fread(&(neuronnet->layer[i].neuron[j].w[k]), sizeof(double), 1, fpModel);
			}
			fread(&(neuronnet->layer[i].neuron[j].b), sizeof(double), 1, fpModel);
		}
	}
	fclose(fpModel);
}

// 读取数据集中灰度值（储存在数组里面）
void initImgArray(FILE* fpImg, double** BufferArray, int Image_Num)
{
	unsigned char* tmpImg = (unsigned char*)malloc(sizeof(unsigned char) * INPUT_SIZE);

	for (int i = 0; i < Image_Num; i++)
	{
		fread(tmpImg, sizeof(unsigned char), INPUT_SIZE, fpImg);

		// 对于第i张图片的第j格像素，都读取其灰度值并存储到指向数据的指针tmpImg中
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			BufferArray[i][j] = tmpImg[j] / 255.0;
		}
	}
	free(tmpImg);
}

// 读取标签
void initLabelArray(FILE* fpLabel, int* BufferArray, int Label_Num)
{
	unsigned char* tmpLabel = (unsigned char*)malloc(sizeof(unsigned char) * Label_Num);

	fread(tmpLabel, sizeof(unsigned char), Label_Num, fpLabel);

	for (int i = 0; i < Label_Num; i++)
	{
		BufferArray[i] = tmpLabel[i];
	}

	free(tmpLabel);
}

// 正确率
double Accuracy_Rate(NeuronNet* neuronnet, double** Test_Buffer_Array_Image, int* Test_Buffer_Array_Label)
{
	int Right = 0;
	for (int i = 0; i < TEST_NUM; i++)
	{
		double Test_Input[INPUT_SIZE];
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			Test_Input[j] = Test_Buffer_Array_Image[i][j];
		}

		Forward_Propagation(neuronnet, Test_Input);

		// 通过输出层激活值大小判断输入的图片对应哪一个数字
		double max = neuronnet->layer[neuronnet->Layer_Num - 1].neuron[0].a;
		int GuessNum = 0;
		for (int j = 0; j < OUTPUT_SIZE; j++)
		{
			if (neuronnet->layer[neuronnet->Layer_Num - 1].neuron[j].a > max)
			{
				max = neuronnet->layer[neuronnet->Layer_Num - 1].neuron[0].a;
				GuessNum = j;
			}
		}

		if (GuessNum == Test_Buffer_Array_Label[i])
		{
			Right++;
		}
	}

	// 返回正确率（double类型）
	return (double)Right / TEST_NUM;
}

// 主函数
int main()
{
	// 读取数据集文件
	int Magic_Num, Pic_Num, Pixel_Row, Pixel_Column;
	int LMagic_Num, Label_Num;

	FILE* fpImg;
    fopen_s(&fpImg, "TrainImage.idx3-ubyte", "rb");
	if (fpImg == NULL)
	{
		printf("训练集数据打开失败!\n");
	}
	else
	{
		fread(&Magic_Num, sizeof(int), 1, fpImg);
		fread(&Pic_Num, sizeof(int), 1, fpImg);
		fread(&Pixel_Row, sizeof(int), 1, fpImg);
		fread(&Pixel_Column, sizeof(int), 1, fpImg);
	}

	FILE* fpLabel;
	fopen_s(&fpLabel, "TrainLabel.idx1-ubyte", "rb");
	if (fpLabel == NULL)
	{
		printf("训练集标签打开失败!\n");
	}
	else
	{
		fread(&LMagic_Num, sizeof(int), 1, fpLabel);
		fread(&Label_Num, sizeof(int), 1, fpLabel);
	}

	// 读取测试集文件
	int Test_Magic_Num, Test_Pic_Num, Test_Pixel_Row, Test_Pixel_Column;
	int Test_LMagic_Num, Test_Label_Num;

	FILE* test_fpImg;
	fopen_s(&test_fpImg, "TestImage.idx3-ubyte", "rb");
	if (test_fpImg == NULL)
	{
		printf("测试集数据打开失败!\n");
	}
	else
	{
		fread(&Test_Magic_Num, sizeof(int), 1, test_fpImg);
		fread(&Test_Pic_Num, sizeof(int), 1, test_fpImg);
		fread(&Test_Pixel_Row, sizeof(int), 1, test_fpImg);
		fread(&Test_Pixel_Column, sizeof(int), 1, test_fpImg);
	}

	FILE* test_fpLabel;
	fopen_s(&test_fpLabel, "TestLabel.idx1-ubyte", "rb");
	if (test_fpLabel == NULL)
	{
		printf("测试集标签打开失败!\n");
	}
	else
	{
		fread(&Test_LMagic_Num, sizeof(int), 1, test_fpLabel);
		fread(&Test_Label_Num, sizeof(int), 1, test_fpLabel);
	}

	// 存储训练集中灰度值
	double** Buffer_Array_Image = (double**)malloc(sizeof(double*) * TRAIN_NUM);
	for (int i = 0; i < TRAIN_NUM; i++)
	{
		Buffer_Array_Image[i] = (double*)malloc(sizeof(double) * INPUT_SIZE);
	}
	initImgArray(fpImg, Buffer_Array_Image, TRAIN_NUM);

	// 存储训练集中标签
	int* Buffer_Array_Label = (int*)malloc(sizeof(int) * TRAIN_NUM);
	initLabelArray(fpLabel, Buffer_Array_Label, TRAIN_NUM);

	// 存储测试集中灰度值
	double** Test_Buffer_Array_Image = (double**)malloc(sizeof(double*) * TEST_NUM);
	for (int i = 0; i < TEST_NUM; i++)
	{
		Test_Buffer_Array_Image[i] = (double*)malloc(sizeof(double) * INPUT_SIZE);
	}
	initImgArray(test_fpImg, Test_Buffer_Array_Image, TEST_NUM);

	// 存储测试集中标签
	int* Test_Buffer_Array_Label = (int*)malloc(sizeof(int) * TEST_NUM);
	initLabelArray(test_fpLabel, Test_Buffer_Array_Label, TEST_NUM);

	// 分配内存空间
	NeuronNet* neuron_net = (NeuronNet*)malloc(sizeof(NeuronNet));

	// 储存每层神经元个数的数组
	int Array_Of_Neuron_Num_Each_Layer[LAYERS_NUM] = { INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE};

	// 初始化随机种子
	srand((unsigned int)time(NULL));

	// 初始化神经网络内存空间
	InitialNeuronNet(neuron_net, LAYERS_NUM, Array_Of_Neuron_Num_Each_Layer);

	// 初始化用来保存数据的文件指针
	FILE* fpModel = NULL;

	// 设置计数器 
	int sw = 0;

	// 开始训练（for循环——>每一轮训练中每一张样本图片的每一格像素点）
	for (int i = 0; i < EPOCHS; i++)
	{
		for (int j = 0; j < TRAIN_NUM; j++)
		{
			double inputs[INPUT_SIZE];
			for (int k = 0; k < INPUT_SIZE; k++)
			{
				inputs[k] = Buffer_Array_Image[j][k];
			}

			double targets[OUTPUT_SIZE];
			memset(targets, 0, sizeof(double)* OUTPUT_SIZE);
			// memset(要填充的内存块, 要被设置的值, 要被设置该值的字符数)
			// memset返回一个指向要填充的内存块的指针
			targets[Buffer_Array_Label[j]] = 1.0;

			// 前向传播
			Forward_Propagation(neuron_net, inputs);

			// 反向传播
			Back_Propagation(neuron_net, targets);

			// 每训练10000个数据（即10000张图片），返回一次结果
			if ((j + 1) % 10000 == 0)
			{
				printf("Epoch: %d , Index of Image: %d , Label of Image: %d\n", i + 1, j + 1, Buffer_Array_Label[j]);
				for (int k = 0; k < neuron_net->layer[4].Neuron_Num; k++)
				{
					// 打印激活值
					printf("%d:  %.20lf\n", k, neuron_net->layer[4].neuron[k].a);
				}
				printf("\n");
			}
		}

		// 保存数据
		Save_Database(neuron_net, fpModel);

		// 打印正确率
		printf("Accuracy Rate:  %lf%%\n", 100 * Accuracy_Rate(neuron_net, Test_Buffer_Array_Image, Test_Buffer_Array_Label));

		// 学习率动态衰减（每训练10轮衰减一次）
		if (sw != 0 && sw % 10 == 0)
		{
			// learning_rate *= 0.1;
			learning_rate *= 1; // 保持学习率不变（控制变量做实验时使用）
		}
		
		// 计数器自增
		sw++;
	}

	// 释放空间
	free(Buffer_Array_Image);
	free(Buffer_Array_Label);
	free(Test_Buffer_Array_Image);
	free(Test_Buffer_Array_Label);

	fclose(fpImg);
	fclose(fpLabel);

	getchar();
	return 0;
}
