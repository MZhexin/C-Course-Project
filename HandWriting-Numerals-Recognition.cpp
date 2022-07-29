#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS

# include<stdio.h>
# include<stdlib.h>
# include<math.h>
# include<time.h>
# include<string.h>

# define TRAIN_NUM  60000  // ѵ��������
# define TEST_NUM  10000  // ����������
# define LEARNING_RATE 0.1  // ��ʼѧϰ��
# define LAYERS_NUM 5  // ���������
# define INPUT_SIZE 784  // �������Ԫ��������>����Ϊ28��28���ص�ͼƬ��28��28=784��
# define HIDDEN_SIZE1 50  // ������һ��Ԫ����
# define HIDDEN_SIZE2 16  // ���������Ԫ����
# define HIDDEN_SIZE3 16  // ����������Ԫ����
// # define HIDDEN_SIZE4 256  // ����������Ԫ����
# define OUTPUT_SIZE 10  // �������Ԫ����
# define EPOCHS 50 // ѵ������

double learning_rate = LEARNING_RATE;

// �����
double Activation_Function(double x)
{
	double function_result;
	
	/*
	// Elu����
	if (x < 0)
	{
		function_result = exp(x) - 1;
	}
	else
	{
		function_result = x;
	}  
	*/
	
	// Sigmoid����
	function_result = 1 / (1 + exp(-x));
	
	// Tanh����
	// function_result = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	
	return function_result;
}

// ���ȡֵ����
double Random_W_B() // ΪȨ��W��ƫ��B�����ֵ
{
	/*
	// �ܽ᣺ ��̬�ֲ������������ֱ�����������

	// ����Box-Muller����������̬�ֲ��������
	static double U1, U2;
	static double PI = 3.14;
	double temp, random_num;

	// U1��U2ȡֵ��ΧΪ(0,1]����>����[m, n]���������x = rand() % (n - m + 1) + m
	U1 = (double)(rand() % (1 - 0 + 1));
    U2 = (double)(rand() % (1 - 0 + 1));
	temp = sqrt(2.0 * fabs(log(U1 + 0.5)))* sin(2.0 * PI * U2);
	random_num = (-1.0) + temp * 2;
	*/

	double random_num = (-1) + rand() % (1 - (-1) + 1);
	return random_num;
}

// ������Ԫ
typedef struct Neuron
{
	double b; // ƫ��
	double a; // ����ֵ
	double z; // ��Ȩ��
	double* w; // ָ��Ȩ�ص�ָ��
	double pd; // ��ʧ�����Ե�ǰ��Ԫƫ�õ�ƫ�����ݶȣ�
} Neuron;

// �����񾭲�
typedef struct Layer
{
	int Neuron_Num;
	Neuron* neuron;
} Layer;

// ����������
typedef struct NeuronNet
{
	int Layer_Num;
	Layer* layer;
}NeuronNet;

// ��ʼ����Ԫ
void InitialNeuronNet(NeuronNet* neuronnet, int layer_num, int* Array_Of_Neuron_Num_Each_Layer)
// ���Σ���Ԫ���ṹ�壩���񾭲���Ŀ�����ͣ���ÿ����Ԫ���������飩
{
	// Ϊ�񾭲㿪�ٶ�̬�ռ�
	neuronnet->Layer_Num = layer_num;
	neuronnet->layer = (Layer*)malloc(sizeof(Layer) * layer_num);

	// Ϊÿ����Ԫ���䶯̬�ڴ�
	for (int i = 0; i < layer_num; i++)
	{
		neuronnet->layer[i].Neuron_Num = Array_Of_Neuron_Num_Each_Layer[i];
		neuronnet->layer[i].neuron = (Neuron*)malloc(sizeof(Neuron) * Array_Of_Neuron_Num_Each_Layer[i]);
	}

	// ��ʼ����Ԫ�������ӵڶ��㿪ʼ��
	for (int i = 1; i < neuronnet->Layer_Num; i++)
	{
		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{

			// ���ٿռ䣨Ȩ�ظ���Ϊ��һ�񾭲����Ԫ������
			neuronnet->layer[i].neuron[j].w = (double*)malloc(sizeof(double) * neuronnet->layer[i - 1].Neuron_Num);

			// ��ʼ��ƫ��
			neuronnet->layer[i].neuron[j].b = Random_W_B();

			// ��ʼ��Ȩ��: ��Ȩ�������е�ÿһ��Ȩ�ظ����ֵ
			for (int k = 0; k < neuronnet->layer[i - 1].Neuron_Num; k++)
			{
				double weight = Random_W_B();
				neuronnet->layer[i].neuron[j].w[k] = weight;
			}
		}
	}
}

// ���򴫲�
void Forward_Propagation(NeuronNet* neuronnet, double* inputs)
{
	for (int i = 0; i < neuronnet->layer[0].Neuron_Num; i++)
	{
		// ��������Ԫ����ֵ��ʼ��
		neuronnet->layer[0].neuron[i].a = inputs[i];
	}

	// ��ʼ���ڶ���
	for (int i = 1; i < neuronnet->Layer_Num; i++)
	{
		/*
		// ������һ��������
		double max[10000];
		double min[10000];
		*/

		for (int j = 0; j < neuronnet->layer[i].Neuron_Num; j++)
		{
			// ��Ȩ��
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
		// ��һ������Elu������������X-min��/ ��max - min��+ A��
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

// ���򴫲�
void Back_Propagation(NeuronNet* neuronnet, double* targets)
{
	// Neuron_Num_Of_The_Last_LayerΪ���һ����Ԫ����
	int Neuron_Num_Of_The_Last_Layer = neuronnet->layer[neuronnet->Layer_Num - 1].Neuron_Num;
	// iΪ���һ����Ԫ���±�
	for (int i = 0; i < Neuron_Num_Of_The_Last_Layer; i++)
	{
		// AΪ���һ����Ԫ�ļ���ֵ
		double A = neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].a;
		// �����һ��ÿ����Ԫ���ݶȣ���ʧ������ƫ����ƫ����
	    
		// Sigmoid����
		neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = A * (1 - A) * (targets[i] - A) / OUTPUT_SIZE;
		
		// Tanh����
		// neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = (1 - pow(A, 2)) * (targets[i] - A) / OUTPUT_SIZE;  
		
		/*
		// Elu����
		if(A > 0)
		{
		    neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = (targets[i] - A) / OUTPUT_SIZE;
		}
		else
		{
			neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd = (double)((exp(A) - 1) * (targets[i] - A) / OUTPUT_SIZE);
		}
		*/
		
		// ����Ȩֵ
		neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].b += learning_rate * neuronnet->layer[neuronnet->Layer_Num - 1].neuron[i].pd;
	}

	// ��ʼ���򴫲�
	for (int i = neuronnet->Layer_Num - 1; i > 0; i--)
	{
		// �ӵ����ڶ�����Ԫ��ʼ��jΪǰһ�����Ԫ��
		for (int j = 0; j < neuronnet->layer[i - 1].Neuron_Num; j++)
		{
			// ��ǰһ�㼤��ֵ��ƫ��������ʼ��Ϊ0��
			double PD_SUM = 0;
			// kΪ��ǰ����Ԫ��
			for (int k = 0; k < neuronnet->layer[i].Neuron_Num; k++)
			{
				// ��ǰһ�㼤��ֵ��ƫ����
				PD_SUM += neuronnet->layer[i].neuron[k].w[j] * neuronnet->layer[i].neuron[k].pd;
				/*
				   ���µ�ǰ��Ȩ�ص�ƫ������������ʽ���£���
				   1���Ե�ǰ��Ȩ�ص�ƫ���� = ��ƫ�õ�ƫ���� * ǰһ��ļ���ֵ
				   2�����µ�Ȩ�� = ѧϰ�� * �����Ե�ǰ��Ȩ�ص�ƫ������
				*/
				neuronnet->layer[i].neuron[k].w[j] += learning_rate * neuronnet->layer[i].neuron[k].pd * neuronnet->layer[i - 1].neuron[j].a;
			}
			// ǰһ����Ԫ�ļ���ֵ
			double activation = neuronnet->layer[i - 1].neuron[j].a;
			// ǰһ����Ԫ��ƫ�õĵ����ĸ���
			
			// Sigmoid����
			neuronnet->layer[i - 1].neuron[j].pd = activation * (1 - activation) * PD_SUM; 
			
			// Tanh����
			// neuronnet->layer[i - 1].neuron[j].pd = (1 - pow(activation, 2)) * PD_SUM;  

			/*
			// Elu����
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

// �洢ģ������(�����ƣ�
void Save_Database(NeuronNet* neuronnet, FILE* fpModel)
{
	fopen_s(&fpModel, "./model.dat", "w+b");

	/* 
	    ��¼ÿһ�񾭲��ÿһ��Ԫ��Ȩ�ؼ�ƫ��
	    �����û��Ȩ�أ��ʲ�����1��ʼ���Ǵ�0��ʼ
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

// ��ȡģ������
void Read_Database(NeuronNet* neuronnet, FILE* fpModel)
{
	fopen_s(&fpModel, "./model.dat", "rb");

	/*
		��ȡÿһ�񾭲��ÿһ��Ԫ��Ȩ�ؼ�ƫ��
		�����û��Ȩ�أ��ʲ�����1��ʼ���Ǵ�0��ʼ
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

// ��ȡ���ݼ��лҶ�ֵ���������������棩
void initImgArray(FILE* fpImg, double** BufferArray, int Image_Num)
{
	unsigned char* tmpImg = (unsigned char*)malloc(sizeof(unsigned char) * INPUT_SIZE);

	for (int i = 0; i < Image_Num; i++)
	{
		fread(tmpImg, sizeof(unsigned char), INPUT_SIZE, fpImg);

		// ���ڵ�i��ͼƬ�ĵ�j�����أ�����ȡ��Ҷ�ֵ���洢��ָ�����ݵ�ָ��tmpImg��
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			BufferArray[i][j] = tmpImg[j] / 255.0;
		}
	}
	free(tmpImg);
}

// ��ȡ��ǩ
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

// ��ȷ��
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

		// ͨ������㼤��ֵ��С�ж������ͼƬ��Ӧ��һ������
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

	// ������ȷ�ʣ�double���ͣ�
	return (double)Right / TEST_NUM;
}

// ������
int main()
{
	// ��ȡ���ݼ��ļ�
	int Magic_Num, Pic_Num, Pixel_Row, Pixel_Column;
	int LMagic_Num, Label_Num;

	FILE* fpImg;
    fopen_s(&fpImg, "TrainImage.idx3-ubyte", "rb");
	if (fpImg == NULL)
	{
		printf("ѵ�������ݴ�ʧ��!\n");
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
		printf("ѵ������ǩ��ʧ��!\n");
	}
	else
	{
		fread(&LMagic_Num, sizeof(int), 1, fpLabel);
		fread(&Label_Num, sizeof(int), 1, fpLabel);
	}

	// ��ȡ���Լ��ļ�
	int Test_Magic_Num, Test_Pic_Num, Test_Pixel_Row, Test_Pixel_Column;
	int Test_LMagic_Num, Test_Label_Num;

	FILE* test_fpImg;
	fopen_s(&test_fpImg, "TestImage.idx3-ubyte", "rb");
	if (test_fpImg == NULL)
	{
		printf("���Լ����ݴ�ʧ��!\n");
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
		printf("���Լ���ǩ��ʧ��!\n");
	}
	else
	{
		fread(&Test_LMagic_Num, sizeof(int), 1, test_fpLabel);
		fread(&Test_Label_Num, sizeof(int), 1, test_fpLabel);
	}

	// �洢ѵ�����лҶ�ֵ
	double** Buffer_Array_Image = (double**)malloc(sizeof(double*) * TRAIN_NUM);
	for (int i = 0; i < TRAIN_NUM; i++)
	{
		Buffer_Array_Image[i] = (double*)malloc(sizeof(double) * INPUT_SIZE);
	}
	initImgArray(fpImg, Buffer_Array_Image, TRAIN_NUM);

	// �洢ѵ�����б�ǩ
	int* Buffer_Array_Label = (int*)malloc(sizeof(int) * TRAIN_NUM);
	initLabelArray(fpLabel, Buffer_Array_Label, TRAIN_NUM);

	// �洢���Լ��лҶ�ֵ
	double** Test_Buffer_Array_Image = (double**)malloc(sizeof(double*) * TEST_NUM);
	for (int i = 0; i < TEST_NUM; i++)
	{
		Test_Buffer_Array_Image[i] = (double*)malloc(sizeof(double) * INPUT_SIZE);
	}
	initImgArray(test_fpImg, Test_Buffer_Array_Image, TEST_NUM);

	// �洢���Լ��б�ǩ
	int* Test_Buffer_Array_Label = (int*)malloc(sizeof(int) * TEST_NUM);
	initLabelArray(test_fpLabel, Test_Buffer_Array_Label, TEST_NUM);

	// �����ڴ�ռ�
	NeuronNet* neuron_net = (NeuronNet*)malloc(sizeof(NeuronNet));

	// ����ÿ����Ԫ����������
	int Array_Of_Neuron_Num_Each_Layer[LAYERS_NUM] = { INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE};

	// ��ʼ���������
	srand((unsigned int)time(NULL));

	// ��ʼ���������ڴ�ռ�
	InitialNeuronNet(neuron_net, LAYERS_NUM, Array_Of_Neuron_Num_Each_Layer);

	// ��ʼ�������������ݵ��ļ�ָ��
	FILE* fpModel = NULL;

	// ���ü����� 
	int sw = 0;

	// ��ʼѵ����forѭ������>ÿһ��ѵ����ÿһ������ͼƬ��ÿһ�����ص㣩
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
			// memset(Ҫ�����ڴ��, Ҫ�����õ�ֵ, Ҫ�����ø�ֵ���ַ���)
			// memset����һ��ָ��Ҫ�����ڴ���ָ��
			targets[Buffer_Array_Label[j]] = 1.0;

			// ǰ�򴫲�
			Forward_Propagation(neuron_net, inputs);

			// ���򴫲�
			Back_Propagation(neuron_net, targets);

			// ÿѵ��10000�����ݣ���10000��ͼƬ��������һ�ν��
			if ((j + 1) % 10000 == 0)
			{
				printf("Epoch: %d , Index of Image: %d , Label of Image: %d\n", i + 1, j + 1, Buffer_Array_Label[j]);
				for (int k = 0; k < neuron_net->layer[4].Neuron_Num; k++)
				{
					// ��ӡ����ֵ
					printf("%d:  %.20lf\n", k, neuron_net->layer[4].neuron[k].a);
				}
				printf("\n");
			}
		}

		// ��������
		Save_Database(neuron_net, fpModel);

		// ��ӡ��ȷ��
		printf("Accuracy Rate:  %lf%%\n", 100 * Accuracy_Rate(neuron_net, Test_Buffer_Array_Image, Test_Buffer_Array_Label));

		// ѧϰ�ʶ�̬˥����ÿѵ��10��˥��һ�Σ�
		if (sw != 0 && sw % 10 == 0)
		{
			// learning_rate *= 0.1;
			learning_rate *= 1; // ����ѧϰ�ʲ��䣨���Ʊ�����ʵ��ʱʹ�ã�
		}
		
		// ����������
		sw++;
	}

	// �ͷſռ�
	free(Buffer_Array_Image);
	free(Buffer_Array_Label);
	free(Test_Buffer_Array_Image);
	free(Test_Buffer_Array_Label);

	fclose(fpImg);
	fclose(fpLabel);

	getchar();
	return 0;
}
