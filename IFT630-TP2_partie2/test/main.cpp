/*
 * Source originale : https://github.com/Dakkers/OpenCL-examples/blob/master/example00/main.cpp
 * Modifi� par : Daniel-Junior Dub�
*/

#pragma warning(disable : 4996)
#pragma comment(lib, "OpenCL.lib")
#include <iostream>
#include <CL/cl.hpp>
#include <string>
#include <fstream>
using namespace std;

int main(int argc, char* args[]) {

	// Obtention des plateformes.
	std::vector<cl::Platform> available_platforms;
	cl::Platform::get(&available_platforms);
	if (available_platforms.size() == 0) {
		std::cout << "No platforms found. Check OpenCL installation!" << std::endl;
		exit(1);
	}

	// Affichage des informations OpenCL (plateformes, versions et p�riph�riques).
	for (auto& platform : available_platforms) {
		std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
		std::string version = platform.getInfo<CL_PLATFORM_VERSION>();
		std::cout << "Found platform '" << platform_name << "' with version : '" << version << "'" << std::endl;

		std::vector<cl::Device> available_devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &available_devices);
		for (auto& device : available_devices) {
			std::cout << "	With device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		}
	}

	// Obtention de la platforme d'ex�cution du programme (premier disponible).
	cl::Platform default_platform = available_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	// S�lection du p�riph�rique de calcul (premier disponible).
	std::vector<cl::Device> available_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &available_devices);
	cl::Device default_device = available_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

	// D�claration du contexte de calcul.
	cl::Context context({
		default_device
		});

	// Code source du programme GPU.
	cl::Program::Sources sources;

	// Cha�ne de caract�res contenant le programme pour appliquer `C = A + B` sur chaque �l�ment de tableaux.
	// Note: �get_global_id� est une fonction pr�d�finie pour les items de travail (�Work-Item Functions�).
	// https://www.khronos.org/registry/OpenCL/specs/opencl-1.1.pdf
	// https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf
	// https://www.khronos.org/registry/OpenCL/specs/opencl-2.0.pdf
	/*std::string kernel_code =
		"   void kernel demo(global const int* A, global const int* B, global int* C) {             "
		"       C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];                    "
		"   }                                                                                       ";*/

	std::string kernel_code =
		"   void kernel multiplication(global const int* mat1, global const int* mat2, global int* mat3) {										 "
		"       mat3[get_global_id(1), get_global_id(0)] = mat1[get_global_id(1), get_global_id(0)] * mat2[get_global_id(1), get_global_id(0)];  "
		"   }																																	 ";

	// Sauvegarde de la source.
	sources.push_back({
		kernel_code.c_str(),
		kernel_code.length()
		});

	// D�claration du programme GPU en l'associant � un contexte de calcul.
	cl::Program program(context, sources);

	// Compilation du programme GPU.
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
		exit(1);
	}

	// Cr�ation des tampons GPU.
	//cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	//cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	//cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

	cl::Buffer buffer_mat1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 9);
	cl::Buffer buffer_mat2(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 9);
	cl::Buffer buffer_mat3(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 9);

	//int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	//int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

	// matrices pour tests
	int mat1[3][3];
	int mat2[3][3];
	ifstream file;
	file.open(args[1]);
	for (int i = 0;i < 3;i++)
	{
		for (int j = 0;j < 3;j++)
		{
			file >> mat1[i][j];
		}
	}
	file.close();
	file.open(args[2]);
	for (int i = 0;i < 3;i++)
	{
		for (int j = 0;j < 3;j++)
		{
			file >> mat2[i][j];
		}
	}
	file.close();



	std::cout << " matrice:" << std::endl;
	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < 3; y++) {
			std::cout << mat1[x][y] << " ";
		}
		std::cout << std::endl;
	}

	// Cr�ation de la queue de traitement GPU dans laquelle nous allons planifier des commandes.
	cl::CommandQueue queue(context, default_device);

	// Planification de l'initialisation des tableaux A et B (transf�re du CPU vers le GPU).
	//queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
	//queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);

	queue.enqueueWriteBuffer(buffer_mat1, CL_TRUE, 0, sizeof(int) * 9, mat1);
	queue.enqueueWriteBuffer(buffer_mat2, CL_TRUE, 0, sizeof(int) * 9, mat2);


	// --------------------------------------------------------------------------------
	// Ex�cution du noyau nomm� �demo�.
	// --------------------------------------------------------------------------------

	//     Premi�re avec foncteurs-noyaux (non disponible en OpenCL 1.2 mais l'est en 1.1 et 2.x).
	//     {
	//cl::KernelFunctor demo(cl::Kernel(program, "demo"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
	//demo(buffer_A, buffer_B, buffer_C);
	//     }

	//     Version g�n�rique.
	//     {
	cl::Kernel kernel_multiply = cl::Kernel(program, "multiplication");

	// Assignation des param�tres du noyau.
	/*kernel_add.setArg(0, buffer_A);
	kernel_add.setArg(1, buffer_B);
	kernel_add.setArg(2, buffer_C);*/
	kernel_multiply.setArg(0, buffer_mat1);
	kernel_multiply.setArg(1, buffer_mat2);
	kernel_multiply.setArg(2, buffer_mat3);

	// Param�tres de �enqueueNDRangeKernel� :
	//  - const Kernel &kernel  : Noyau � ex�cuter
	//  - const NDRange &offset : D�calages des indices globaux (cl::NullRange == aucun).
	//  - const NDRange &global : Dimension des items de travail (ex: �X�, �X * Y�, �X * Y * Z�, etc.).
	//  - const NDRange &local  : Dimension des groupes de travail locaux (nombre de work-items par work-group).
	queue.enqueueNDRangeKernel(kernel_multiply, cl::NullRange, cl::NDRange(3, 3), cl::NullRange);

	// Exemple d'une ex�cution en 2D.
	//queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(800, 600), cl::NullRange);
	queue.finish();
	//     }

	// D�claration du tampon d'extraction des donn�es (CPU).
	//int C[10];
	int mat3[3][3];

	// Planification de l'�criture des r�sultat de `buffer_C` vers `C` (transf�re du GPU vers le CPU).
	//queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);
	queue.enqueueReadBuffer(buffer_mat3, CL_TRUE, 0, sizeof(int) * 9, mat3);

	// Affichage des r�sultats � la console.
	/*std::cout << " result:" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << C[i] << " ";
	}*/

	// Affichage des r�sultats � la console.
	std::cout << " result:" << std::endl;
	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < 3; y++) {
			std::cout << mat3[x][y] << " ";
		}
		std::cout << std::endl;
	}

	// Fin.
	return 0;
}