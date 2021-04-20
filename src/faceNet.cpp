/*
 ============================================================================
 Name        : faceNet.c
 Author      : Cristian Johan Miranda Orosteguie
 Version     : 1.0
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "input.h"
#include "deep_learning_CMO.h"

#define image_min(A, B) ((A) < (B) ? (A) : (B))
#define image_max(A, B) ((A) < (B) ? (B) : (A))

using namespace cv;
using namespace std;

char nombre[15];
char linea[80];
char buffer[2000];
char enco[2000];


//https://github.com/espressif/esp-face/blob/master/image_util/image_util.c
void imageResize(float *initial_ima,float *final_image, int ini_w, int ini_h, int ini_c, int fin_w, int fin_h)
{
	float scale_x = (float)fin_w / ini_w;
	float scale_y = (float)fin_h / ini_h;

	int dst_stride = ini_c * ini_w;
	int src_stride = ini_c * fin_w;

	for (int y = 0; y < ini_h; y++)
	{
		float fy[2];
		fy[0] = (float)((y + 0.5) * scale_y - 0.5);
		int src_y = (int)fy[0];
		fy[0] -= src_y;
		fy[1] = 1 - fy[0];
		src_y = image_max(0, src_y);
		src_y = image_min(src_y, fin_h - 2);

		for (int x = 0; x < ini_w; x++)
		{
			float fx[2];
			fx[0] = (float)((x + 0.5) * scale_x - 0.5);
			int src_x = (int)fx[0];
			fx[0] -= src_x;
			if (src_x < 0)
			{
				fx[0] = 0;
				src_x = 0;
			}
			if (src_x > fin_w - 2)
			{
				fx[0] = 0;
				src_x = fin_w - 2;
			}
			fx[1] = 1 - fx[0]; // x2 - x

			for (int c = 0; c < ini_c; c++)
			{
				initial_ima[y * dst_stride + x * ini_c + c] = round(final_image[src_y * src_stride + src_x * ini_c + c] * fx[1] * fy[1] + final_image[src_y * src_stride + (src_x + 1) * ini_c + c] * fx[0] * fy[1] + final_image[(src_y + 1) * src_stride + src_x * ini_c + c] * fx[1] * fy[0] + final_image[(src_y + 1) * src_stride + (src_x + 1) * ini_c + c] * fx[0] * fy[0]);
			}
		}
	}

}


int main()
{
	Mat dest, gray, frame, cara;

	int bandera1 = 0; //para el mensanje de las caras
	int op, intento, num;
	//int i = 5;
	char afi[1];
	int b = 0; //manipular pixeles

	int contador = 0; //Variable volatil ATENCION: podria generar problemas


	string ruta;
	string txt;


	matrix_NHWC *out;
	matrix_NHWC *image = matrix_NHWC_alloc(1, 96, 96, 3);

	Mat imagePrueba = imread("/home/cristian/eclipse-workspace/DisplayImage/images/1.jpg");

	//char line[5];

	FILE *fp;


	//1.  Agregar nueva ID persona y generar encode
	//2.  Predecir
	//-1. Salir

	CascadeClassifier detector;

	if(!detector.load("/home/cristian/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"))
		cout << "No se puede abrir clasificador." << endl;

	VideoCapture cap(0);

	//Descomentar
	if(!cap.open(0))
	{
		cout << "No se puede acceder a la webcam." << endl;
		return -1;
	}

	while(true)
	{
		printf("\n************Ingresar la Opcion************\nOpcion: ");
		cin>>op;

		intento = 0;


		if (op == 2)
		{

			array<string,20> nombres; //Para 20 personas
			array<int,20> numeroNombres;

			//Cargamos el dataset de los encodes

			fp = fopen("/home/cristian/eclipse-workspace/faceNet/files/conteo.txt","r");
			if(fp ==NULL)
			{
				printf("\n**Archivo no encontrado**");
				op = -1;
				//return -1;
				break;
			}

			char *token;
			char *token2;

			txt = "";
			//string p[][];
			b=0;
			while(fgets(linea, 80, fp))
			{
				token = strtok(linea,"*");
				num = atoi(token);
				token = strtok(NULL,"*");
				if(b == 20)
				{
					printf("\n*****Error de memoria, hay mas de 20 identidades de usuarios");
					printf("Porfavor comunicarse con soporte");
					return -1;
				}
				nombres.at(b) = token;
				numeroNombres.at(b) = num;
				b= b +1;
				//here this
				//txt = txt +  "/home/cristian/eclipse-workspace/faceNet/encode/"+to_string(num) +"_"+token+".txt#";
				//cout<<token<<endl;
			}

			float encodeArr[num+1][128];
			//cout<<txt<<endl;
			//cout<<enco<<endl;
			fclose(fp);

			int cols = 0;


			for(int k = 0;  k < num +1; k ++)
			{
				ruta = "/home/cristian/eclipse-workspace/faceNet/encode/"+to_string(numeroNombres.at(k)) +"_"+nombres.at(k)+".txt";
				fp = fopen(ruta.c_str(),"r");
				if(fp ==NULL)
				{
					printf("\n**Archivo no encontrado**");
					op = -1;
				}
				else
				{
					fgets(enco,2000,fp);
					//cout<<"enconde: "<<enco<<endl;

					token2 = strtok(enco,"*");

					if(token2 != NULL)
					{
						while(token2 != NULL)
						{
							//cout<<"*num:"<<token2<<"*";
							encodeArr[k][cols] = atof(token2);
							token2 =strtok(NULL , "*");
							cols = cols + 1;
						}
						//cout<<endl;
					}
					cols = 0;

				}

				fclose(fp);

			}

			bandera1 = 0;

			//identificacion facial k - a varios
			while(op == 2)
			{
				if(!cap.open(0))
				{
						cerr << "ERROR! Unable to open camera\n";
						return -1;
				}
				else
				{
					cap >> frame;
					cap >> frame;

					cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
					equalizeHist(gray, dest);

					vector<Rect> rect;
					detector.detectMultiScale(dest, rect, 1.2, 3, 0, Size(60,60));

					for(Rect rc : rect)
					{
						if(bandera1 == 0)
						{
							printf("\nSe detectaron caras o cara");
							bandera1 = 1;
						}

						//rectangle(frame, Point(rc.x, rc.y),Point(rc.x + rc.width, rc.y + rc.height), CV_RGB(0,255,0), 2);

						Rect myRoi(rc.x, rc.y, rc.width, rc.height);

						cara = frame(myRoi);

						imshow("cara", cara);
						waitKey(0);
						destroyAllWindows();

						printf("\nDeseas identificar el rostro anterior? (N : Toma otra captura)");
						printf("\n YY: SI Identifica y SI guarda la imagen");
						printf("\n YN: SI Identifica pero NO guarda la imagen");
						printf("\n NY: NO Identifica y SI guarda la imagen");
						printf("\n NN: NO Identifica y NO guarda la imagen");
						printf("\n __: ?");
						cin>>afi;
						afi[0] = toupper(afi[0]);
						afi[1] = toupper(afi[1]);

						ruta = "/home/cristian/eclipse-workspace/faceNet/dataset/" + to_string(contador) +"_"+to_string(num+1)+".jpg";

						if(afi[1] == 'Y')
						{
							//modificar ruta
							if(imwrite(ruta,cara) == false)
							{
								printf("\n No se pudo guardar la imagen");
							}
							else
							{
								printf("\n Imagen guardada");
								contador = contador + 1;
							}

						}

						if( afi[0] == 'Y')
						{
							matrix_NHWC *imageSin = matrix_NHWC_alloc(1, rc.width, rc.height, 3);

							b=0;
							for(int j = 0; j < cara.rows; j++)
							{

								for(int k = 0; k< cara.cols; k++)
								{

									Vec3b pixel = cara.at<Vec3b>(j, k, 0);

									imageSin->item[b] =pixel[2];
									b = b +1;
									imageSin->item[b] = pixel[1];
									b = b +1;
									imageSin->item[b] = pixel[0];
									b = b +1;

								}
							}

							imageResize(image->item, imageSin->item, image->w, image->h, image->c, imageSin->w, imageSin->h);

							for(int k =0; k < image->w*image->h*image->c*image->n; k++)
							{
								image->item[k] = image->item[k+1]/255;
							}

							out = faceNetIdent(image);


							//Distancia L2

							float dis = 0.0;
							float mini  = 9999.9;
							int id = -1;
							txt = "\n#";
							for(int k = 0; k < num+1; k++)
							{
								dis = 0;

								for(int j = 0; j < 128; j++)
								{
									dis = dis + pow(out->item[j] - encodeArr[k][j],2);
								}

								dis = sqrt(dis);
								txt = txt+to_string(dis)+"**";

								if(dis <= 0.7)
								{
									if(dis < mini )
									{
										id = numeroNombres.at(k);
										//cout<<endl<<"id: "<<id;
										mini = dis;
									}
								}


								printf("**%.4f", dis);
							}

							printf("\n***La id que se identifico fue: %i", id);

							fp = fopen("/home/cristian/eclipse-workspace/faceNet/dataset/resultados.txt","a");
							fputs(txt.c_str(),fp);
							fclose(fp);

							cmo_lib_free(imageSin);
							cmo_lib_free(out);

							op = 0;

						}


					}
				}
				cap.release();

			}


		}


		if(op == 1)
		{
			printf("\nEsta en la Opcion 1");

			fp = fopen("/home/cristian/eclipse-workspace/faceNet/files/conteoNum.txt","r");
			if(fp ==NULL)
			{
				printf("\n**Archivo no encontrado**");
				op = -1;
			}
			fgets(linea,5,fp);
			num = atoi(linea);
			fclose(fp);


			printf("\nIntroduzca el nombre de la persona: ");
			cin>>nombre;

			while(op == 1)
			{

				intento = intento +1;

				if(!cap.open(0))
				{
						cerr << "ERROR! Unable to open camera\n";
						return -1;
				}
				else
				{
					cap >> frame;
					cap >> frame;

					cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
					equalizeHist(gray, dest);

					vector<Rect> rect;
					detector.detectMultiScale(dest, rect, 1.2, 3, 0, Size(60,60));

					for(Rect rc : rect)
					{
						if(bandera1 == 0)
						{
							printf("\nSe detectaron caras o cara");
							bandera1 = 1;
						}

						//rectangle(frame, Point(rc.x, rc.y),Point(rc.x + rc.width, rc.y + rc.height), CV_RGB(0,255,0), 2);

						Rect myRoi(rc.x, rc.y, rc.width, rc.height);

						cara = frame(myRoi);

						imshow("cara", cara);
						waitKey(0);
						destroyAllWindows();

						printf("\nDeseas guardar la imagen anterior (Agregar la persona a la base de datos"
								")? \n(EL encode siempre se genera con la ultima imagen guardada)");
						printf("\nY: guardar -> generar encoder -> salir al menu de opciones");
						printf("\nN: Localizar nuevamente la cara");
						printf("\nY or N?: ");
						cin>>afi;
						afi[0] = toupper(afi[0]);

						if( afi[0] == 'Y')
						{

							op= 0;
							fp = fopen("/home/cristian/eclipse-workspace/faceNet/files/conteo.txt","a");
							if(fp ==NULL)
							{
								printf("\n**Archivo no encontrado**");
								op = -1;
							}
							fputs( (to_string(num)+"*"+nombre+"*\n").c_str(),fp);
							fclose(fp);

							matrix_NHWC *imageSin = matrix_NHWC_alloc(1, rc.width, rc.height, 3);

							b=0;
							for(int j = 0; j < cara.rows; j++)
							{

								for(int k = 0; k< cara.cols; k++)
								{

									Vec3b pixel = cara.at<Vec3b>(j, k, 0);

									imageSin->item[b] =pixel[2];
									b = b +1;
									imageSin->item[b] = pixel[1];
									b = b +1;
									imageSin->item[b] = pixel[0];
									b = b +1;

								}
							}

							imageResize(image->item, imageSin->item, image->w, image->h, image->c, imageSin->w, imageSin->h);

							Mat mostrar = Mat::zeros(96,96,CV_8UC3);

							b=0;
							for(int j = 0; j < mostrar.rows; j++)
							{
								for(int k = 0; k< mostrar.cols; k++)
								{
									mostrar.at<Vec3b>(j, k, 0)[2] = int(image->item[b]);
									mostrar.at<Vec3b>(j, k, 0)[1] = int(image->item[b+1]);
									mostrar.at<Vec3b>(j, k, 0)[0] = int(image->item[b+2]);
									b=b+3;
								}
							}



							ruta = "/home/cristian/eclipse-workspace/faceNet/image/" + to_string(num) +"_"+nombre+".jpg";

							if(imwrite(ruta,mostrar) == false)
							{
								printf("\n No se pudo guardar la imagen");
							}
							else
							{
								printf("\nImagen guardada");
							}
							ruta = "/home/cristian/eclipse-workspace/faceNet/encode/"+to_string(num) +"_"+nombre+".txt";

							//cout<<ruta<<endl;
							fp = fopen(ruta.c_str(),"w");

							if(fp ==NULL)
							{
								printf("\n**No se pudo guardar el encode**");
								op = -1;
							}
							else
							{

								for(int k =0; k < image->w*image->h*image->c*image->n; k++)
								{
									image->item[k] = image->item[k+1]/255;
								}

								out = faceNetIdent(image);

								txt = "";

								for(int k = 0; k < 128; k++)
								{
									txt = txt + to_string(out->item[k]);
									if(k != 127)
									{
										txt = txt + "*";
									}
								}
								fputs(txt.c_str(),fp);

								printf("\nEncode Guardado");
							}

							fclose(fp);

							//imshow("MOSTRAR", mostrar);
							//waitKey(0);
							//destroyAllWindows();


							//cout<<ruta<<endl;
							//printf("\n*w:%i*h:%i*c:%i",imageSin->w, imageSin->h, imageSin->c);

							cmo_lib_free(imageSin);
							cmo_lib_free(out);


						}

					}

					bandera1 = 0;
					cap.release();
				}




			}

			//Actualizamos el conteo
			fp = fopen("/home/cristian/eclipse-workspace/faceNet/files/conteoNum.txt","w");
			if(fp ==NULL)
			{
				printf("\n**Archivo no encontrado**");
				op = -1;
			}
			num = num + 1;
			fputs(to_string(num).c_str(), fp);
			fclose(fp);





		}


		if(op == -1)
		{
			printf("\nEstas Saliendo del programa");
			break;
		}

		if( (op!=1)&(op!=2)&(op!=-1)&(op!=0))
		{
			printf("\n Opcion no valida, porfavor ingresar Opcion valida");
		}



	}

	cmo_lib_free(image);

	destroyAllWindows();
	cap.release();



	/*
	VideoCapture cap;


	if(!cap.open(0))
		cout << "No se puede acceder a la webcam." << endl;

	CascadeClassifier detector;

	if(!detector.load("/home/cristian/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"))
		cout << "No se puede abrir clasificador." << endl;

	while(true)
	{
		Mat dest, gray, imagen;

		cap >> imagen;

		cv::cvtColor(imagen, gray, cv::COLOR_RGB2GRAY);
		equalizeHist(gray, dest);

		vector<Rect> rect;
		detector.detectMultiScale(dest, rect, 1.2, 3, 0, Size(60,60));

		for(Rect rc : rect)
		{
			rectangle(imagen,
				Point(rc.x, rc.y),
				Point(rc.x + rc.width, rc.y + rc.height),
				CV_RGB(0,255,0), 2);
		}

		imshow("Deteccion de rostros", imagen);

		if(waitKey(1) >= 0) break;
	}*/


	system("read -p 'Press Enter to continue...' var");


	return EXIT_SUCCESS;
}



/*
Mat dest;
Mat gray;
Mat imagen = imread("/home/cristian/eclipse-workspace/faceNet/image/lenna.png");

CascadeClassifier detector;

if(!detector.load("/home/cristian/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"))
	cout << "No se puede abrir clasificador." << endl;

cv::cvtColor(imagen, gray, cv::COLOR_RGB2GRAY);

equalizeHist(gray, dest);

vector<Rect> rect;
detector.detectMultiScale(dest, rect);

for(Rect rc : rect)
{
	rectangle(imagen,
			  Point(rc.x, rc.y),
			  Point(rc.x + rc.width, rc.y + rc.height),
			  CV_RGB(0,255,0), 2);
}

imshow("Imagen original", imagen);
imshow("Imagen en escala de grises", gray);
imshow("Imagen al aplicar ecualizacion de histograma",dest);

waitKey(0);
*/



/*
matrix_NHWC *image = matrix_NHWC_alloc(1, 96, 96, 3);

for (int i = 0; i < image->h * image->w * image->c; i++)
{
	image->item[i] = input_item_array[i] / 255.0f;
}


matrix_NHWC *out;


for(int k = 0; k < 50; k++)
{
	out = faceNetIdent(image);

	cmo_lib_free(out);

	if((k == 10) || (k==50))
	{
		system("read -p 'Press Enter to continue...' var");
	}


}

out = faceNetIdent(image);

printf("\n\n******Salida********\n\n");

printf("c : %i \n", out->c);
printf("h : %i \n", out->h);
printf("w : %i \n", out->w);
printf("n : %i \n", out->n);

for(int i=0; i < (out->h)*(out->w)*(out->n)*(out->c) ; i++)
//for(int i=0; i < 128 ; i++)
{
	printf(" %.6f ", out->item[i]);

}

cmo_lib_free(out);*/


