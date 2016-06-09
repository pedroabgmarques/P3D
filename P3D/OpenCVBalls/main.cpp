/*
OpenCV + OpenGL: Realidade Aumentada

Trabalho realizado no âmbito da cadeira de Programação 3D do 2º Ano / 2º Semestre de EDJD @ IPCA
Victor Marçal: 6713
Paulo Nunes: 11287
Pedro Marques: 10855
Vitor Gomes: 10658

Este programa implementa 4 modos de realidade aumentada:
1 - "Positional Tracking"
	É feito o tracking de um objeto com uma cor diferente da do background (configurável em runtime) e a posição do
	objeto é aplicada à posição da camara.

2 - "Augmented Reality"
	É feito o tracking de um objeto colorido e o objeto detetado é "substituído" por uma esfera com a textura da terra,
	em volta da qual se move outra esfera com a textura da lua.

3 - "Instragram masks"
	É feito o tracking da face do utilizador e uma textura é renderizada sobreposta à face, acompanhando o movimento
	do utilizador. É possivel alterar a textura utilizada com a tecla N.

4 - "Marker Detection"
	É feito o tracking de um marker Aruco e um modelo 3D é renderizado na posição e com a orientação do marker.
	É possível alterar o modelo renderizado com a tecla N.

Controlos:
q - Sair
m - Próximo modo
n - Próxima textura / modelo (modos 3 e 4)

Dependências / Frameworks utilizadas:

OpenGL
https://www.opengl.org/

OpenCV
http://opencv.org/

Glew
http://glew.sourceforge.net/

Freeglut
http://freeglut.sourceforge.net/

Aruco
http://www.uco.es/investiga/grupos/ava/node/26

TGA
ajbrf@yahoo.com

VideoFaceDetector
https://github.com/mc-jesus/face_detect_n_track

GLM
http://www.pobox.com/~nate
*/

#pragma region Includes

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/video.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "Dependencies\glew\glew.h"
#include "Dependencies\freeglut\freeglut.h"
#include "Dependencies\aruco\aruco.h"
#include "Dependencies\aruco\cvdrawingutils.h"
#include <math.h>
#include "tga.h"
#include "VideoFaceDetector.h"
#include "glm.h"

#pragma endregion

#pragma region Namespaces

using namespace cv;
using namespace std;
using namespace aruco;

#pragma endregion

#pragma region Propriedades

//Materiais utilizados
Mat frameOriginal, frameHSV, frameFiltered, frameFlipped, fgMaskMOG, controlFlipped, tempimage, tempimage2, 
	faceDetection, undistorted;

//Instância de MOG Background subtractor
Ptr<BackgroundSubtractor> pMOG; 

//Valores iniciais do filtro de cor
int iLowH = 0;
int iHighH = 179;
int iLowS = 133;
int iHighS = 250;
int iLowV = 180;
int iHighV = 255;

//Instância de camera capture
VideoCapture cap(CV_CAP_ANY);
bool frameCapturedSuccessfully = false;
//Dimensões do frame capturado / janela glut
int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
Size GlWindowSize = Size(width, height);

//Classificadores para faces e olhos
char *classifierFaces = "haarcascade_frontalface_default.xml";
char *classifierEyes = "haarcascade_mcs_eyepair_big.xml";

//Detector de faces
VideoFaceDetector detector(classifierFaces, cap);
Point circleCenter = Point(0, 0);
int circleRadius = 2;
Rect faceRectangle;
Point facePosition;

//Display list para o chão do Modo 1
int myDL;

//Modos existentes: 1 - Position tracking; 2 - Augmented reality; 3 - Instagram masks; 4 - Marker Detection
int demoModes = 4;
//Modo atual
int demoMode = 0;

//Modo 2, Planeta / Lua - texturas, rotação, orbita, etc.
tgaInfo *im;
GLuint textureEarth, textureMoon;
GLUquadric *mysolid;
GLfloat spin = 0.05;
float planetCenterX = 0, planetCenterY = 0;
float raioOrbita = 0.1;
float periodoOrbital = 1.0;
float moonOrbitIterator = 0;
GLuint textures[2];

//Modo 3, Instagram masks - gestão de texturas
const int nFacetextures = 4;
GLuint faceDetectionTextures[nFacetextures];
int faceTextureAtual = 0;

//Usado para implementar um rolling moving average de modo a limpar o sinal 
float newValuesWeight = 1.0;
float accumulatorX = 0, accumulatorY = 0, accumulatorZ = 0;

//Usado para implementar o marker detection (biblioteca Aruco)
MarkerDetector MDetector;
vector<Marker> Markers;
CameraParameters CamParam;

//Modelos 3D para o modo de marker detection
const int nModelos = 7;
int modeloAtual = 0;
GLMmodel* pmodel[nModelos];

#pragma endregion

#pragma region Methods Declaration
void floorAndWallsDL(void);
void applymaterial(int type);
#pragma endregion

#pragma region Metodos

#pragma region Initialize

//Opções e inicialização do OpenGL
void init(void)
{

	glEnable(GL_CULL_FACE_MODE);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	// Define tÃ©cnica de shading: GL_FLAT, GL_SMOOTH
	glShadeModel(GL_SMOOTH);

	glPolygonMode(GL_FRONT, GL_FILL); // GL_LINE, GL_POINT, GL_FILL

	// Compila o modelo
	floorAndWallsDL();
}

//Define e ativa duas fonte de luz: posicional e cónica
void initLights(void)
{
	// Define a luz ambiente global
	GLfloat global_ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	// Define a luz light0. Existem 8 fontes de luz no total.
	GLfloat light0_ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat light0_diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
	GLfloat light0_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	// Define a luz light1. Existem 8 fontes de luz no total.
	GLfloat light1_ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	GLfloat light1_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat light1_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat spot_angle = 45.0f;
	GLfloat spot_exp = 12.0f; // Maior valor = maior concentraÃ§Ã£o de luz no centro

	// Fonte de luz ambiente
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);

	// Fonte de luz posicional
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
	glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1);
	glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05);
	glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.05);

	// Fonte de luz cÃ³nica
	glLightfv(GL_LIGHT1, GL_AMBIENT, light1_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
	glLightfv(GL_LIGHT1, GL_SPECULAR, light1_specular);
	glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, spot_angle);
	glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, spot_exp);

	// Activa a utilizaÃ§Ã£o de iluminaÃ§Ã£o
	glEnable(GL_LIGHTING);
	// Activa a fonte de luz light0
	glEnable(GL_LIGHT0);
	// Activa a fonte de luz light1
	glEnable(GL_LIGHT1);
}

#pragma endregion

#pragma region DisplayLists

//Define a display list do chão do Modo 1
void floorAndWallsDL(void)
{
	int x, z;

	myDL = glGenLists(1);

	glNewList(myDL, GL_COMPILE);
	// Floor
	for (x = -100; x <= 100; x += 2)
	{
		for (z = -100; z <= 100; z += 2)
		{
			applymaterial(0);

			glBegin(GL_QUADS);
			glNormal3f(0.0, 1.0, 0.0);
			glVertex3f(x, 0.0f, z);				// Top Left
			glVertex3f(x + 1.0f, 0.0f, z);		// Top Right
			glVertex3f(x + 1.0f, 0.0f, z - 1.0f);	// Bottom Right
			glVertex3f(x, 0.0f, z - 1.0f);		// Bottom Left
			glEnd();

		}
		for (z = -99; z <= 100; z += 2)
		{
			applymaterial(1);

			glBegin(GL_QUADS);
			glNormal3f(0.0f, 1.0f, 0.0f);
			glVertex3f(x, 0.0f, z);				// Top Left
			glVertex3f(x + 1.0f, 0.0f, z);		// Top Right
			glVertex3f(x + 1.0f, 0.0f, z - 1.0f);	// Bottom Right
			glVertex3f(x, 0.0f, z - 1.0f);		// Bottom Left
			glEnd();

		}
	}
	for (x = -99; x <= 100; x += 2)
	{
		for (z = -99; z <= 100; z += 2)
		{
			applymaterial(0);

			glBegin(GL_QUADS);
			glNormal3f(0.0, 1.0, 0.0);
			glVertex3f(x, 0.0f, z);				// Top Left
			glVertex3f(x + 1.0f, 0.0f, z);		// Top Right
			glVertex3f(x + 1.0f, 0.0f, z - 1.0f);	// Bottom Right
			glVertex3f(x, 0.0f, z - 1.0f);		// Bottom Left
			glEnd();

		}
		for (z = -100; z <= 100; z += 2)
		{
			applymaterial(2);

			glBegin(GL_QUADS);
			glNormal3f(0.0f, 1.0f, 0.0f);
			glVertex3f(x, 0.0f, z);				// Top Left
			glVertex3f(x + 1.0f, 0.0f, z);		// Top Right
			glVertex3f(x + 1.0f, 0.0f, z - 1.0f);	// Bottom Right
			glVertex3f(x, 0.0f, z - 1.0f);		// Bottom Left
			glEnd();

		}
	}

	glBegin(GL_QUADS);
	glNormal3f(1.0f, 0.0f, 0.0f);
	glVertex3f(-100.0f, 20.0f, -100.0f);				// Top Left
	glVertex3f(100.0f, 20.0f, -100.0f);		// Top Right
	glVertex3f(100.0f, 0.0f, -100.0f);	// Bottom Right
	glVertex3f(-100.0f, 0.0f, -100.0f);		// Bottom Left
	glEnd();

	glBegin(GL_QUADS);
	glNormal3f(-1.0f, 0.0f, 0.0f);
	glVertex3f(-100.0f, 20.0f, 100.0f);				// Top Left
	glVertex3f(100.0f, 20.0f, 100.0f);		// Top Right
	glVertex3f(100.0f, 0.0f, 100.0f);	// Bottom Right
	glVertex3f(-100.0f, 0.0f, 100.0f);		// Bottom Left
	glEnd();

	glBegin(GL_QUADS);
	glNormal3f(-1.0f, 0.0f, 0.0f);
	glVertex3f(100.0f, 20.0f, -100.0f);				// Top Left
	glVertex3f(100.0f, 20.0f, 100.0f);		// Top Right
	glVertex3f(100.0f, 0.0f, 100.0f);	// Bottom Right
	glVertex3f(100.0f, 0.0f, -100.0f);		// Bottom Left
	glEnd();

	glBegin(GL_QUADS);
	glNormal3f(-1.0f, 0.0f, 0.0f);
	glVertex3f(-100.0f, 20.0f, -100.0f);				// Top Left
	glVertex3f(-100.0f, 20.0f, 100.0f);		// Top Right
	glVertex3f(-100.0f, 0.0f, 100.0f);	// Bottom Right
	glVertex3f(-100.0f, 0.0f, -100.0f);		// Bottom Left
	glEnd();



	glEndList();
}

#pragma endregion

#pragma region ApplyLightAndMaterial

//Define e aplica um determinado material
void applymaterial(int type)
{
	// Define as propriedades dos materiais
	// Type: 0 (Branco); 1 (Amarelo); 2 (Ciano); 3 (Branco-Emissor)
	GLfloat mat_ambient[4][4] = { { 1.0f, 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } };
	GLfloat mat_diffuse[4][4] = { { 0.5f, 0.5f, 0.5f, 1.0f }, { 0.5f, 0.5f, 0.0f, 1.0f }, { 0.0f, 0.5f, 0.5f, 1.0f }, { 0.5f, 0.5f, 0.5f, 1.0f } };
	GLfloat mat_specular[4][4] = { { 1.0f, 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } };
	GLfloat mat_emission[4][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } };
	GLfloat mat_shininess[4][1] = { { 20.0f }, { 20.0f }, { 20.0f }, { 20.0f } };

	if ((type >= 0) && (type < 4))
	{
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient[type]); // GL_FRONT, GL_FRONT_AND_BACK , GL_BACK, 
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse[type]);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular[type]);
		glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission[type]);
		glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess[type]);
	}
}

//Define a posição e aplica duas fontes de luz
void applylights(void)
{
	// Define a posição de light0
	GLfloat light0_position[] = { -1.0f, -3.0f, 0.0f, 1.0f };
	// Define a posição de direcção de light1
	GLfloat spot_position[] = { 0.0f, 3.0f, -1.0f, 1.0f };
	GLfloat spot_direction[] = { 0.0f, -1.0f, 0.0f };

	// Aplica a light0
	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);

	// Aplica a light1
	glLightfv(GL_LIGHT1, GL_POSITION, spot_position);
	glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, spot_direction);

	glDisable(GL_LIGHTING);

	// Desenha uma esfera que sinaliza a posição da light0
	glPushMatrix();
	glColor3f(1.0, 1.0, 1.0);
	glTranslatef(0.0f, 3.0f, 0.0f);
	glutSolidSphere(0.1, 20, 20);
	glPopMatrix();

	// Desenha uma esfera que sinaliza a posição da light1
	glPushMatrix();
	glColor3f(1.0, 1.0, 1.0);
	glTranslatef(0.0f, 3.0f, -10.0f);
	glutSolidSphere(0.1, 20, 20);
	glPopMatrix();

	glEnable(GL_LIGHTING);
}

#pragma endregion

#pragma region Utils

//Carrega um modelo 3D em formato obj com uma determinada escala para um vector de modelos 3D
void loadmodel(int nModelo, std::string nome, float scale)
{
	if (pmodel[nModelo] == NULL)
	{

		std::string impathfile = "models/" + nome + ".obj";
		std::vector<char> writable(impathfile.begin(), impathfile.end());
		writable.push_back('\0');

		pmodel[nModelo] = glmReadOBJ(&writable[0]);
		if (pmodel[nModelo] == NULL) { exit(0); }
		else
		{
			glmUnitize(pmodel[nModelo]);
			glmLinearTexture(pmodel[nModelo]);
			glmScale(pmodel[nModelo], scale);
			glmFacetNormals(pmodel[nModelo]);
			glmVertexNormals(pmodel[nModelo], 90.0);
		}
	}
}

//Carrega uma textura em formato tga para um vetor de texturas
void load_tga_image(std::string nome, GLuint texture, bool transparency)
{
	std::string impathfile = "textures/" + nome + ".tga";

	std::vector<char> writable(impathfile.begin(), impathfile.end());
	writable.push_back('\0');

	// Carrega a imagem de textura
	im = tgaLoad(&writable[0]);
	//printf("IMAGE INFO: %s\nstatus: %d\ntype: %d\npixelDepth: %d\nsize%d x %d\n", impathfile, im->status, im->type, im->pixelDepth, im->width, im->height);

	// Seleciona a textura atual
	glBindTexture(GL_TEXTURE_2D, texture);

	// set up quadric object and turn on FILL draw style for it
	mysolid = gluNewQuadric();
	gluQuadricDrawStyle(mysolid, GLU_FILL);

	// turn on texture coordinate generator for the quadric
	gluQuadricTexture(mysolid, GL_TRUE);

	// select modulate to mix texture with color for shading
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // MIPMAP
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// build our texture mipmaps
	if (!transparency){
		gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, im->width, im->height, GL_RGB, GL_UNSIGNED_BYTE, im->imageData); // MIPMAP
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, im->width, im->height, 0, GL_RGB, GL_UNSIGNED_BYTE, im->imageData);
	}
	else{
		//Textura com transparÃªncia (anÃ©is de saturno)
		gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, im->width, im->height, GL_RGB, GL_UNSIGNED_BYTE, im->imageData); // MIPMAP
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im->width, im->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, im->imageData);
	}


	// Destroi a imagem
	tgaDestroy(im);
}

//Desenha um eixo 3D com um determinado comprimento
void drawAxes(float length)
{
	glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);

	glDisable(GL_LIGHTING);

	glBegin(GL_LINES);
	glColor3f(1, 0, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(length, 0, 0);

	glColor3f(0, 1, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(0, length, 0);

	glColor3f(0, 0, 1);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, length);
	glEnd();

	glPopAttrib();

}

//Aredonda por excesso um valor double
double round(double d)
{
	return floor(d + 0.5);
}

//Renderiza texto num espaço tridimensional, utilizando o metodo glutStrokeCharacter
void glRenderString(float x, float y, char str[])
{
	glDisable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 1.0);

	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(-30, 28 - y, (float)-glutStrokeWidth(GLUT_STROKE_MONO_ROMAN, str[0]) / 2);

	glColor3f(1.0f, 1.0f, 1.0f);
	glScalef(0.012, 0.012, 0.012);
	for (int i = 0; i < strlen(str); i++)glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, str[i]);

	glPopMatrix();

	glEnable(GL_LIGHTING);
}

//Tranforma coordenadas de ecrã em coordenadas do mundo
float ScreenToWorld(float input, float input_start, float input_end, float output_start, float output_end, float divisor){
	double slope = 1.0 * (output_end - output_start) / (input_end - input_start);
	return (output_start + slope * (input - input_start)) / divisor;
}

//Filtra um material (imagem da camara) por uma determinada cor, configuravel em runtime
void OrangeFilter(Mat& source, Mat& destination){
	inRange(source, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), destination); //Threshold the image

	//morphological opening (remove small objects from the foreground)
	erode(destination, destination, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(destination, destination, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	//morphological closing (fill small holes in the foreground)
	dilate(destination, destination, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(destination, destination, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

}

//Detecta um objeto colorido através da utilização dos metodos findContours, approxPolyDP e minEnclosingCircle
void ObjectDetection(Mat& src_gray, Mat& src_display)
{

	int largest_area = 0;
	int largest_contour_index = 0;
	RNG rng(12345);
	Scalar color;

	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(src_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

	if (contours.size() > 0){

		//Foram encontrados contornos

		//cout << contours.size() << endl;

		for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
		{
			double a = contourArea(contours[i], false);  //  Find the area of contour
			if (a>largest_area){
				largest_area = a;
				largest_contour_index = i;                //Store the index of largest contour
			}
		}

		//cout << largest_contour_index << endl << endl;

		vector<vector<Point> > contours_poly(contours.size());
		vector<Point2f>center(contours.size());
		vector<float>radius(contours.size());

		approxPolyDP(Mat(contours[largest_contour_index]), contours_poly[largest_contour_index], 3, true);
		minEnclosingCircle((Mat)contours_poly[largest_contour_index], center[largest_contour_index], radius[largest_contour_index]);

		//Encontramos um circulo
		/*for (int i = 0; i< contours.size(); i++)
		{
		drawContours(src_display, contours_poly, i, Scalar(0, 0, 0), 1, 8, vector<Vec4i>(), 0, Point());
		}*/
		if ((int)radius[largest_contour_index] > 15)
		{
			/*circle(src_display, center[largest_contour_index], (int)radius[largest_contour_index], Scalar(255, 0, 0), 2, 8, 0);
			circle(src_display, center[largest_contour_index], 5, Scalar(255, 0, 0), -1);*/

			switch (demoMode)
			{
			case 0:{
				newValuesWeight = 0.4;
				//Positional Tracking
				accumulatorX = (newValuesWeight * center[largest_contour_index].x) + (1.0 - newValuesWeight) * accumulatorX;
				accumulatorY = (newValuesWeight * center[largest_contour_index].y) + (1.0 - newValuesWeight) * accumulatorY;
				accumulatorZ = (newValuesWeight * (int)radius[largest_contour_index]) + (1.0 - newValuesWeight) * accumulatorZ;
				circleCenter = Point(accumulatorX, accumulatorY);
				circleRadius = accumulatorZ;
				break;
			}
			case 1:{
				//Realidade aumentada, planeta por cima da bola
				//X e Y acompanham instantaneamente, limpamos o ruido do raio do planeta
				newValuesWeight = 1.0;
				accumulatorX = (newValuesWeight * center[largest_contour_index].x) + (1.0 - newValuesWeight) * accumulatorX;
				accumulatorY = (newValuesWeight * center[largest_contour_index].y) + (1.0 - newValuesWeight) * accumulatorY;
				circleCenter = Point(accumulatorX, accumulatorY);
				newValuesWeight = 0.8;
				accumulatorZ = (newValuesWeight * (int)radius[largest_contour_index]) + (1.0 - newValuesWeight) * accumulatorZ;
				circleRadius = accumulatorZ;
				break;
			}
			default:
				break;
			}

		}
	}

	///HoughDetection - Slow and unreliable

	//// will hold the results of the detection
	//std::vector<Vec3f> circles;
	//// runs the actual detection
	//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 16, cannyThreshold, accumulatorThreshold, 5, 800);
	//for (size_t i = 0; i < circles.size(); i++)
	//{
	//	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	//	int radius = cvRound(circles[i][2]);
	//	// circle center
	//	circle(src_display, center, 3, Scalar(0, 255, 0), -1, 8, 0);
	//	// circle outline
	//	circle(src_display, center, radius, Scalar(0, 0, 255), 3, 4, 0);
	//}

	//Flip and show the control window
	//flip(src_gray, controlFlipped, 0);


}

#pragma endregion

#pragma region Callbacks

//Lógica de deteção de objetos / face / marcadores e renderização de cada um dos modos
void display()
{
	// clear the window
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// read a new frame from video
	if (frameCapturedSuccessfully) //if not success, break loop
	{
		if (demoMode == 0 || demoMode == 1){
			cvtColor(frameOriginal, frameHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

			//Filtrar para deixar apenas coisas avermelhadas
			OrangeFilter(frameHSV, frameFiltered);

			//Retirar o background
			pMOG->operator()(frameFiltered, fgMaskMOG);

			//Blur para suavizar as fronteiras
			GaussianBlur(fgMaskMOG, frameFiltered, Size(9, 9), 4, 4);

			flip(frameFiltered, controlFlipped, 1);
			imshow("Control", controlFlipped);

			//runs the detection, and update the display
			HoughDetection(frameFiltered, frameOriginal);
		}
	}
	else{
		cout << "No frame captured!" << endl;
	}

	//////////////////////////////////////////////////////////////////////////////////
	// Here, set up new parameters to render a scene viewed from the camera.

	//set viewport
	//glViewport(0, 0, frameFlipped.size().width, frameFlipped.size().height);
	glViewport(0, 0, width, height);


	float zoomRange = 0;
	float x = 0, y = 0, z = 0;
	float raioOrbitaAtual = 0;
	switch (demoMode)
	{
	case 0:
		//MODO POSITIONAL TRACKING

		glDisable(GL_TEXTURE_2D);

		//set projection matrix using intrinsic camera params
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//FPV, zoom afeta FOV
		gluPerspective(60 - circleRadius / 5, width / height, 0.1, 100);

		//you will have to set modelview matrix using extrinsic camera params
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);

		glPushMatrix();

		zoomRange = RangeAToRangeB((float)circleRadius, 20, 150, 0, 10, 1);

		//move to the position where you want the 3D object to go
		glTranslatef(RangeAToRangeB((float)circleCenter.x, 0.0, (float)width, -width / 2.0, width / 2.0, 100.0),
			RangeAToRangeB((float)circleCenter.y, 0.0, (float)height, -height / 2.0, height / 2.0, 50.0),
			zoomRange);


		// Floor
		glCallList(myDL);

		drawAxes(1.0);

		applylights();

		glTranslatef(-2, 0, 0);

		glutSolidTeapot(0.5);

		glTranslatef(4, 0, 0);

		glutSolidTeacup(0.5);

		glPopMatrix();

		glRenderString(0.0f, 0.0f, "Modo 1 - Positional Tracking");
		glRenderString(0.0f, 2.0f, "Use um objeto vermelho");
		glRenderString(0.0f, 4.0f, "(de preferencia uma esfera).");
		glRenderString(0.0f, 6.0f, "Pode configurar a cor a detetar");
		glRenderString(0.0f, 8.0f, "com os sliders da janela Controlo.");
		glRenderString(0.0f, 10.0f, "Tecla M para mudar de modo");

		break;
	case 1:
		//MODO AUGMENTED REALITY PLANETA

		glClear(GL_COLOR_BUFFER_BIT);

		glDisable(GL_TEXTURE_2D);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		flip(frameOriginal, tempimage, 0);
		flip(tempimage, tempimage2, 1);
		putText(tempimage2, "Modo 2 - Augmented Reality", cvPoint(10, height - 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		putText(tempimage2, "Utilize um objeto vermelho (de preferencia uma esfera).", cvPoint(10, height - 40), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		putText(tempimage2, "Pode alterar a cor a detetar com os sliders da janela Controlo", cvPoint(10, height - 60), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		putText(tempimage2, "Tecla M para passar ao proximo modo", cvPoint(10, height - 80), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		glDrawPixels(tempimage2.size().width, tempimage2.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage2.ptr());

		glClear(GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho((float)width / (float)height, (float)-width / (float)height, -1, 1, -100, 100);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glEnable(GL_TEXTURE_2D);

		glColor4f(1.0, 1.0, 1.0, 1.0);

		glBindTexture(GL_TEXTURE_2D, textures[0]);

		applymaterial(0);

		applylights();

		// Draw sphere
		glPushMatrix();

		planetCenterX = RangeAToRangeB((float)circleCenter.x, 0.0, (float)width, -width / 2.0, width / 2.0, width / 2.65);
		planetCenterY = -RangeAToRangeB((float)circleCenter.y, 0.0, (float)height, -height / 2.0, height / 2.0, height / 2.0);

		//move to the position where you want the 3D object to go
		glTranslatef(planetCenterX, planetCenterY, 0);
		glRotatef(-spin, 0.0, 1.0, 0.0);
		//Endireitar os planetas
		glRotatef(-90.0, 1.0, 0.0, 0.0);
		gluSphere(mysolid, circleRadius / (height / 2.0), 100, 100);

		glPopMatrix();
		glPushMatrix();

		glBindTexture(GL_TEXTURE_2D, textures[1]);

		applymaterial(0);

		applylights();

		raioOrbitaAtual = raioOrbita + (circleRadius / 100.0);
		//CÃ¡lculo do movimento de translaÃ§Ã£o do planeta em volta do sol
		x = planetCenterX + raioOrbitaAtual * sin(moonOrbitIterator / 180.0 * 3.14);
		y = planetCenterY;// *cos(moonOrbitIterator / 180.0 * 3.14);
		z = circleRadius / (height / 2.0) + raioOrbitaAtual * cos(moonOrbitIterator / 180.0 * 3.14);

		//Esta linha define a velocidade orbital
		moonOrbitIterator += 3.14 * 2 / (periodoOrbital);
		if (moonOrbitIterator >= 360)
		{
			//Demos uma volta completa ao planeta (um ano)
			moonOrbitIterator = 0;
		}

		glTranslatef(
			x,
			y,
			z);

		//Endireitar os planetas
		glRotatef(-90.0, 1.0, 0.0, 0.0);

		//RotaÃ§Ã£o sobre si prÃ³prio
		//glRotatef(_rotacaoAtual, 0.0, 1.0, 0.0);

		//Endireitar os planetas
		//glRotatef(-90.0, 1.0, 0.0, 0.0);

		gluSphere(mysolid, (circleRadius / (height / 2.0)) / 2.0, 64, 64);

		glPopMatrix();

		spin = spin + 2;
		if (spin > 360.0) spin = spin - 360.0;

		break;
	case 2:{
		//MODO FACE DETECTION

		glClear(GL_COLOR_BUFFER_BIT);

		glDisable(GL_TEXTURE_2D);

		//Deteta as faces e olhos
		//detectFaces(frameOriginal);

		//Debug - desenhar o rectangulo e centro de deteÃ§Ã£o de face
		/*cv::rectangle(frameOriginal, detector.face(), cv::Scalar(255, 0, 0));
		cv::circle(frameOriginal, detector.facePosition(), 30, cv::Scalar(0, 255, 0));*/

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		////Desenha o resultado da deteÃ§Ã£o
		flip(frameOriginal, tempimage, 0);
		flip(tempimage, tempimage2, 1);
		putText(tempimage2, "Modo 3 - Face Detection", cvPoint(10, height - 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		putText(tempimage2, "Tecla N para alterar a textura", cvPoint(10, height - 40), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		putText(tempimage2, "Tecla M para passar ao proximo modo", cvPoint(10, height - 60), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		glDrawPixels(tempimage2.size().width, tempimage2.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage2.ptr());

		glClear(GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho((float)width / (float)height, (float)-width / (float)height, -1, 1, -100, 100);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glEnable(GL_TEXTURE_2D);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glColor4f(1.0, 1.0, 1.0, 1.0);

		glBindTexture(GL_TEXTURE_2D, faceDetectionTextures[faceTextureAtual]);

		applymaterial(0);

		applylights();

		glPushMatrix();
		Rect face = detector.face();
		Point facePos = detector.facePosition();

		//Escrever valores
		//cout << face.width << endl;

		float scale = RangeAToRangeB((float)face.width, 80.0, 480.0, 0.35, 1.4, 1.0);

		newValuesWeight = 0.3;
		accumulatorZ = (newValuesWeight * scale) + (1.0 - newValuesWeight) * accumulatorZ;

		//Dar a escala correcta Ã  textura aplicada

		glScalef(accumulatorZ, accumulatorZ, 0);

		//Escrever valores
		//cout << facePos.x << " x " << facePos.y << endl;

		//Colocar a textura no sitio certo
		float faceCenterX = RangeAToRangeB((float)facePos.x, 10, 640, -width / 2.0, width / 2.0, 180);
		float faceCenterY = -RangeAToRangeB((float)facePos.y, 40, 480, -height / 2.0, height / 2.0, 180);

		newValuesWeight = 0.5;

		accumulatorX = (newValuesWeight * faceCenterX) + (1.0 - newValuesWeight) * accumulatorX;
		accumulatorY = (newValuesWeight * faceCenterY) + (1.0 - newValuesWeight) * accumulatorY;

		//cout << faceCenterX << " x " << faceCenterY << endl << endl;

		glTranslatef(accumulatorX, accumulatorY, 0);

		//Desenhar a textura num quad
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f);
		glVertex3f(-1.0f, -1.0f, 0.0f);
		glTexCoord2f(1.0f, 0.0f); // sempre 1.0f se quiser aplicar toda a textura
		glVertex3f(1.0f, -1.0f, 0.0f);
		glTexCoord2f(1.0f, 1.0f);
		glVertex3f(1.0f, 1.0f, 0.0f);
		glTexCoord2f(0.0f, 1.0f);
		glVertex3f(-1.0f, 1.0f, 0.0f);
		glEnd();
		glPopMatrix();

		break;
	}
	case 3:{
		//MODO MARKER DETECTION

		glDisable(GL_TEXTURE_2D);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho((float)width / (float)height, (float)-width / (float)height, -1, 1, -100, 100);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		/*glPixelZoom(-1, -1);
		glRasterPos2f(-1.3, 1);*/

		flip(frameOriginal, tempimage, 0);
		flip(tempimage, tempimage2, 1);

		//Fazer undistort Ã  imagem de acordo com os parametros da camara
		cv::undistort(tempimage2, undistorted, CamParam.CameraMatrix, CamParam.Distorsion);

		//Detetar marcadores
		MDetector.detect(undistorted, Markers, CamParam, 0.045);
		//Desenhar imagem da camara

		putText(undistorted, "Modo 4 - Marker Detection", cvPoint(10, height - 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		putText(undistorted, "Tecla N para alterar o modelo 3D", cvPoint(10, height - 40), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		putText(undistorted, "Tecla M para passar ao proximo modo", cvPoint(10, height - 60), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, 8, true);
		glDrawPixels(undistorted.size().width, undistorted.size().height, GL_BGR, GL_UNSIGNED_BYTE, undistorted.ptr());
		glClear(GL_DEPTH_BUFFER_BIT);

		double proj_matrix[16];

		////Desenhar cenas 3D na posiÃ§Ã£o do marker
		CamParam.glGetProjectionMatrix(undistorted.size(), GlWindowSize, proj_matrix, 0.05, 10, true);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glLoadMatrixd(proj_matrix);

		double modelview_matrix[16];

		applymaterial(0);

		applylights();

		for (unsigned int m = 0; m < Markers.size(); m++)
		{

			Markers[m].glGetModelViewMatrix(modelview_matrix);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glLoadMatrixd(modelview_matrix);

			float size = 0.05;

			glPushMatrix();
			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, textures[0]);
			glTranslatef(0.0, 0.0, size);
			glRotatef(90, 1.0, 0.0, 0.0);
			//gluSphere(mysolid, size / 2, 20, 20);
			glmDraw(pmodel[modeloAtual], GLM_SMOOTH | GLM_MATERIAL);
			glPopMatrix();
		}

		break;
	}
	default:
		break;
	}

	// show the rendering on the screen
	glutSwapBuffers();

	// post the next redisplay
	glutPostRedisplay();
}

//Callback de reshape da janela glut
void reshape(int w, int h)
{
	// set OpenGL viewport (drawable area)
	glViewport(0, 0, width, height);
}

//Callback de processamento do rato
void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{

	}
}

//Callback de processamento do teclado
void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q':
		// quit when q is pressed
		exit(0);
		break;
	case 'm':
		demoMode += 1;
		if (demoMode >= demoModes){
			demoMode = 0;
		}
		accumulatorX = 0;
		accumulatorY = 0;
		accumulatorZ = 0;
		break;
	case 'n':
		if (demoMode == 2){
			faceTextureAtual += 1;
			if (faceTextureAtual >= nFacetextures){
				faceTextureAtual = 0;
			}
		}
		if (demoMode == 3){
			modeloAtual += 1;
			if (modeloAtual >= nModelos){
				modeloAtual = 0;
			}
		}
		break;

	default:
		break;
	}
}

//Leitura de um novo frame a partir da camara
void idle()
{
	// grab a frame from the camera
	if (demoMode == 2){
		detector >> frameOriginal;
	}
	else{
		frameCapturedSuccessfully = cap.read(frameOriginal);
	}

	CamParam.resize(frameOriginal.size());

}

#pragma endregion

#pragma endregion

#pragma region Entry Point
int main(int argc, char** argv)
{
	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	pMOG = new BackgroundSubtractorMOG(); //MOG approach

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	//Create trackbars in "Control" window
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);

	// initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(width, height);
	glutCreateWindow("OpenGL / OpenCV Example");

	// InicializaÃ§Ãµes
	init();
	initLights();
	//Texturas para o planeta e lua
	glGenTextures(2, textures);
	load_tga_image("earth", textures[0], false);
	load_tga_image("moon", textures[1], false);
	//Texturas para sobrepor Ã  face detetada
	glGenTextures(4, faceDetectionTextures);
	load_tga_image("ironman", faceDetectionTextures[0], true);
	load_tga_image("mrt", faceDetectionTextures[1], true);
	load_tga_image("lion", faceDetectionTextures[2], true);
	load_tga_image("hitler", faceDetectionTextures[3], true);
	//Modelos 3D para o modo de marker detection
	loadmodel(0, "f-16", 0.05);
	loadmodel(1, "al", 0.05);
	loadmodel(2, "dolphins", 0.05);
	loadmodel(3, "flowers", 0.05);
	loadmodel(4, "porsche", 0.05);
	loadmodel(5, "rose+vase", 0.05);
	loadmodel(6, "soccerball", 0.03);

	//Read camera calibration files
	try{
		CamParam.readFromXMLFile("camera.xml");
	}
	catch (std::exception &ex){
		cout << "Exception: " << ex.what() << endl;
	}
	
	// set up GUI callback functions
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);

	// start GUI loop
	glutMainLoop();

	return 0;
}
#pragma endregion
