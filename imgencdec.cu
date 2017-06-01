/*
 * noise_remover.cpp
 *
 * This program removes noise from an image based on Speckle Reducing Anisotropic Diffusion
 * Y. Yu, S. Acton, Speckle reducing anisotropic diffusion,
 * IEEE Transactions on Image Processing 11(11)(2002) 1260-1270 <http://people.virginia.edu/~sc5nf/01097762.pdf>
 * Original implementation is Modified by Burak BASTEM
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <math.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE  16

#define MATCH(s) (!strcmp(argv[ac], (s)))




__global__ void rsaEncryption(int height,int width,unsigned char *image_device, uint64_t *writebackdevice,uint64_t publicKey,uint64_t mod, int recursion,int *binarydevice){

  int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   int tidx = threadIdx.x;
   int tidy = threadIdx.y;
   if(i >= height || j >= width) return;

  long k = i * width + j;    // position of current element


	uint64_t x=image_device[k];

  uint64_t d = 1;
  int remainder;


  for(int t = 0; t<=recursion-1;t++){

      if(binarydevice[t]==1){
        d = (d*x)%mod;
      }
        x = (x*x)%mod;

  }


	writebackdevice[k] =	d;



	}



  __global__ void rsaDecryption(int height,int width,unsigned char *image_device, uint64_t *readfromdevice,uint64_t privateKey,uint64_t mod,int recursionDec,int *binarydeviceDec){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
     int j = blockDim.y * blockIdx.y + threadIdx.y;
     int tidx = threadIdx.x;
     int tidy = threadIdx.y;
     if(i >= height || j >= width) return;

    long k = i * width + j;    // position of current element



    	uint64_t x=readfromdevice[k];

      int c = 0;
      uint64_t d = 1;


        for(int t = 0; t<=recursionDec-1;t++){

            if(binarydeviceDec[t]==1){
              d = (d*x)%mod;
            }
              x = (x*x)%mod;

        }

    	image_device[k] =	d;



  	}








int ex_gcd(int a,int b,int n) //computes the GCD using the Extended Euclid method
{
int x=0,y=1,lastx=1,lasty=0;
int temp,q;
while(b!=0)
{
temp =b;
q = a/b;
b = a%b;
a = temp;

temp=x;
x = lastx - q*x;
lastx = temp;

temp =y;
y = lasty - q*y;
lasty = temp;
}
if(n==1) return a;
else return lasty;
}



int calcPriv(int p, int q, int e){
int t = (p-1)*(q-1);
int n = p * q;
int d=1;
int x= (d * e) % t;
while(x!=1){
d = d+1;
	x = (d * e) % t;
}
return  d;
}

int calcPub(int p, int q){
	int t = (p-1)*(q-1);
	int n = p * q;
	int e = 1;
	srand(time(NULL));
	int random_num=rand()%t;
	for(int i=2;i<(t-random_num);i++){
		if(ex_gcd(t,i,1) == 1){
			e=i;
		}
	}
	return e;

}





// returns the current time
static const double kMicro = 1.0e-6;
double get_time() {
	struct timeval TV;
	struct timezone TZ;
	const int RC = gettimeofday(&TV, &TZ);
	if(RC == -1) {
		printf("ERROR: Bad call to gettimeofday\n");
		return(-1);
	}
	return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}


int main(int argc, char *argv[]){


	// Part I: allocate and initialize variables
	double time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8, time_9;	// time variables
	time_0 = get_time();
	const char *filename = "input.pgm";
	const char *outputname = "output.png";
	int width, height, pixelWidth, n_pixels;
	unsigned int temp;
	int prime1, prime2;
	uint64_t privateKey, publicKey;
//	cudaError_t err = cudaSuccess;
cudaError_t err = cudaSuccess;

	bool en_de;


	unsigned long long int denemeResult;

	// Part II: parse command line arguments
	time_1 = get_time();

	if(argc<6) {
	  printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n",argv[0]);
	  return(-1);
	}
	for(int ac=1;ac<argc;ac++) {
		if(MATCH("-i")) {
			filename = argv[++ac];
		} else if(MATCH("-d")) {
			en_de=false;
			privateKey=atoi(argv[++ac]);
		} else if(MATCH("-e")) {
			en_de=true;
		} else if(MATCH("-p1")) {
			prime1=atoi(argv[++ac]);
		} else if(MATCH("-p2")) {
			prime2=atoi(argv[++ac]);
		} else if(MATCH("-o")) {
			outputname = argv[++ac];
		} else {
		printf("Usage: %s [-i < filename>] [-d <for decryption>] [-e <for encryption>] [-p1 <prime number>] [-p2 <prime number>] [-o <outputfilename>]\n",argv[0]);
		return(-1);
		}
	}
	time_2 = get_time();

	// Part III: read image
	printf("Reading image...\n");
	 unsigned char *image = stbi_load(filename, &width, &height, &pixelWidth,0);
	if (!image) {
		fprintf(stderr, "Couldn't load image.\n");
		return (-1);
	}

	time_3 = get_time();
//	FILE *re = fopen("out.txt","rb");

	printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);
	n_pixels = height * width;



  float test = 0;
  for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        test += image[i * width + j];
    }
  }
  test /= n_pixels;
  printf("Average of sum of pixels: %9.6f\n", test);



for(int i = 0; i<1; i++){
  for(int j = 0; j<1; j++){
    int k = i*width+j;
    int x = image[0];
    printf("DEĞER: %lu",x);
  }
}

time_4 = get_time();

	uint64_t *writeback = (uint64_t*)malloc(n_pixels * sizeof(uint64_t));

	uint64_t  *writebackdevice;


	unsigned char *image_device;
  uint64_t *readfromdevice;

	err =  cudaMalloc((void**)&writebackdevice,sizeof(uint64_t)*n_pixels);
		if (err != cudaSuccess)
		{
				fprintf(stderr, "Failed to allocate writebackdevice(error code %s)!\n", cudaGetErrorString(err));
			}


	err =  cudaMalloc((void**)&image_device,sizeof(unsigned char)*n_pixels * pixelWidth);
				  if (err != cudaSuccess)
				  {
				      fprintf(stderr, "Failed to allocate image_device (error code %s)!\n", cudaGetErrorString(err));
				      exit(EXIT_FAILURE);
				  }



          err =  cudaMalloc((void**)&readfromdevice,sizeof(uint64_t)*n_pixels);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate readfromdevice(error code %s)!\n", cudaGetErrorString(err));
              }


uint64_t mod=prime1*prime2;


time_5 = get_time();



//Encryption


if(en_de){

	time_6 = get_time();

//Calculate Keys
	publicKey = calcPub(prime1,prime2);
	printf("Public Key: %llu\n", publicKey);
	privateKey = calcPriv(prime1,prime2,publicKey);
  printf("Private Key: %llu\n", privateKey);


uint64_t binaryPublickey;
binaryPublickey = publicKey;


int remainder;
int *binary = new int[8];

int recursion = 0;


  while(binaryPublickey != 0){

    remainder = binaryPublickey %2;
    binaryPublickey = binaryPublickey / 2;
    binary[recursion] = remainder;
    recursion++;
  }



  int *binarydevice;
  err =  cudaMalloc((void**)&binarydevice,sizeof(int)*recursion);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate writebackdevice(error code %s)!\n", cudaGetErrorString(err));
      }


	time_7 = get_time();


	dim3 threads(BLOCK_SIZE,BLOCK_SIZE,1);

	dim3 grid(height/threads.x,width/threads.y);

	err =  cudaMemcpy(image_device,image,sizeof(unsigned char)*n_pixels * pixelWidth,cudaMemcpyHostToDevice);
	  if (err != cudaSuccess)
	  {
	      fprintf(stderr, "Failed to xxxxxxxxxxx image_device (error code %s)!\n", cudaGetErrorString(err));
	      exit(EXIT_FAILURE);
	  }

		err =  cudaMemcpy(writebackdevice,writeback,sizeof(uint64_t)*n_pixels,cudaMemcpyHostToDevice);
		  if (err != cudaSuccess)
		  {
		      fprintf(stderr, "Failed to copy writeback (error code %s)!\n", cudaGetErrorString(err));
		      exit(EXIT_FAILURE);
		  }

      err =  cudaMemcpy(binarydevice,binary,sizeof(int)*recursion,cudaMemcpyHostToDevice);
  		  if (err != cudaSuccess)
  		  {
  		      fprintf(stderr, "Failed to copy writeback (error code %s)!\n", cudaGetErrorString(err));
  		      exit(EXIT_FAILURE);
  		  }



	rsaEncryption<<<grid,threads>>>(height,width,image_device,writebackdevice,publicKey,mod,recursion,binarydevice);


	err =  cudaMemcpy(image,image_device,sizeof(unsigned char)*n_pixels * pixelWidth,cudaMemcpyDeviceToHost);
	  if (err != cudaSuccess)
	  {
	      fprintf(stderr, "Failed to aaaaaaaaaaaaaaaaaaaaaaaaaa image_device (error code %s)!\n", cudaGetErrorString(err));
	      exit(EXIT_FAILURE);
	  }



		err =  cudaMemcpy(writeback,writebackdevice,sizeof(uint64_t)*n_pixels,cudaMemcpyDeviceToHost);
		  if (err != cudaSuccess)
		  {
		      fprintf(stderr, "Failed to bbbbbbbbbbbbbbbbbbbb image_device (error code %s)!\n", cudaGetErrorString(err));
		      exit(EXIT_FAILURE);
		  }


time_8 = get_time();

FILE *writefile;
FILE *writefiletxt;
unsigned long long int son;
writefiletxt = fopen("output.txt","wb");
for (int i = 0; i <height ; i++) {
	for (int j = 0; j <width; j++) {
son = writeback[i*width+j];

fprintf(writefiletxt,"%llu" "%s",son," ");
}
}


time_9 = get_time();


printf("%9.6f s =>initialize Big Numbers\n", (time_5 - time_4));
printf("%9.6f s =>Started Encryption\n", (time_6 - time_5));
printf("%9.6f s =>Publish Private and Public Key\n", (time_7 - time_6));
printf("%9.6f s =>Encrypt The İmage\n", (time_8 - time_7));
printf("%9.6f s =>Write Back To Encrypted File\n", (time_9 - time_8));
printf("%9.6f s =>Total Tİme\n", (time_9 - time_0));





}else{
//decryption
double time_11,time_12;
time_11 = get_time();

uint64_t *readfrom = (uint64_t*)malloc(n_pixels * sizeof(uint64_t));

//uint64_t *readfromdevice;


	FILE *readfile = fopen("output.txt","rb");
	unsigned long long int readlong;

	for (int i = 0; i <height ; i++) {
		for (int j = 0; j <width; j++) {
	if(fscanf(readfile,"%llu",&readlong)==1){
				readfrom[i*width+j] = readlong;
	}
}
}


uint64_t binaryPrivatekey;
binaryPrivatekey = privateKey;


int remainder;
int *binaryDec = new int[8];

int recursionDec = 0;




  while(binaryPrivatekey != 0){

    remainder= binaryPrivatekey %2;
    binaryPrivatekey = binaryPrivatekey / 2;
    binaryDec[recursionDec] = remainder;
    recursionDec++;
  }

  int *binarydeviceDec;
  err =  cudaMalloc((void**)&binarydeviceDec,sizeof(int)*recursionDec);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate writebackdevice(error code %s)!\n", cudaGetErrorString(err));
      }


    err =  cudaMemcpy(image_device,image,sizeof(unsigned char)*n_pixels * pixelWidth,cudaMemcpyHostToDevice);
  	  if (err != cudaSuccess)
  	  {
  	      fprintf(stderr, "Failed to copy image_device (error code %s)!\n", cudaGetErrorString(err));
  	      exit(EXIT_FAILURE);
  	  }



  		err =  cudaMemcpy(readfromdevice,readfrom,sizeof(uint64_t)*n_pixels,cudaMemcpyHostToDevice);
  		  if (err != cudaSuccess)
  		  {
  		      fprintf(stderr, "Failed to copy readfromdevice (error code %s)!\n", cudaGetErrorString(err));
  		      exit(EXIT_FAILURE);
  		  }

        err =  cudaMemcpy(binarydeviceDec,binaryDec,sizeof(int)*recursionDec,cudaMemcpyHostToDevice);
          if (err != cudaSuccess)
          {
              fprintf(stderr, "Failed to copy readfromdevice (error code %s)!\n", cudaGetErrorString(err));
              exit(EXIT_FAILURE);
          }


        dim3 threads(BLOCK_SIZE,BLOCK_SIZE,1);

      	dim3 grid(height/threads.x,width/threads.y);

        rsaDecryption<<<grid,threads>>>(height,width,image_device,readfromdevice,privateKey,mod,recursionDec,binarydeviceDec);


        err =  cudaMemcpy(image,image_device,sizeof(unsigned char)*n_pixels * pixelWidth,cudaMemcpyDeviceToHost);
      	  if (err != cudaSuccess)
      	  {
      	      fprintf(stderr, "Failed to copy image_device (error code %s)!\n", cudaGetErrorString(err));
      	      exit(EXIT_FAILURE);
      	  }



      		err =  cudaMemcpy(readfrom,readfromdevice,sizeof(uint64_t)*n_pixels,cudaMemcpyDeviceToHost);
      		  if (err != cudaSuccess)
      		  {
      		      fprintf(stderr, "Failed to copy image_device (error code %s)!\n", cudaGetErrorString(err));
      		      exit(EXIT_FAILURE);
      		  }



time_12 = get_time();
printf("%9.6f s =>Decrpyt The İmage\n", (time_12 - time_11));


  float test = 0;
  for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        test += image[i * width + j];
    }
  }
  test /= n_pixels;
  printf("Average of sum of pixels: %9.6f\n", test);


	stbi_write_png(outputname, width, height, pixelWidth, image, 0);

}





	printf("Bitti:\n");

	return 0;
}
