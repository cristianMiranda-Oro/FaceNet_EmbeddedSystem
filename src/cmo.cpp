/*
 * cmo.h
 *
 *  Created on: 23/03/2021
 *      Author: CRISTIANMIRANDA
 */



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cnnDeepC.h"
#include "deep_learning_CMO.h"


/*
static inline void cmo_lib_free(void *d)
{
    if (NULL == d)
        return;

    free(((void **)d)[-1]);
}


static void *cmo_lib_calloc(int cnt, int size, int align)
{
    int total_size = cnt * size + align + sizeof(void *);
    void *res = malloc(total_size);
    if (NULL == res)
    {
        return NULL;
    }
    bzero(res, total_size);
    void **data = (void **)res + 1;
    void **aligned;
    if (align)
        aligned = (void **)(((size_t)data + (align - 1)) & -align);
    else
        aligned = data;

    aligned[-1] = res;
    return (void *)aligned;
}


static inline matrix_NHWC *matrix_NHWC_alloc(int n, int w, int h, int c)
{
	matrix_NHWC *r = (matrix_NHWC *)cmo_lib_calloc(1, sizeof(matrix_NHWC), 0);
    if (NULL == r)
    {
        printf("internal r failed.\n");
        return NULL;
    }
    float *items = (float *)cmo_lib_calloc(n * w * h * c, sizeof(float), 0);
    if (NULL == items)
    {
        printf("matrix3d item alloc failed.\n");
        cmo_lib_free(r);
        return NULL;
    }

    r->w = w;
    r->h = h;
    r->c = c;
    r->n = n;
    r->stride = w * c;
    r->item = items;

    return r;
}
*/


void cmo_NHWC_l2_normalize(matrix_NHWC *s)
{

	float sum = 0;

	for(int i=0; i < s->w*s->h*s->n*s->c; i++)
	{
		sum = sum + s->item[i]*s->item[i];
	}


	for(int i=0; i < s->n*s->h*s->w*s->c; i++)
	{
		s->item[i] = s->item[i]/sqrt(sum);
	}

}

void cmo_NHWC_ActivationRelu(matrix_NHWC *in)
{
	for(int i = 0; i < in->w*in->c*in->h*in->n; i++)
	{
		if(in->item[i] < 0)
		{
			in->item[i] = 0;
		}
	}
}

matrix_NHWC *cmo_NHWC_MaxPooling(matrix_NHWC *in,
                                   int f_w,
                                   int f_h,
                                   int stride_x,
                                   int stride_y , free_type f)
{
	int newW = floor((in->w-f_w)/stride_x) + 1;
	int newH = floor((in->h-f_h)/stride_y) + 1;
	int newC = (in->c);

	matrix_NHWC *s = matrix_NHWC_alloc(1, newW, newH, newC);
	//s = matrix_NHWC_alloc(1, newW, newH, newC);


	int w = 0;
	int ws;

	float maxiA = -99999999;


	for(int l = 0; l < in->c; l++)
	{
		for(int k =0; k < newH; k++)
		{
			for(int p = 0; p < newW; p++)
			{
				for(int j=0; j < f_h; j++)
				{
					for(int i=0; i < f_w; i++)
					{
						w = i*in->c + in->stride*j + p*in->c*stride_x + k*in->stride*stride_y+l;
						//sum = sum + in->item[w] * filter->item[wf];
						//printf(" %.3f ", in->item[w]);

						if( in->item[w] > maxiA)
						{
							maxiA = in->item[w];
						}

					}
					//printf("**");

				}
				ws = p*s->c + k*s->stride + l;
				s->item[ws] = maxiA;
				//printf("*maximo: %i*\n" , ws);
				maxiA = -99999999;


			}
		}

	}

	if(f == FREE_MEMORY_Y)
	{
		cmo_lib_free(in);
	}

	return s;
}

matrix_NHWC *cmo_NHWC_AveragePooling(matrix_NHWC *in,
                                   int f_w,
                                   int f_h,
                                   int stride_x,
                                   int stride_y, free_type f)
{
	int newW = floor((in->w-f_w)/stride_x) + 1;
	int newH = floor((in->h-f_h)/stride_y) + 1;
	int newC = (in->c);

	matrix_NHWC *s = matrix_NHWC_alloc(1, newW, newH, newC);
	//s = matrix_NHWC_alloc(1, newW, newH, newC);

	int w = 0;
	int ws;

	float sum = 0;


	for(int l = 0; l < in->c; l++)
	{
		for(int k =0; k < newH; k++)
		{
			for(int p = 0; p < newW; p++)
			{
				for(int j=0; j < f_h; j++)
				{
					for(int i=0; i < f_w; i++)
					{
						w = i*in->c + in->stride*j + p*in->c*stride_x + k*in->stride*stride_y+l;
						//sum = sum + in->item[w] * filter->item[wf];
						//printf(" %.3f ", in->item[w]);

						sum = sum + in->item[w];


					}
					//printf("**");

				}

				ws = p*s->c + k*s->stride + l;
				s->item[ws] = sum/( f_w*f_h );
				sum = 0;
				//printf("*maximo: %i*\n" , ws);


			}
		}

	}

	if(f == FREE_MEMORY_Y)
	{
		cmo_lib_free(in);
	}

	return s;
}


matrix_NHWC *cmo_NHWC_conv(matrix_NHWC *in,
										const matrix_NHWC *filter,
										const matrix_NHWC *bias,
                                         int stride_x,
                                         int stride_y,
										 padding_type padding, free_type f)
{



	int newW = floor((in->w-filter->w)/stride_x) + 1;
	int newH = floor((in->h-filter->h)/stride_y) + 1;
	int newC = (filter->n);

	matrix_NHWC *s = matrix_NHWC_alloc(1, newW, newH, newC);
	//s = matrix_NHWC_alloc(1, newW, newH, newC);

	//printf("c : %i \n", s->c);
	//printf("h : %i \n", s->h);
	//printf("w : %i \n", s->w);
	//printf("n : %i \n", s->n);

	float sum = 0;

	int w = 0;
	int wf = 0;
	int ws = 0;

	int strideD = in->w*in->c;


	for(int n =0 ; n < filter->n; n++)
	{

		for(int l = 0; l<newH; l++)
		{
			for(int k = 0 ; k<newW ; k++)
			{
				wf = n*filter->w*filter->c*filter->h;
				for(int p = 0; p<filter->h; p++)
				{
					for(int j=0; j < filter->w; j++)
					{
						for(int i=0; i < filter->c; i++)
						{
							w = i + j*in->c + strideD*p + k*in->c*stride_x + l*strideD*stride_y ;//+ k*strideD*stride_y;
							sum = sum + in->item[w] * filter->item[wf];
							//printf(" %i ", wf);
							wf = 1 + wf;
						}
						//printf("**");

					}
				}
				ws = k*s->c + n + l*s->w*s->c;
				sum = sum + bias->item[n];
				s->item[ws] = sum;
				//printf(" %.6f ", sum);
				//printf("\n\n");
				sum = 0;
			}
		}
	}

	//free(in->item);
	//free(in);

	if(f == FREE_MEMORY_Y)
	{
		cmo_lib_free(in);
	}

	return s;

}



void cmo_NHWC_dense(matrix_NHWC *out,
								matrix_NHWC *in,
								const matrix_NHWC *filter,
								const matrix_NHWC *bias, free_type f)
{
	float sum = 0;

	for(int i = 0; i < filter->h; i++)
	{
		for(int j = 0; j < filter->w;j++)
		{
			sum = sum + in->item[j]*filter->item[j+i*filter->w];
		}
		sum = sum + bias->item[i];
		//printf("suma: %.6f ", sum);

		out->item[i] = sum;
		sum = 0;
	}

	if(f == FREE_MEMORY_Y)
	{
		cmo_lib_free(in);
	}
}

void cmo_NHWC_batch_normalize(matrix_NHWC *s,
		const matrix_NHWC *scale,
		const matrix_NHWC *offset)
{
	for(int i = 0; i < s->w*s->h*s->n; i++)
	{
		for(int c = 0; c < s->c; c++)
		{
			s->item[c + i*s->c] = s->item[c + i*s->c]*scale->item[c] + offset->item[c];
			//printf(" *%i* ", c+i*s->c);
		}
	}
}

matrix_NHWC *cmo_NHWC_concat4(matrix_NHWC *in_1, matrix_NHWC *in_2, matrix_NHWC *in_3, matrix_NHWC *in_4 , free_type f)
{
	int w = in_1->w;
	int h = in_1->h;
	int c = in_1->c + in_2->c + in_3->c + in_4->c;


	matrix_NHWC *s = matrix_NHWC_alloc(1, w, h, c);
	//s = matrix_NHWC_alloc(1, w, h, c);

	//printf(" %i ", s->c);

	//int p = 0;

	for(int j = 0; j < in_1->w*in_1->h; j++)
	{

		for(int k = 0; k < in_1->c; k++)
		{

			//p = k + j*in_2->c + j*in_1->c + j*in_2->c + j*in_4->c;
			//printf("p= %i*",p);
			s->item[ k + j*in_1->c + j*in_2->c + j*in_3->c + j*in_4->c] =in_1->item[k + j*in_1->c];
		}

		//printf("\n");
	}

		//printf("\n\n");

	for(int j = 0; j < in_2->h*in_2->w; j ++)
	{
		for(int l = 0; l < in_2->c; l++)
		{
			//p =l + (j+1)*in_1->c + j*in_2->c + j*in_3->c + j*in_4->c;
			//printf("p= %i*",p);

			s->item[l + (j+1)*in_1->c + j*in_2->c + j*in_3->c + j*in_4->c] = in_2->item[l + j*in_2->c];
		}
		//printf("\n");

	}

	for(int j = 0; j < in_3->h*in_3->w; j ++)
	{
		for(int l = 0; l < in_3->c; l++)
		{
			//p =l + (j+1)*in_1->c + (j+1)*in_2->c + j*in_3->c + j*in_4->c;
			//printf("p= %i*",p);

			s->item[l + (j+1)*in_1->c + (j+1)*in_2->c + j*in_3->c + j*in_4->c] = in_3->item[l + j*in_3->c];
		}
		//printf("\n");

	}

	for(int j = 0; j < in_4->h*in_4->w; j ++)
	{
		for(int l = 0; l < in_4->c; l++)
		{
			//p =l + (j+1)*in_1->c + (j+1)*in_2->c + (j+1)*in_3->c + j*in_4->c;
			//printf("p= %i*",p);

			s->item[l + (j+1)*in_1->c + (j+1)*in_2->c + (j+1)*in_3->c + j*in_4->c] = in_4->item[l + j*in_4->c];
		}
		//printf("\n");

	}

	if( f == FREE_MEMORY_Y)
	{
		cmo_lib_free(in_1);
		cmo_lib_free(in_2);
		cmo_lib_free(in_3);
		cmo_lib_free(in_4);

	}


	return s;

}

matrix_NHWC *cmo_NHWC_concat3(matrix_NHWC *in_1, matrix_NHWC *in_2, matrix_NHWC *in_3, free_type f )
{
	int w = in_1->w;
	int h = in_1->h;
	int c = in_1->c + in_2->c + in_3->c;


	matrix_NHWC *s = matrix_NHWC_alloc(1, w, h, c);
	//s = matrix_NHWC_alloc(1, w, h, c);



	//printf(" %i ", s->c);

	//int p = 0;

	for(int j = 0; j < in_1->w*in_1->h; j++)
	{

		for(int k = 0; k < in_1->c; k++)
		{

			//p =k + j*in_2->c + j*in_1->c;
			//printf("p= %i*",p);
			s->item[ k + j*in_1->c + j*in_2->c + j*in_3->c] =in_1->item[k + j*in_1->c];
		}

		//printf("\n");
	}

		//printf("\n\n");

	for(int j = 0; j < in_2->h*in_2->w; j ++)
	{
		for(int l = 0; l < in_2->c; l++)
		{
			//p =l + j*in_2->c;
			//printf("p= %i*",p);

			s->item[l + (j+1)*in_1->c + j*in_2->c + j*in_3->c] = in_2->item[l + j*in_2->c];
		}
		//printf("\n");

	}

	for(int j = 0; j < in_3->h*in_3->w; j ++)
	{
		for(int l = 0; l < in_3->c; l++)
		{
			//p =l + j*in_2->c;
			//printf("p= %i*",p);

			s->item[l + (j+1)*in_1->c + (j+1)*in_2->c + j*in_3->c] = in_3->item[l + j*in_3->c];
		}
		//printf("\n");

	}

	if(f == FREE_MEMORY_Y)
	{
		cmo_lib_free(in_1);
		cmo_lib_free(in_2);
		cmo_lib_free(in_3);
	}
	return s;

}

matrix_NHWC *cmo_NHWC_concat2(matrix_NHWC *in_1, matrix_NHWC *in_2, free_type f)
{
	int w = in_1->w;
	int h = in_1->h;
	int c = in_1->c + in_2->c;


	matrix_NHWC *s = matrix_NHWC_alloc(1, w, h, c);
	//s = matrix_NHWC_alloc(1, w, h, c);

	//printf(" %i ", s->c);

	//int p = 0;

	for(int j = 0; j < in_1->w*in_1->h; j++)
	{

		for(int k = 0; k < in_1->c; k++)
		{

			//p =k + j*in_2->c + j*in_1->c;
			//printf("p= %i*",p);
			s->item[ k + j*in_1->c + j*in_2->c] =in_1->item[k + j*in_1->c];
		}

		//printf("\n");
	}

		//printf("\n\n");

	for(int j = 0; j < in_2->h*in_2->w; j ++)
	{
		for(int l = 0; l < in_2->c; l++)
		{
			//p =l + j*in_2->c;
			//printf("p= %i*",p);

			s->item[l + (j+1)*in_1->c + j*in_2->c] = in_2->item[l + j*in_2->c];
		}
		//printf("\n");

	}

	if(f == FREE_MEMORY_Y)
	{
		cmo_lib_free(in_1);
		cmo_lib_free(in_2);
	}
	return s;

}


matrix_NHWC *cmo_NHWC_padding(unsigned int padd_L, unsigned int padd_R, unsigned int padd_U,
		unsigned int padd_D,matrix_NHWC *imageOrigi, free_type f )
{


	int w = imageOrigi->w + padd_L + padd_R;
	int h = imageOrigi->h + padd_U + padd_D;
	int c = imageOrigi->c;


	matrix_NHWC *s = matrix_NHWC_alloc(1, w, h, c);
	//s = matrix_NHWC_alloc(1, w, h, c);

	float * dataP = s->item; //imagen para el padding
	float * imageP = imageOrigi->item; //imagen original




	for(int ni = 0; ni < imageOrigi->n; ni++)
	{
		//Arriba
		for(int pad = 0; pad < padd_U; pad++)
		{
			for(int p = 0; p < s->w; p++)
			{
				for(int c = 0; c < s->c; c++)
				{
					dataP[c] = 0;
				}
				dataP += s->c;
			}
		}
		//*****

		for(int i = 0; i < s->h-padd_U - padd_D; i++) //2
		{
			for(int pad=0; pad < padd_L; pad++)
			{
				for(int c = 0; c < s->c; c++)
				{
					dataP[c] = 0;
				}
				dataP += s->c;

			}

			for(int j=0; j < s->w - padd_L - padd_R; j++)
			{
				for(int c = 0; c < s->c; c++)
				{
					dataP[c] = imageP[c];
				}
				dataP += s->c;
				imageP += s->c;
			}

			for(int pad=0; pad < padd_R; pad++)
			{
				for(int c = 0; c < s->c; c++)
				{
					dataP[c] = 0;
				}
				dataP += s->c;

			}

		}

		for(int pad = 0; pad < padd_D; pad++)
		{
			for(int p = 0; p < w; p++)
			{
				for(int c = 0; c < s->c; c++)
				{
					dataP[c] = 0;
				}
				dataP += s->c;
			}
		}
	}

	if(f == FREE_MEMORY_Y)
	{
		free(imageOrigi->item);
		free(imageOrigi);
	}

	return s;

}


matrix_NHWC * faceNetIdent(matrix_NHWC *image)
{

	matrix_NHWC *out_1 = NULL;
	matrix_NHWC *X_3x3 = NULL;
	matrix_NHWC *X_5x5 = NULL;
	matrix_NHWC *X_pool = NULL;
	matrix_NHWC *X_1x1 = NULL;
	matrix_NHWC *conca = NULL;
	matrix_NHWC *out_10 = NULL;
	matrix_NHWC *out_11 = NULL;

	//zero-padding
	out_1 = cmo_NHWC_padding(3, 3, 3, 3,image, FREE_MEMORY_N );

	//First Block
	out_1 = cmo_NHWC_conv(out_1, &conv1_kernel, &conv1_bias, 2, 2, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(out_1, &bn1_scale,&bn1_offset);
	cmo_NHWC_ActivationRelu(out_1);



	// Zero-Padding + MaxPool
	out_1 = cmo_NHWC_padding(1, 1, 1, 1, out_1, FREE_MEMORY_Y );
	//Stride x, Stride y -> Estan intercambiados
	out_1 =cmo_NHWC_MaxPooling(out_1, 3, 3, 2, 2, FREE_MEMORY_Y);


	// Second Block
	out_1 = cmo_NHWC_conv(out_1, &conv2_kernel, &conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(out_1, &bn2_scale,&bn2_offset);
	cmo_NHWC_ActivationRelu(out_1);


	// Zero-Padding + MAXPOOL
	out_1 = cmo_NHWC_padding(1, 1, 1, 1, out_1, FREE_MEMORY_Y);


	// third Block
	out_1 = cmo_NHWC_conv(out_1, &conv3_kernel, &conv3_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(out_1, &bn3_scale,&bn3_offset);
	cmo_NHWC_ActivationRelu(out_1);


	// Zero-Padding + MAXPOOL
	out_1 = cmo_NHWC_padding(1, 1, 1, 1, out_1, FREE_MEMORY_Y);
	//Stride x, Stride y -> Estan intercambiados
	out_1 =cmo_NHWC_MaxPooling(out_1, 3, 3, 2, 2, FREE_MEMORY_Y);



	//**********************Inception 1: a/b/c************************

	//inception_block_1a

	X_3x3 = cmo_NHWC_conv(out_1, &inception_3a_3x3_conv1_kernel, &inception_3a_3x3_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_3x3, &inception_3a_3x3_bn1_scale,&inception_3a_3x3_bn1_offset);
	cmo_NHWC_ActivationRelu(X_3x3);
	X_3x3 = cmo_NHWC_padding(1, 1, 1, 1, X_3x3, FREE_MEMORY_Y );
	X_3x3 = cmo_NHWC_conv(X_3x3, &inception_3a_3x3_conv2_kernel, &inception_3a_3x3_conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_3x3, &inception_3a_3x3_bn2_scale,&inception_3a_3x3_bn2_offset);
	cmo_NHWC_ActivationRelu(X_3x3);

	X_5x5 = cmo_NHWC_conv(out_1, &inception_3a_5x5_conv1_kernel, &inception_3a_5x5_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_5x5, &inception_3a_5x5_bn1_scale,&inception_3a_5x5_bn1_offset);
	cmo_NHWC_ActivationRelu(X_5x5);
	X_5x5 = cmo_NHWC_padding(2, 2, 2, 2, X_5x5, FREE_MEMORY_Y );
	X_5x5 = cmo_NHWC_conv(X_5x5, &inception_3a_5x5_conv2_kernel, &inception_3a_5x5_conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_5x5, &inception_3a_5x5_bn2_scale,&inception_3a_5x5_bn2_offset);
	cmo_NHWC_ActivationRelu(X_5x5);

	X_pool =cmo_NHWC_MaxPooling(out_1, 3, 3, 2, 2, FREE_MEMORY_N);
	X_pool = cmo_NHWC_conv(X_pool, &inception_3a_pool_conv_kernel, &inception_3a_pool_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_pool, &inception_3a_pool_bn_scale,&inception_3a_pool_bn_offset);
	cmo_NHWC_ActivationRelu(X_pool);
	X_pool = cmo_NHWC_padding(3, 4, 3, 4, X_pool, FREE_MEMORY_Y);

	X_1x1 = cmo_NHWC_conv(out_1, &inception_3a_1x1_conv_kernel, &inception_3a_1x1_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_1x1, &inception_3a_1x1_bn_scale,&inception_3a_1x1_bn_offset);
	cmo_NHWC_ActivationRelu(X_1x1);

	conca = cmo_NHWC_concat4(X_3x3, X_5x5, X_pool, X_1x1, FREE_MEMORY_Y);//concatenacion


	//Inception_bloc_1b

	X_3x3 = cmo_NHWC_conv(conca, &inception_3b_3x3_conv1_kernel, &inception_3b_3x3_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_3x3, &inception_3b_3x3_bn1_scale,&inception_3b_3x3_bn1_offset);
	cmo_NHWC_ActivationRelu(X_3x3);
	X_3x3 = cmo_NHWC_padding(1, 1, 1, 1, X_3x3, FREE_MEMORY_Y );
	X_3x3 = cmo_NHWC_conv(X_3x3, &inception_3b_3x3_conv2_kernel, &inception_3b_3x3_conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_3x3, &inception_3b_3x3_bn2_scale,&inception_3b_3x3_bn2_offset);
	cmo_NHWC_ActivationRelu(X_3x3);

	X_5x5 = cmo_NHWC_conv(conca, &inception_3b_5x5_conv1_kernel, &inception_3b_5x5_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_5x5, &inception_3b_5x5_bn1_scale,&inception_3b_5x5_bn1_offset);
	cmo_NHWC_ActivationRelu(X_5x5);
	X_5x5 = cmo_NHWC_padding(2, 2, 2, 2, X_5x5, FREE_MEMORY_Y);
	X_5x5 = cmo_NHWC_conv(X_5x5, &inception_3b_5x5_conv2_kernel, &inception_3b_5x5_conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_5x5, &inception_3b_5x5_bn2_scale,&inception_3b_5x5_bn2_offset);
	cmo_NHWC_ActivationRelu(X_5x5);

	X_pool =cmo_NHWC_AveragePooling(conca, 3, 3, 3, 3, FREE_MEMORY_N);
	X_pool = cmo_NHWC_conv(X_pool, &inception_3b_pool_conv_kernel, &inception_3b_pool_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_pool, &inception_3b_pool_bn_scale,&inception_3b_pool_bn_offset);
	cmo_NHWC_ActivationRelu(X_pool);
	X_pool = cmo_NHWC_padding(4, 4, 4, 4, X_pool, FREE_MEMORY_Y );

	X_1x1 = cmo_NHWC_conv(conca, &inception_3b_1x1_conv_kernel, &inception_3b_1x1_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_1x1, &inception_3b_1x1_bn_scale,&inception_3b_1x1_bn_offset);
	cmo_NHWC_ActivationRelu(X_1x1);


	conca = cmo_NHWC_concat4(X_3x3, X_5x5, X_pool, X_1x1, FREE_MEMORY_Y);


	//inception_block_1c

	X_3x3 = cmo_NHWC_conv(conca, &inception_3c_3x3_conv1_kernel, &inception_3c_3x3_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_3x3, &inception_3c_3x3_bn1_scale,&inception_3c_3x3_bn1_offset);
	cmo_NHWC_ActivationRelu(X_3x3);
	X_3x3 = cmo_NHWC_padding(1, 1, 1, 1, X_3x3, FREE_MEMORY_Y);
	X_3x3 = cmo_NHWC_conv(X_3x3, &inception_3c_3x3_conv2_kernel, &inception_3c_3x3_conv2_bias, 2, 2, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_3x3, &inception_3c_3x3_bn2_scale,&inception_3c_3x3_bn2_offset);
	cmo_NHWC_ActivationRelu(X_3x3);

	X_5x5 = cmo_NHWC_conv(conca, &inception_3c_5x5_conv1_kernel, &inception_3c_5x5_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_5x5, &inception_3c_5x5_bn1_scale,&inception_3c_5x5_bn1_offset);
	cmo_NHWC_ActivationRelu(X_5x5);
	X_5x5 = cmo_NHWC_padding(2, 2, 2, 2, X_5x5, FREE_MEMORY_Y);
	X_5x5 = cmo_NHWC_conv(X_5x5, &inception_3c_5x5_conv2_kernel, &inception_3c_5x5_conv2_bias, 2, 2, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_5x5, &inception_3c_5x5_bn2_scale,&inception_3c_5x5_bn2_offset);
	cmo_NHWC_ActivationRelu(X_5x5);

	X_pool =cmo_NHWC_MaxPooling(conca, 3, 3, 2, 2, FREE_MEMORY_Y);
	X_pool = cmo_NHWC_padding(0, 1, 0, 1, X_pool, FREE_MEMORY_Y);

	conca = cmo_NHWC_concat3(X_3x3, X_5x5, X_pool, FREE_MEMORY_Y);


	//**********************Inception 2: a/b************************


	//inception_block_2a

	X_3x3 = cmo_NHWC_conv(conca, &inception_4a_3x3_conv1_kernel, &inception_4a_3x3_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_3x3, &inception_4a_3x3_bn1_scale,&inception_4a_3x3_bn1_offset);
	cmo_NHWC_ActivationRelu(X_3x3);
	X_3x3 = cmo_NHWC_padding(1, 1, 1, 1, X_3x3, FREE_MEMORY_Y);
	X_3x3 = cmo_NHWC_conv(X_3x3, &inception_4a_3x3_conv2_kernel, &inception_4a_3x3_conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_3x3, &inception_4a_3x3_bn2_scale,&inception_4a_3x3_bn2_offset);
	cmo_NHWC_ActivationRelu(X_3x3);

	X_5x5 = cmo_NHWC_conv(conca, &inception_4a_5x5_conv1_kernel, &inception_4a_5x5_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_5x5, &inception_4a_5x5_bn1_scale,&inception_4a_5x5_bn1_offset);
	cmo_NHWC_ActivationRelu(X_5x5);
	X_5x5 = cmo_NHWC_padding(2, 2, 2, 2, X_5x5, FREE_MEMORY_Y );
	X_5x5 = cmo_NHWC_conv(X_5x5, &inception_4a_5x5_conv2_kernel, &inception_4a_5x5_conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_5x5, &inception_4a_5x5_bn2_scale,&inception_4a_5x5_bn2_offset);
	cmo_NHWC_ActivationRelu(X_5x5);

	X_pool =cmo_NHWC_AveragePooling(conca, 3, 3, 3, 3, FREE_MEMORY_N);
	X_pool = cmo_NHWC_conv(X_pool, &inception_4a_pool_conv_kernel, &inception_4a_pool_conv_bias, 1, 1, PADDING_VALID , FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_pool, &inception_4a_pool_bn_scale,&inception_4a_pool_bn_offset);
	cmo_NHWC_ActivationRelu(X_pool);
	X_pool = cmo_NHWC_padding(2, 2, 2, 2, X_pool, FREE_MEMORY_Y);

	X_1x1 = cmo_NHWC_conv(conca, &inception_4a_1x1_conv_kernel, &inception_4a_1x1_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_1x1, &inception_4a_1x1_bn_scale,&inception_4a_1x1_bn_offset);
	cmo_NHWC_ActivationRelu(X_1x1);

	conca = cmo_NHWC_concat4(X_3x3, X_5x5, X_pool, X_1x1, FREE_MEMORY_Y);


	//inception_block_2b

	X_3x3 = cmo_NHWC_conv(conca, &inception_4e_3x3_conv1_kernel, &inception_4e_3x3_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_3x3, &inception_4e_3x3_bn1_scale,&inception_4e_3x3_bn1_offset);
	cmo_NHWC_ActivationRelu(X_3x3);
	X_3x3 = cmo_NHWC_padding(1, 1, 1, 1, X_3x3, FREE_MEMORY_Y );
	X_3x3 = cmo_NHWC_conv(X_3x3, &inception_4e_3x3_conv2_kernel, &inception_4e_3x3_conv2_bias, 2, 2, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_3x3, &inception_4e_3x3_bn2_scale,&inception_4e_3x3_bn2_offset);
	cmo_NHWC_ActivationRelu(X_3x3);

	X_5x5 = cmo_NHWC_conv(conca, &inception_4e_5x5_conv1_kernel, &inception_4e_5x5_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_5x5, &inception_4e_5x5_bn1_scale,&inception_4e_5x5_bn1_offset);
	cmo_NHWC_ActivationRelu(X_5x5);
	X_5x5 = cmo_NHWC_padding(2, 2, 2, 2, X_5x5, FREE_MEMORY_Y);
	X_5x5 = cmo_NHWC_conv(X_5x5, &inception_4e_5x5_conv2_kernel, &inception_4e_5x5_conv2_bias, 2, 2, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_5x5, &inception_4e_5x5_bn2_scale,&inception_4e_5x5_bn2_offset);
	cmo_NHWC_ActivationRelu(X_5x5);

	X_pool =cmo_NHWC_MaxPooling(conca, 3, 3, 2, 2, FREE_MEMORY_Y);
	X_pool = cmo_NHWC_padding(0, 1, 0, 1, X_pool, FREE_MEMORY_Y);

	conca = cmo_NHWC_concat3(X_3x3, X_5x5, X_pool, FREE_MEMORY_Y);


	//**********************Inception 3: a/b************************

	//inception_block_3a

	X_3x3 = cmo_NHWC_conv(conca, &inception_5a_3x3_conv1_kernel, &inception_5a_3x3_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_3x3, &inception_5a_3x3_bn1_scale,&inception_5a_3x3_bn1_offset);
	cmo_NHWC_ActivationRelu(X_3x3);
	X_3x3 = cmo_NHWC_padding(1, 1, 1, 1, X_3x3, FREE_MEMORY_Y);
	X_3x3 = cmo_NHWC_conv(X_3x3, &inception_5a_3x3_conv2_kernel, &inception_5a_3x3_conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_3x3, &inception_5a_3x3_bn2_scale,&inception_5a_3x3_bn2_offset);
	cmo_NHWC_ActivationRelu(X_3x3);

	X_pool =cmo_NHWC_AveragePooling(conca, 3, 3, 3, 3, FREE_MEMORY_N);
	X_pool = cmo_NHWC_conv(X_pool, &inception_5a_pool_conv_kernel, &inception_5a_pool_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_pool, &inception_5a_pool_bn_scale,&inception_5a_pool_bn_offset);
	cmo_NHWC_ActivationRelu(X_pool);
	X_pool = cmo_NHWC_padding(1, 1, 1, 1, X_pool, FREE_MEMORY_Y);

	X_1x1 = cmo_NHWC_conv(conca, &inception_5a_1x1_conv_kernel, &inception_5a_1x1_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_1x1, &inception_5a_1x1_bn_scale,&inception_5a_1x1_bn_offset);
	cmo_NHWC_ActivationRelu(X_1x1);

	conca = cmo_NHWC_concat3(X_3x3, X_pool, X_1x1, FREE_MEMORY_Y);



	//inception_block_3b

	X_3x3 = cmo_NHWC_conv(conca, &inception_5b_3x3_conv1_kernel, &inception_5b_3x3_conv1_bias, 1, 1, PADDING_VALID, FREE_MEMORY_N);
	cmo_NHWC_batch_normalize(X_3x3, &inception_5b_3x3_bn1_scale,&inception_5b_3x3_bn1_offset);
	cmo_NHWC_ActivationRelu(X_3x3);
	X_3x3 = cmo_NHWC_padding(1, 1, 1, 1, X_3x3, FREE_MEMORY_Y);
	X_3x3 = cmo_NHWC_conv(X_3x3, &inception_5b_3x3_conv2_kernel, &inception_5b_3x3_conv2_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_3x3, &inception_5b_3x3_bn2_scale,&inception_5b_3x3_bn2_offset);
	cmo_NHWC_ActivationRelu(X_3x3);

	X_pool =cmo_NHWC_MaxPooling(conca, 3, 3, 2, 2, FREE_MEMORY_N);
	X_pool = cmo_NHWC_conv(X_pool, &inception_5b_pool_conv_kernel, &inception_5b_pool_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_pool, &inception_5b_pool_bn_scale,&inception_5b_pool_bn_offset);
	cmo_NHWC_ActivationRelu(X_pool);
	X_pool = cmo_NHWC_padding(1, 1, 1, 1, X_pool, FREE_MEMORY_Y);

	X_1x1 = cmo_NHWC_conv(conca, &inception_5b_1x1_conv_kernel, &inception_5b_1x1_conv_bias, 1, 1, PADDING_VALID, FREE_MEMORY_Y);
	cmo_NHWC_batch_normalize(X_1x1, &inception_5b_1x1_bn_scale,&inception_5b_1x1_bn_offset);
	cmo_NHWC_ActivationRelu(X_1x1);

	conca = cmo_NHWC_concat3(X_3x3, X_pool, X_1x1, FREE_MEMORY_Y);


	// Top Layer

	out_10 =cmo_NHWC_AveragePooling(conca, 3, 3, 1, 1, FREE_MEMORY_Y);
	out_11 = matrix_NHWC_alloc(1, 1, 1, 128);
	cmo_NHWC_dense(out_11, out_10, &dense_layer_kernel, &dense_layer_bias, FREE_MEMORY_Y);

	cmo_NHWC_l2_normalize(out_11);

	return out_11;


}

