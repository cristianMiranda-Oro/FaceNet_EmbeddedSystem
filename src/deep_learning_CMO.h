/*
 * deep_learning_CMO.h
 *
 *  Created on: 23/03/2021
 *      Author: CRISTIANMIRANDA
 */

#ifndef DEEP_LEARNING_CMO_H_
#define DEEP_LEARNING_CMO_H_

#include <math.h>

typedef struct
{
    int w;        /*!< Width */
    int h;        /*!< Height */
    int c;        /*!< Channel */
    int n;        /*!< Number of filter, input and output must be 1 */
    int stride;   /*!< Step between lines */
    float *item; /*!< Data */
} matrix_NHWC;


typedef enum
{
    PADDING_VALID = 0,                   /*!< Valid padding */
    PADDING_SAME = 1,                    /*!< Same padding, from right to left, free input */
    PADDING_SAME_DONT_FREE_INPUT = 2,    /*!< Same padding, from right to left, do not free input */
    PADDING_SAME_MXNET = 3,              /*!< Same padding, from left to right */
} padding_type;

typedef enum
{
	FREE_MEMORY_Y = 1,
	FREE_MEMORY_N = 0,
} free_type;


// Description: memory free
// Input = shape pointer -> matrix_NHWC
// Output = void
// take of =
static inline void cmo_lib_free(matrix_NHWC *d)
{
    if (NULL == d)
        return;

    free(d->item);
    free(d);
    //free(((matrix_NHWC **)d)[-1]);
}

// Description: Zero-initialized and allocate space.
// for memory free should use cmo_lib_free() function
// This function (matrix_NHWC_alloc) is a succession of another top function.
// took of = https://github.com/espressif/esp-face/blob/master/lib/include/dl_lib_matrix3d.h
/*static void *cmo_lib_calloc(int cnt, int size, int align)
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
}*/


// Description: host memory for NHWC type structure
// Input = n:kernel or filter, w:width, h:high, c:channels
// Output = pointer (matrix_NHWC)
static inline matrix_NHWC *matrix_NHWC_alloc(int n, int w, int h, int c)
{

	matrix_NHWC *r = (matrix_NHWC *)malloc(sizeof(matrix_NHWC));
	float *items = (float *)malloc(n*w*h*c*sizeof(float));

	/*
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
    */

    r->w = w;
    r->h = h;
    r->c = c;
    r->n = n;
    r->stride = w * c;
    r->item = items;

    return r;
}


// Description: apply l2 normalize on all the volume
// Input = s( pointer NHWC)
// Output = void
void cmo_NHWC_l2_normalize(matrix_NHWC *s);

// Description: apply max pooling
// Input = in:input volume, f_w: window width, f_h: window High
//			stride_X: stride in horizontal, stride_Y: stride in vertical
//		    f: f=1 free memory pointer *in and other wise (f=0) the memory is not free
// Output = pointer (matrix_NHWC)
matrix_NHWC *cmo_NHWC_MaxPooling(matrix_NHWC *in, int f_w, int f_h,
								int stride_x, int stride_y, free_type f);

// Description: apply average on volumen
// Input = in:input volume, f_w: window width, f_h: window High
//			stride_X: stride in horizontal, stride_Y: stride in vertical
//		    f: f=1 free memory pointer *in and other wise (f=0) the memory is not free
// Output = pointer (matrix_NHWC)
matrix_NHWC *cmo_NHWC_AveragePooling(matrix_NHWC *in, int f_w,
                                   int f_h, int stride_x,
                                   int stride_y, free_type f);

// Description: Convolution between filters and volume.
// Input = in: input volume, filter: input kernel,
//		   bias: input bias, stride_x: horizontal window stride
//		   stride_y: vertical stride window
//		   padding: not implemented
//		   f: f=1 free memory pointer *in and other wise (f=0) the memory is not free
// Output = pointer(matriz_NHWC)
matrix_NHWC *cmo_NHWC_conv(matrix_NHWC *in, const matrix_NHWC *filter,
							const matrix_NHWC *bias, int stride_x,
                            int stride_y, padding_type padding,  free_type f);

// Description: fully connected
// Input = out: pointer to exit, in: input volume
//			filter: fully-connected's kernel, bias: bias
//		   f: f=1 free memory pointer *in and other wise (f=0) the memory is not free
// Output = not return additional pointer.
void cmo_NHWC_dense(matrix_NHWC *out, matrix_NHWC *in,
				const matrix_NHWC *filter,const  matrix_NHWC *bias, free_type f);

// Description: batch normalization on volume
// Input = s:volume to normalize, scale: scale (view description)
//			offset: offset(view description)
// Output = not return additional pointer.
void cmo_NHWC_batch_normalize(matrix_NHWC *s, const matrix_NHWC *scale,
							const matrix_NHWC *offset);

// Description: concatenate 4 volume in order.
// Input: volumes
// 		f: f=1 free memory pointer *in and other wise (f=0) the memory is not free
// Output = pointer (NWHC)
matrix_NHWC *cmo_NHWC_concat4(matrix_NHWC *in_1, matrix_NHWC *in_2,
				matrix_NHWC *in_3, matrix_NHWC *in_4 , free_type f);

// Description: concatenate 3 volume in order.
// Input: volumes
//		f: f=1 free memory pointer *in and other wise (f=0) the memory is not free
// Output = pointer (NWHC)
matrix_NHWC *cmo_NHWC_concat3(matrix_NHWC *in_1, matrix_NHWC *in_2,
				matrix_NHWC *in_3, free_type f );

// Description: concatenate 2 volume in order.
// Input: volumes
//		f: f=1 free memory pointer *in and other wise (f=0) the memory is not free
// Output = pointer (NWHC)
matrix_NHWC *cmo_NHWC_concat2(matrix_NHWC *in_1, matrix_NHWC *in_2, free_type f);

// Description: add padding to volume.
// Input: padd_L: padding left, padd_R: padding right
//		  padd_U: padding up, padd_D: padding down,
//		  imageOrigi: image to apply padding
//		  f: 1 = free memory pointer, 0 = no free memory pointer
// Output = pointer (NWHC)
matrix_NHWC *cmo_NHWC_padding(unsigned int padd_L, unsigned int padd_R, unsigned int padd_U,
		unsigned int padd_D,matrix_NHWC *imageOrigi, free_type f );

// Description: Apply Relu to layer
// Input: volume
// output: void
void cmo_NHWC_ActivationRelu(matrix_NHWC *in);


// Description: FaceNet (facial identification)
// Input: Image shape NHWC
// Output: Pointer with size of 128
matrix_NHWC * faceNetIdent(matrix_NHWC *image);


#endif /* DEEP_LEARNING_CMO_H_ */
