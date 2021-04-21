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
    int w;        /*width */
    int h;        /*height */
    int c;        /*channel */
    int n;        /*filters*/
    int stride;   /*w*c*/
    float *item; /*data*/
} matrix_NHWC;


typedef enum
{
    PADDING_VALID = 0,                
    PADDING_SAME = 1,   
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

}


// Description: host memory for NHWC type structure
// Input = n:kernel or filter, w:width, h:high, c:channels
// Output = pointer (matrix_NHWC)
static inline matrix_NHWC *matrix_NHWC_alloc(int n, int w, int h, int c)
{

	matrix_NHWC *r = (matrix_NHWC *)malloc(sizeof(matrix_NHWC));
	float *items = (float *)malloc(n*w*h*c*sizeof(float));

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
