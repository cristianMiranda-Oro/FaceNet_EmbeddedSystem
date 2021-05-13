# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:20:38 2020
@author: CRISTIANMIRANDA
"""

####VERSION ACTUALIZDA v1.4#################################

import numpy as np
import h5py
import math  

import xml.etree.ElementTree as ET

tree = ET.parse("../tutoOpencv/dataSetMine/modelo2.xml")
root = tree.getroot()


rutaFileCnn = "../tutoOpencv/dataSetMine/mnist/cnnDeepC.h"

#ruta ="../tutoOpencv/dataSetMine/pesosESP32DeepLearning.h5"
#ruta ="../tutoOpencv/dataSetMine/toESP32.h5"
#ruta ="../tutoOpencv/dataSetMine/toESP32.h5"
#ruta ="../tutoOpencv/dataSetMine/ESP32Normalization2.h5"
#ruta ="../tutoOpencv/dataSetMine/mnist/deepC.h5"
ruta ="../tutoOpencv/datasetProyectoGrado/resize96x96/FaceNetRe.h5"

#ruta ="../tutoOpencv/datasetProyectoGrado/resize96x96/FaceNetRe.h5"
#ruta ="../tutoOpencv/dataSetMine/ESP32Debugging2.h5"

keys = []
file=h5py.File(ruta, 'r')
file.visit(keys.append)

txt = ""

txtMain = ""#Para el main.c




txt = """#pragma once 
#include \"dl_lib_matrix3d.h\"
#include \"dl_lib_matrix3dq.h\"\n    
"""

txtMain = """#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "cnn.h"
#include "input.h"
    """
txtMain = txtMain+"\n\nvoid test(void *arg)\n{\n"
txtMain = txtMain+"\tdl_matrix3d_t *"+root[0][0].text+" = dl_matrix3d_alloc(1,"+root[0][0].attrib["w"]+","+root[0][0].attrib["h"]+",1);\n\n"
txtMain = txtMain + "\twhile(1)\n\t{\n"

"""
for child in root:
    print(child.tag, child.attrib)

print(root[0][1].attrib)

root[0][1].keys()
"""

ww = root[0][0].attrib["w"]
hh = root[0][0].attrib["h"]



#&conv2d_kernel, &conv2d_bias, 1, 1, PADDING_SAME);
contador = 0
contadorMain = 1#Para el main.c
entrada = ""
salida = ""

#PARA EL MAIN#
for layer in root:
    for elemen in layer:
        
        try:
            
            if elemen.tag.find("input") >= 0:
                entrada = elemen.text
                salida = "o1_"+str(contadorMain)
                

                
            if elemen.tag.find("output") >= 0:
                salida = elemen.text
                
                
                #txtMain = txtMain
            
            if elemen.tag.find("conv") >= 0:
                txtMain = txtMain + "\n\t\tdl_matrix3d_t *"+salida+" = dl_matrix3dff_conv_common"
                txtMain = txtMain + "("+entrada+", &"+keys[contador]+"_kernel"+", &"+keys[contador]+"_bias, "+elemen.attrib["stride_X"]+","+elemen.attrib["stride_Y"]+", "+elemen.attrib["padding"]+");\n"
                
                entrada = salida
                contadorMain = contadorMain + 1
                salida = "o1_"+str(contadorMain)
                
            if elemen.tag.find("padding") >= 0:
                
                entrada = salida
                contadorMain = contadorMain + 1
                salida = "o1_"+str(contadorMain)
            
            if elemen.tag.find("pooling") >= 0:
                txtMain = txtMain + "\t\tdl_matrix3d_t *"+salida+" = dl_matrix3d_pooling"+"("+entrada+", "
                txtMain = txtMain + elemen.attrib["w"]+", "+elemen.attrib["h"]+", "+elemen.attrib["stride_X"]+", "+elemen.attrib["stride_Y"]+", "+elemen.attrib["padding"]+", "+elemen.attrib["type"]+");\n"
                
                entrada = salida
                contadorMain = contadorMain + 1
                salida = "o1_"+str(contadorMain)
            
            if elemen.tag.find("relu") >= 0:
                entrada = "o1_"+str(contadorMain-1)                                
                txtMain = txtMain + "\t\tdl_matrix3d_relu("+entrada+");\n\n"                
                salida = "o1_"+str(contadorMain)
                
            if elemen.tag.find("dense") >= 0:
                txtMain = txtMain + "\t\tdl_matrix3d_t *"+salida+" = dl_matrix3d_alloc(1,1,1,"+keys[contador]+"_kernel.h);\n"
                txtMain = txtMain + "\t\tdl_matrix3dff_fc_with_bias("+salida+", "+entrada+", &"+keys[contador]+"_kernel, &"+keys[contador]+"_bias);\n"
                
                entrada = salida
                contadorMain = contadorMain + 1
                salida = "o1_"+str(contadorMain)
                
        except:
            print("fuera de rango")
    
    contador = contador + 4

txtMain = txtMain + "\t\tvTaskDelay(100);\n\t}\n}\n\nvoid app_main()\n{\n\t"
txtMain = txtMain + 'xTaskCreatePinnedToCore(&test, "test", 4096, NULL, 5, NULL, 0);\n}\n'

f= open("../tutoOpencv/dataSetMine/mainPadding.c","w+")   
f.write(txtMain)   
f.close()
    
##############

for contador in range(0,len(keys)):
    
    #PARA EL BATCH NORMALIZATION
    
    if (keys[contador].count("bn") == 2) and (keys[contador].count("beta") == 1) :
        
        beta = file[keys[contador]].value
        gamma = file[keys[contador+1]].value
        mean = file[keys[contador+2]].value
        variance = file[keys[contador+3]].value
        epsilon = 0.00001
        
        scale = gamma / np.sqrt( variance + epsilon )
        offset = beta - (mean*gamma)/np.sqrt( variance + epsilon )
        
        p = 0
        txt = txt + "const static fptp_t "+ keys[contador-2]+"_scale_item_array[] = { \n"
        for i in range(scale.shape[0]):
                txt = txt+"%.6ff, " % scale[i] 
                p = p + 1
                if p == 8:
                    txt = txt + "\n"
                    p = 0
                    
        txt = txt + "};\n\nconst static dl_matrix3d_t "+keys[contador-2]+"_scale = {\n"
        txt = txt+".w=1,\n.h=1, \n"
        txt = txt+".c = "+str(scale.shape[0])+",\n"
        txt = txt+".n = 1,\n.stride = "+str(scale.shape[0])+",\n"
        txt = txt+".item = (fptp_t *)(&"+keys[contador-2]+"_scale_item_array[0])\n};\n\n"
        
        p = 0
        txt = txt + "const static fptp_t "+ keys[contador-2]+"_offset_item_array[] = { \n"
        
        for i in range(offset.shape[0]):
                txt = txt+"%.6ff, " % offset[i] 
                p = p + 1
                if p == 8:
                    txt = txt + "\n"
                    p = 0
        
        txt = txt + "};\n\nconst static dl_matrix3d_t "+keys[contador-2]+"_offset = {\n"
        txt = txt+".w=1,\n.h=1, \n"
        txt = txt+".c = "+str(offset.shape[0])+",\n"
        txt = txt+".n = 1,\n.stride = "+str(offset.shape[0])+",\n"
        txt = txt+".item = (fptp_t *)(&"+keys[contador-2]+"_offset_item_array[0])\n};\n\n"
    
    
    #PARA EL DENSE
    if keys[contador].count("dense") == 2:
        
        if keys[contador].count("bias") == 1:
            p=0
            txt = txt + "const static fptp_t "+ keys[contador-2]+"_bias_item_array[] = { \n"
            
            bias = file[keys[contador]].value
        
            for i in range(bias.shape[0]):
                txt = txt+"%.6ff, " % bias[i] 
                p = p + 1
                if p == 8:
                    txt = txt + "\n"
                    p = 0
            
            txt = txt + "};\n\nconst static dl_matrix3d_t "+keys[contador-2]+"_bias = {\n"
            txt = txt+".w=1,\n.h=1, \n"
            txt = txt+".c = "+str(bias.shape[0])+",\n"
            txt = txt+".n = 1,\n.stride = "+str(bias.shape[0])+",\n"
            txt = txt+".item = (fptp_t *)(&"+keys[contador-2]+"_bias_item_array[0])\n};\n\n"
            
        
        if keys[contador].count("kernel") == 1:
            p=0
            txt = txt + "const static fptp_t "+ keys[contador-3]+"_kernel_item_array[] = { \n"
            kernel = file[keys[contador]].value
            for f in range(kernel.shape[1]):
                for h in range(kernel.shape[0]):
                    txt = txt+"%.6ff, " % kernel[h,f]
                    p = p + 1
                    if p == 8:
                        txt = txt + "\n"
                        p=0

            txt = txt + "}; \n\nconst static dl_matrix3d_t "+keys[contador-3]+"_kernel = {\n"+".w=none"+str( 0 )+",\n.h="+str(kernel.shape[1])+","
            txt = txt + "\n.c=1,\n.n=1,"
            txt = txt + "\n.stride = none"+ str(0)+","
            txt = txt + "\n.item = (fptp_t *)(&"+keys[contador-3]+"_kernel_item_array[0])\n};\n"

    
    #PARA EL BIAS
    if (keys[contador].count("conv") == 2 ) and (keys[contador].count("bias") == 1):
            
        p=0
        txt = txt + "const static fptp_t "+ keys[contador-2]+"_bias_item_array[] = { \n"
        
        bias = file[keys[contador]].value
        
        for i in range(bias.shape[0]):
            txt = txt+"%.6ff, " % bias[i] 
            p = p + 1
            if p == 8:
                txt = txt + "\n"
                p = 0
        
        txt = txt + "};\n\nconst static dl_matrix3d_t "+keys[contador-2]+"_bias = {\n"
        txt = txt+".w=1,\n.h=1, \n"
        txt = txt+".c = "+str(bias.shape[0])+",\n"
        txt = txt+".n = 1,\n.stride = "+str(bias.shape[0])+",\n"
        txt = txt+".item = (fptp_t *)(&"+keys[contador-2]+"_bias_item_array[0])\n};\n\n"
    
   
    #PARA EL KERNEL
    if (keys[contador].count("conv") == 2 ) and (keys[contador].count("kernel") == 1):
        
        p = 0
        txt = txt + "const static fptp_t "+keys[contador-3]+"_kernel_item_array[] = {\n"
        kernel = file[keys[contador]].value
        
        for f in range(kernel.shape[3]):
            for h in range(kernel.shape[0]):
                for w in range(kernel.shape[1]):
                    for c in range(kernel.shape[2]):
                        txt = txt+"%.6ff, " % kernel[h,w,c,f]
                        p = p + 1
                        if p == 8:
                            txt = txt + "\n"
                            p = 0
        
        txt = txt + "}; \n\nconst static dl_matrix3d_t "+keys[contador-3]+"_kernel = {\n" +".w="+str(kernel.shape[1])+",\n.h="+str(kernel.shape[0])+","
        txt = txt + "\n.c="+str(kernel.shape[2])+",\n.n="+str(kernel.shape[3])+","
        txt = txt + "\n.stride = "+ str(kernel.shape[1]*kernel.shape[2])+","
        txt = txt + "\n.item = (fptp_t *)(&"+keys[contador-3]+"_kernel_item_array[0])\n};\n"
    
    
    #contador = contador + 1
    """try:
        if (keys[contador+4].find("conv2d") >= 0) or (keys[contador+4].find("dense") >= 0):
            #contador = contador + 4
        else:
            #bandera = -1
    except:
        print("indice fuera del array")
        #bandera = -1"""


f= open(rutaFileCnn,"w+")   
f.write(txt)
f.close()


#######################################################
