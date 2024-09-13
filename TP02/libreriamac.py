
import imageio
import numpy as np
import matplotlib.pyplot as plt  # para utilizar con el comando imshow


#def Visualizar(imagen)
  #imgA=imageio.imread(imagen)
 # imgA= np.clip(imgA/255,0,1)
 # print(imgA.shape,imgA.dtype)
 # plt.imshow(imgA)
  #plt.show()
 # return imagen


  
# La representación en el espacio YIQ es práctica dado que separa la luminancia (Y) de la cromaticidad.
def RGB2YIQ(imgRgb):
    imgYiq = np.zeros(imgRgb.shape) #genera una variable yip tipo matriz de componente cero 0, y con shape le digo que sea del mismo tamaño que la original
    imgYiq[:,:,0] = np.clip( 0.229   *imgRgb[:,:,0] + 0.587   *imgRgb[:,:,1] + 0.114   *imgRgb[:,:,2],       0,1     ) #Y   con Y’ <= 1 para que no se vaya de rango
    imgYiq[:,:,1] = np.clip( 0.595716*imgRgb[:,:,0] - 0.274453*imgRgb[:,:,1] - 0.321263*imgRgb[:,:,2], -0.5957,0.5957) #I  -0.5957 < I’ < 0.5957para que no se vaya de rango
    imgYiq[:,:,2] = np.clip( 0.211456*imgRgb[:,:,0] - 0.522591*imgRgb[:,:,1] + 0.311135*imgRgb[:,:,2], -0.5226,0.5226) #Q  -0.5226 < Q’ < 0.5226 para que no se vaya de rango
    return imgYiq


# Para trabajar con Y, la iluminancia
def modifYIQ(imgYiq,a,b):
    imgYiq[:,:,0] = np.clip( a * imgYiq[:,:,0],      0,1)
    imgYiq[:,:,1] = np.clip( b * imgYiq[:,:,1],-0.5957,0.5957)
    imgYiq[:,:,2] = np.clip( b * imgYiq[:,:,2],-0.5226,0.5226)
    return imgYiq

def YIQ2RGB(imgYip):
    imgRgb = np.zeros(imgYip.shape)
    imgRgb[:,:,0] = np.clip( 1 *imgYip[:,:,0] + 0.9663*imgYip[:,:,1] + 0.6210*imgYip[:,:,2] , 0,1)
    imgRgb[:,:,1] = np.clip( 1 *imgYip[:,:,0] + -0.2721*imgYip[:,:,1] + -0.6474*imgYip[:,:,2] , 0,1)
    imgRgb[:,:,2] = np.clip( 1 *imgYip[:,:,0] + -1.1070*imgYip[:,:,1] + 1.7046*imgYip[:,:,2] , 0,1)
    return imgRgb