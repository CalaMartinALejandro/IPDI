import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

from scipy.signal import convolve2d

# Variable global para almacenar la imagen procesada
imagen_procesada = None

# Función para convertir RGB a YIQ
def RGB2YIQ(image):
    transformation_matrix = np.array([[0.299, 0.587, 0.114],
                                      [0.596, -0.275, -0.321],
                                      [0.212, -0.523, 0.311]])
    return np.dot(image, transformation_matrix.T)

# Función para cargar imagen usando PIL
def cargar_imagen():
    global imgA, imgYiq, image_luminance_255
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    
    if filepath:
        # Cargar la imagen con PIL y convertir a RGB
        imgA_pil = Image.open(filepath).convert('RGB')
        imgA = np.array(imgA_pil)  # Convertir imagen a un arreglo numpy
        
        # Normalizar la imagen (dividir por 255)
        imgA_normalizada = np.clip(imgA / 255.0, 0., 1.)
        
        # Convertir a espacio de color YIQ
        imgYiq = RGB2YIQ(imgA_normalizada)
        image_luminance = imgYiq[:, :, 0]  # Extraer el canal Y (luminancia)
        
        # Escalar la luminancia de vuelta a 0-255 para visualizar
        image_luminance_255 = np.clip(image_luminance * 255, 0, 255).astype('uint8')
        
        # Mostrar la imagen en escala de grises (canal Y)
        mostrar_imagen(image_luminance_255, 'Imagen A (YIQ Luminancia)')

# Función para mostrar imagen en ventana
def mostrar_imagen(imagen, titulo):
    imagen_pil = Image.fromarray(imagen)  # Convertir a formato PIL
    imagen_tk = ImageTk.PhotoImage(imagen_pil)
    
    ventana_imagen = tk.Toplevel()
    ventana_imagen.title(titulo)
    label_imagen = tk.Label(ventana_imagen, image=imagen_tk)
    label_imagen.image = imagen_tk
    label_imagen.pack()

# Función para procesar imagen según la opción seleccionada
def procesar_imagenA():
    opcion = lista_desplegable_A.get()
    procesar_imagen_general(image_luminance_255, opcion, 'A')

def procesar_imagenB():
    opcion = lista_desplegable_B.get()
    procesar_imagen_general(image_luminance_255, opcion, 'B')
    
# Función para procesar imagen según la opción seleccionada

def procesar_imagen_general(imagen, opcion, nombre):

    global imagen_procesada  # para guardar imagen procesada 

    # Diccionario que mapea cada opción a su respectiva función
    funciones_procesamiento = {
        "PasaBajos llano 3x3": Pasa_llano_3x3,
        "PasaBajos llano 5x5": Pasa_llano_5x5,
        "PasaBajos llano 7x7": Pasa_llano_7x7,
        "Bartlett 3x3": lambda img: aplicar_filtro_convolucion(img, bartlett(3)),
        "Bartlett 5x5": lambda img: aplicar_filtro_convolucion(img, bartlett(5)),
        "Bartlett 7x7": lambda img: aplicar_filtro_convolucion(img, bartlett(7)),
        "Gaussiano 5x5": lambda img: aplicar_filtro_convolucion(img, gauss(5, 1.0)),
        "Pascal 3x3": lambda img: aplicar_filtro_convolucion(img, pascal(3)),        
        "PasaAltos Matriz": filtro_pasaalto,
        "Pasaaltos Laplaciano v4 3x3": lambda img: aplicar_filtro_convolucion(img, laplace(_type=4,normalize=False)),
        "Pasaaltos Laplaciano v8 3x3": lambda img: aplicar_filtro_convolucion(img, laplace(_type=8,normalize=False)),   
        "Identidad 3x3": lambda img: aplicar_filtro_convolucion(img, identity((3, 3))),
        "Dog 5x5": lambda img: aplicar_filtro_convolucion(img, dog(5, fs=1, cs=2)),
        "Binarización": binarizacion_imagen,
        "Sobel 0 N": lambda img: aplicar_filtro_convolucion(img, sobel_kernel('0 N')),
        "Sobel 1": lambda img: aplicar_filtro_convolucion(img, sobel_kernel('1')),
        "Sobel 2 O": lambda img: aplicar_filtro_convolucion(img, sobel_kernel('2 O')),
        "Sobel 3": lambda img: aplicar_filtro_convolucion(img, sobel_kernel('3')),
        "Sobel 4 S": lambda img: aplicar_filtro_convolucion(img, sobel_kernel('4 S')),
        "Sobel 5": lambda img: aplicar_filtro_convolucion(img, sobel_kernel('5')),
        "Sobel 6 E": lambda img: aplicar_filtro_convolucion(img, sobel_kernel('6 E')),
        "Sobel 7": lambda img: aplicar_filtro_convolucion(img, sobel_kernel('7')),
        # Aquí se añaden nuevas funciones de procesamiento
      

    }
        # Verificar si la opción seleccionada está en el diccionario
    if opcion in funciones_procesamiento:
        imagen_procesada = funciones_procesamiento[opcion](imagen)  # Aplicar filtro
        # Normalizar la imagen procesada para que esté en el rango [0, 255]
        imagen_procesada = np.clip(imagen_procesada, 0, 255)  # Limitar valores fuera del rango
        imagen_procesada = imagen_procesada.astype(np.uint8)   # Convertir a entero sin signo

        mostrar_imagen(imagen_procesada, f"Imagen {nombre} Procesada - {opcion}")
    else:
        messagebox.showwarning("Advertencia", f"Proceso {opcion} no implementado")

def aplicar_filtro_convolucion(image,kernel):
    convolved = np.zeros((np.array(image.shape)-np.array(kernel.shape)+1))
    for x in range(convolved.shape[0]):
        for y in range(convolved.shape[1]):
            convolved[x, y]=(image[x:x+kernel.shape[0],y:y+kernel.shape[1]]*kernel).sum()
    return convolved

# filtro pasa bajos (promedio) "llano 3x3", valores del kernel todos 1 divido a 9
def Pasa_llano_3x3(imagen):
    kernel = np.ones((3, 3), np.float32) / 9
    return aplicar_filtro_convolucion(imagen, kernel)
# filtro pasa bajos (promedio) "llano 5x5", valores del kernel todos 1
def Pasa_llano_5x5(imagen):    
    kernel = np.ones((5, 5), np.float32) /25
    return aplicar_filtro_convolucion(imagen, kernel)
# filtro pasa bajos (promedio) "llano 7x7", valores del kernel todos 1
def Pasa_llano_7x7(imagen):    
    kernel = np.ones((7, 7), np.float32) /49
    return aplicar_filtro_convolucion(imagen, kernel)


# Bartlett 3x3 -  5x5 -  7x7
def bartlett(s):
    a = (s+1)//2-np.abs(np.arange(s)-s//2)
    k = np.outer(a,a.T)
    return k / k.sum()     # 'k.sumes' la normalizacion


def gauss(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    gausseano = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gausseano / gausseano.sum()  #normalizar kernell


    # Restar las dos gaussianas para obtener el filtro DoG
def dog(size,fs=1,cs=2):
    return gauss(size,fs)-gauss(size,cs)

def pascal(s):
    def pascal_triangle(steps,last_layer = np.array([1])):
        if steps==1:
            return last_layer
        next_layer = np.array([1,*(last_layer[:-1]+last_layer[1:]),1])
        return pascal_triangle(steps-1,next_layer)
    a = pascal_triangle(s)
    k = np.outer(a,a.T)
    return k / k.sum()

# Filtro pasa altos (detectar bordes)
def filtro_pasaalto(imagen):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return aplicar_filtro_convolucion(imagen, kernel)

# Aplicar el filtro Laplaciano con tipo 4 o 8 con normalización (true) ej: tipo4 "def laplace(_type=4,normalize=False):"

def laplace(_type,normalize):
    if _type==4:
        kernel =  np.array([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]])
    if _type==8:
        kernel =  np.array([[-1.,-1.,-1.],[-1.,8.,-1.],[-1.,-1.,-1.]])
    if normalize:
        kernel /= np.sum(np.abs(kernel))
    return kernel


#    Kernel identidad (Identity kernel): Es un kernel que no altera la imagen. El centro es 1, y el resto de los valores son 0.
#    Genera un kernel de identidad del tamaño s.
#    - s: Tupla (alto, ancho) que define el tamaño del kernel.
#    Retorna:  Kernel identidad de tamaño s.
def identity(s):
    kernel = np.zeros(s)                                               #**  preguntar porque no no sale
    kernel[s[0]//2,s[1]//2] = 1. 
    return kernel

# función para binarización
def binarizacion_imagen(imagen):
    threshold = 128  # Umbral para binarización
    return np.where(imagen > threshold, 255, 0).astype('uint8')  # Imagen binarizada

def sobel_kernel(direction):
    SOBEL = {
        '0 N': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        '1': np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
        '2 O': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        '3': np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
        '4 S': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        '5': np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
        '6 E': np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        '7': np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    }
    return SOBEL.get(direction, None)




def guardar_imagen():
    global imagen_procesada
    if imagen_procesada is None:
        messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar.")
        return
    
    filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
    if filepath:
        imagen_pil = Image.fromarray(imagen_procesada.astype('uint8'))  # Convertir la imagen procesada a formato PIL
        imagen_pil.save(filepath)
        messagebox.showinfo("Guardar Imagen", f"Imagen guardada en {filepath}")


# Interfaz Gráfica
root = tk.Tk()
root.title("Procesador de Imágenes")

# Botón para cargar imagen
btn_cargar = tk.Button(root, text="Cargar Imagen A", command=cargar_imagen)
btn_cargar.grid(row=0, column=0, padx=10, pady=10)
#*************************************************************************************************************************
# Lista desplegable para opciones de procesamiento de Imagen A
lista_desplegable_A = ttk.Combobox(root, values=["PasaBajos llano 3x3",
                                                 "PasaBajos llano 5x5",
                                                 "PasaBajos llano 7x7",
                                                 "Bartlett 3x3",
                                                 "Bartlett 5x5",
                                                 "Bartlett 7x7",
                                                 "Gaussiano 5x5",
                                                 "Pascal 3x3",
                                                 "Filtro PasaAltos",
                                                 "Pasaaltos Laplaciano v4 3x3",
                                                 "Pasaaltos Laplaciano v8 3x3",
                                                 "Identidad 3x3",
                                                 "Dog 5x5",
                                                 "Binarización",
                                                 "Sobel 0 N",   
                                                 "Sobel 1",
                                                 "Sobel 2 O",
                                                 "Sobel 3",
                                                 "Sobel 4 S",
                                                 "Sobel 5",
                                                 "Sobel 6 E",
                                                 "Sobel 7"
                                                 
                                                  # Agregar aquí
                                                 
                                                 ], state="readonly")
lista_desplegable_A.grid(row=1, column=0, padx=10, pady=10)
lista_desplegable_A.set("Seleccione una opción para A")

# Botón para procesar Imagen A
btn_procesar_A = tk.Button(root, text="Procesar Imagen A", command=procesar_imagenA)
btn_procesar_A.grid(row=2, column=0, padx=10, pady=10)

# Lista desplegable para opciones de procesamiento de Imagen B
lista_desplegable_B = ttk.Combobox(root, values=["PasaBajos llano 3x3",
                                                 "PasaBajos llano 5x5",
                                                 "PasaBajos llano 7x7",
                                                 "Bartlett 3x3",
                                                 "Bartlett 5x5",
                                                 "Bartlett 7x7",
                                                 "Gaussiano 5x5",
                                                 "Pascal 3x3",
                                                 "Filtro PasaAltos",
                                                 "Pasaaltos Laplaciano v4 3x3",
                                                 "Pasaaltos Laplaciano v8 3x3",
                                                 "Identidad 3x3",
                                                 "Dog 5x5",
                                                 "Binarización",
                                                 "Sobel 0 N",   
                                                 "Sobel 1",
                                                 "Sobel 2 O",
                                                 "Sobel 3",
                                                 "Sobel 4 S",
                                                 "Sobel 5",
                                                 "Sobel 6 E",
                                                 "Sobel 7"
                                                 
                                                  # Agregar aquí
                                                 
                                                 ], state="readonly")
lista_desplegable_B.grid(row=1, column=1, padx=10, pady=10)
lista_desplegable_B.set("Seleccione una opción para B")

# Botón para procesar Imagen B
btn_procesar_B = tk.Button(root, text="Procesar Imagen B", command=procesar_imagenB)
btn_procesar_B.grid(row=2, column=1, padx=10, pady=10)
#*************************************************************************************************************************
# Botón para guardar la imagen procesada
btn_guardar = tk.Button(root, text="Guardar Imagen Procesada", command=guardar_imagen)
btn_guardar.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Iniciar la interfaz gráfica
root.mainloop()
