import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import imageio
from scipy import ndimage

# Filtros disponibles:
# Dilatación: Agranda las áreas brillantes de la imagen.
# Erosión: Reduce las áreas brillantes, haciéndolas más pequeñas.
# Apertura: Erosiona y luego dilata, útil para eliminar pequeños ruidos.
# Cierre: Dilata y luego erosiona, útil para cerrar agujeros pequeños en áreas brillantes.
# Frontera: Resalta los bordes de los objetos en la imagen, calculando la diferencia entre la dilatación y la erosión.
# Mediana: Reduce el ruido aplicando un filtro de mediana.

# Clase principal de la interfaz gráfica
class FiltrosMorfologicosApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filtros Morfológicos")

        # Variables de imagen
        self.image_saved = None
        self.image_filtered = None

        # Labels para mostrar las imágenes
        self.lbl_img1 = tk.Label(root)
        self.lbl_img1.grid(row=0, column=0)

        self.lbl_img2 = tk.Label(root)
        self.lbl_img2.grid(row=0, column=2)

        # Botón para cargar la imagen
        self.btn_load = tk.Button(root, text="Cargar", command=self.open_image)
        self.btn_load.grid(row=1, column=0)

        # Filtros disponibles
        self.filters = {
            'Dilatacion 3x3': np.ones((3, 3), np.uint8),
            'Dilatacion 5x5': np.ones((5, 5), np.uint8),
            'Erosion 3x3': np.ones((3, 3), np.uint8),
            'Erosion 5x5': np.ones((5, 5), np.uint8),
            'Apertura 3x3': np.ones((3, 3), np.uint8),
            'Apertura 5x5': np.ones((5, 5), np.uint8),
            'Cierre 3x3': np.ones((3, 3), np.uint8),
            'Cierre 5x5': np.ones((5, 5), np.uint8),
            'Frontera 3x3': np.ones((3, 3), np.uint8),
            'Frontera 5x5': np.ones((5, 5), np.uint8),
            'Mediana 3x3': 3,
            'Mediana 5x5': 5,
            'Mediana 7x7': 7
        }

        # Menú desplegable para elegir el filtro
        self.filter_var = tk.StringVar(value="Erosion 3x3")
        self.filter_menu = tk.OptionMenu(root, self.filter_var, *self.filters.keys())
        self.filter_menu.grid(row=1, column=1)

        # Botón para aplicar el filtro
        self.btn_apply_filter = tk.Button(root, text="Filtrar", command=self.apply_filter)
        self.btn_apply_filter.grid(row=1, column=2)

        # Botón para copiar la imagen filtrada a la original
        self.btn_copy = tk.Button(root, text="Copiar", command=self.copy_image)
        self.btn_copy.grid(row=1, column=3)

        # Botón para guardar la imagen filtrada
        self.btn_save = tk.Button(root, text="Guardar", command=self.save_image)
        self.btn_save.grid(row=1, column=4)

    # Función para abrir una imagen
    def open_image(self):
        path_image = filedialog.askopenfilename()
        if path_image:
            # Abre la imagen y convierte a escala de grises
            self.image_saved = imageio.imread(path_image)
            if len(self.image_saved.shape) == 3:
                self.image_saved = cv2.cvtColor(self.image_saved, cv2.COLOR_RGB2GRAY)
            self.show_image(self.image_saved, self.lbl_img1)

    # Función para mostrar la imagen en la interfaz
    def show_image(self, img, label):
        img_resized = Image.fromarray(np.clip(img, 0, 255).astype('uint8')).resize((200, 200))
        img_tk = ImageTk.PhotoImage(image=img_resized)
        label.image = img_tk
        label.config(image=img_tk)

    # Función para aplicar el filtro seleccionado
    def apply_filter(self):
        filter_name = self.filter_var.get()
        kernel = self.filters[filter_name]

        if self.image_saved is None:
            messagebox.showerror("Error", "Cargá una imagen antes de aplicar un filtro.")
            return

        # Aplicar el filtro correspondiente
        if 'Dilatacion' in filter_name:
            self.image_filtered = self.dilate(self.image_saved, kernel)
        elif 'Erosion' in filter_name:
            self.image_filtered = self.erode(self.image_saved, kernel)
        elif 'Apertura' in filter_name:
            self.image_filtered = self.open_op(self.image_saved, kernel)
        elif 'Cierre' in filter_name:
            self.image_filtered = self.close_op(self.image_saved, kernel)
        elif 'Frontera' in filter_name:
            self.image_filtered = self.border(self.image_saved, kernel)
        elif 'Mediana' in filter_name:
            self.image_filtered = self.median_filter(self.image_saved, kernel)

        self.show_image(self.image_filtered, self.lbl_img2)

    # Función para copiar la imagen filtrada como la imagen original
    def copy_image(self):
        if self.image_filtered is not None:
            self.image_saved = self.image_filtered
            self.show_image(self.image_saved, self.lbl_img1)

    # Función para guardar la imagen filtrada
    def save_image(self):
        if self.image_filtered is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                imageio.imwrite(file_path, self.image_filtered)

    # Filtros morfológicos
    def dilate(self, img, kernel):
        return cv2.dilate(img, kernel)

    def erode(self, img, kernel):
        return cv2.erode(img, kernel)

    def open_op(self, img, kernel):
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def close_op(self, img, kernel):
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def border(self, img, kernel):
        dilated = self.dilate(img, kernel)
        eroded = self.erode(img, kernel)
        return cv2.absdiff(dilated, eroded)

    def median_filter(self, img, kernel_size):
        return cv2.medianBlur(img, kernel_size)


# Ejecución del programa
if __name__ == "__main__":
    root = tk.Tk()
    app = FiltrosMorfologicosApp(root)
    root.mainloop()