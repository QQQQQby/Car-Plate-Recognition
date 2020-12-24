# coding: utf-8

import tkinter as tk
from tkinter import messagebox, filedialog
import os
from PIL import Image, ImageTk

from detector import PlateDetector
from util import resized_size


class LPRGUI:
    max_image_width = 600
    max_image_height = 600

    def __init__(self):
        self.detector = PlateDetector(chinese_cnn_path='./models/chinese/79.pth',
                                      others_cnn_path='./models/others/49.pth')

        root = tk.Tk()
        root.title('Licence Plate Recognition')
        root.geometry('800x800')
        root.resizable(0, 0)

        self.plate_number_var = tk.StringVar(root, value='None')

        self.add_gap(root, 20)

        # Show image
        self.image_canvas = tk.Canvas(root, width=self.max_image_width, height=self.max_image_height, bg='#E0E0E0')
        self.image_canvas.pack()
        self.tk_image = None

        self.add_gap(root, 20)

        # Show detected plate number
        frame = tk.Frame(root)
        frame.pack()
        tk.Label(frame, text='Plate Number:', font=('Times New Roman', 20)).grid(row=0, column=0)
        tk.Label(frame, textvariable=self.plate_number_var, font=('Times New Roman', 20)).grid(row=0, column=1)

        # Load image button
        self.load_button = tk.Button(root, text="Load Image", width=16, command=self.on_load_image)
        self.load_button.pack(expand=True)

        # Detect button
        self.detect_button = tk.Button(root, text="Detect", width=16, command=self.on_detect)
        self.detect_button.pack(expand=True)
        self.detect_button['state'] = 'disabled'

        # Clear button
        self.clear_button = tk.Button(root, text="Clear", width=16, command=self.on_clear)
        self.clear_button.pack(expand=True)

        print("-------------init success-------------")
        root.mainloop()

    def on_load_image(self):
        file_path = filedialog.askopenfilename(title='Load Image',
                                               filetypes=[('Image Files', '*.jfif *.jpg *.png *.gif'),
                                                          ('All Files', '*')],
                                               initialdir=os.getcwd())

        assert os.path.exists(file_path)

        image = Image.open(file_path)
        self.draw_canvas(image=image)

        self.detector.load_img(file_path)
        self.detect_button['state'] = 'active'
        self.plate_number_var.set('None')

    def on_detect(self):
        try:
            self.detector.find_plate_location()
            self.draw_canvas(Image.fromarray(self.detector.img_after_detected[..., ::-1]))
            self.detector.split_characters()
            self.detector.classify_characters()
            self.plate_number_var.set(''.join(self.detector.result_list))
        except Exception:
            messagebox.showwarning('Error', 'Detection failed!')

    def on_clear(self):
        self.tk_image = None
        self.image_canvas.delete('all')
        self.detector.clear_img()
        self.detect_button['state'] = 'disabled'
        self.plate_number_var.set('None')

    def draw_canvas(self, image):
        self.image_canvas.delete('all')
        image = image.resize(resized_size(image.size, (self.max_image_width, self.max_image_height), mode='scale'))
        self.tk_image = ImageTk.PhotoImage(image)

        horizontal_padding, vertical_padding = 0, 0
        if image.width < self.max_image_width:
            horizontal_padding = (self.max_image_width - image.width) // 2
        else:
            vertical_padding = (self.max_image_height - image.height) // 2
        image_item = self.image_canvas.create_image(horizontal_padding, vertical_padding, anchor='nw')
        self.image_canvas.itemconfig(image_item, image=self.tk_image)

    @classmethod
    def add_gap(cls, root, height):
        tk.Frame(root, height=height).pack()


if __name__ == '__main__':
    LPRGUI()
