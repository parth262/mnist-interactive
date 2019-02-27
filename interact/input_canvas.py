from tkinter import *
from PIL import Image, ImageDraw
from api.mnist_prediction import predict, predict2


class MnistPredictor:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.master = Tk()
        self.master.title("Write a digit")
        self.canvas = Canvas(self.master, width=width, height=height, background="#000000")
        self.canvas.pack()
        self.initialize_image()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", self.key_pressed)
        mainloop()

    def initialize_image(self):
        self.image = Image.new("L", (self.width, self.height), 0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        white = '#ffffff'
        x1, y1 = (event.x - 4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        self.canvas.create_oval(x1, y1, x2, y2, fill=white, outline=white)
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.draw.ellipse([(x1, y1), (x2, y2)], white, outline=white)

    def key_pressed(self, event):
        self.canvas.delete('all')
        im = self.image.resize((28, 28))
        del self.image
        self.initialize_image()
        pxls = [[im.getpixel((x, y))/1.0 for y in range(28) for x in range(28)]]
        y_pred = predict2(pxls)
        print(y_pred)


c = MnistPredictor(140, 140)
