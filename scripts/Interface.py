from ASBclassdef import *
from ASB import stabilizesq
class MatrixInterface(Sandpile):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.matrix = np.random.randint(0,1, (n,n), dtype = 'uint8')
        self.colormap = {0: "white", 1: "blue", 2: "purple", 3: "green"}
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=800, height=800)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_grid()

        self.root.mainloop()
    
    
    def draw_grid(self):
        self.cell_size = 800 // self.n

        for i in range(self.n):
            for j in range(self.n):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = (i + 1) * self.cell_size
                y2 = (j + 1) * self.cell_size

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colormap[(self.matrix[i][j])])

    def on_click(self, event):
        
        i = event.x // self.cell_size
        j = event.y // self.cell_size

        self.matrix[i][j] += 1
        self.matrix = stabilizesq(self.matrix)

        for i in range(self.n):
            for j in range(self.n):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = (i + 1) * self.cell_size
                y2 = (j + 1) * self.cell_size

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colormap[(self.matrix[i][j])])

interface = MatrixInterface(10)


