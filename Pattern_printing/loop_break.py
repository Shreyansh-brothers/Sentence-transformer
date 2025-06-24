# import tkinter as tk
# import random
# def generate_random_color():
#     color = f"#{random.randint(0, 0xFFFFFF):06x}"
#     label.config(text=color, bg=color)
# root = tk.Tk()
# root.title("Color Generator")
# label = tk.Label(root, text="", font=("Arial", 24), width=15, height=5)
# label.pack()
# tk.Button(root, text="Generate Color", command=generate_random_color).pack()
# root.mainloop()

# import turtle
# import random
# screen = turtle.Screen()
# screen.title("Turtle Race")
# colors = ["red", "blue", "green", "yellow", "purple"]
# turtles = []
# for i, color in enumerate(colors):
#     t = turtle.Turtle(shape="turtle")
#     t.color(color)
#     t.penup()
#     t.goto(-200, 100 - i * 40)
#     turtles.append(t)
# while True:
#     for t in turtles:
#         t.forward(random.randint(1, 5))
#         if t.xcor() >= 200:
#             winner = t.color()[0]
#             turtle.write(f"{winner.capitalize()} wins!", align="center", font=("Arial", 16, "bold"))
#             screen.mainloop()

import qrcode
data = input("Enter the data or URL to encode in the QR code: ")
fill_color = input("Enter the fill color (e.g., 'black'): ") or 'black'
back_color = input("Enter the background color (e.g., 'white'): ") or 'white'
file_name = input("Enter the file name to save the QR code (without extension): ") or "qrcode"
file_path = f"{file_name}.png"
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4
)
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image(fill_color=fill_color, back_color=back_color)
img.save(file_path)
print(f"QR code generated and saved as {file_path}")