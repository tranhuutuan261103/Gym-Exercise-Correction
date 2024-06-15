import tkinter as tk
import cv2
from PIL import Image, ImageTk
from components.Home import Home
from components.History import History
from components.History2 import History2

button_dict = {}

# Function to update button appearance
def update_button_appearance(selected_button):
    for button, btn_widget in button_dict.items():
        if button == selected_button:
            load_image = tk.PhotoImage(file=f'./desktop_app/assets/sidebar/buttons/{button.lower()}_selected.png')
            btn_widget.config(image=load_image)
            btn_widget.image = load_image
        else:
            load_image = tk.PhotoImage(file=f'./desktop_app/assets/sidebar/buttons/{button.lower()}.png')
            btn_widget.config(image=load_image)
            btn_widget.image = load_image

app = tk.Tk()

app.title("My Desktop App")
app.geometry("1440x775")

camera = cv2.VideoCapture(0)

# Function to switch tags
def switch_tag(tag):
    # Delete previous content in the main frame
    for widget in main_content.winfo_children():
        widget.destroy()
    
    # Insert new content based on the selected tag
    if tag == "Home":
        home = Home(main_content, app, camera)
        home.pack()
    elif tag == "History":
        history = History2(main_content, app)
        history.pack(fill="both", expand=True)
    elif tag == "Profile":
        label = tk.Label(main_content, text="Profile Page", bg="white")
        label.pack()

    update_button_appearance(tag)

# Sidebar
sidebar = tk.Frame(app, width=120, bg="#3C2937", height=725)
sidebar.pack(side="left", fill="y", expand=False)

avatar = tk.PhotoImage(file="./desktop_app/assets/sidebar/avatar.png")
avatar_label = tk.Label(sidebar, image=avatar, bg="#3C2937")
avatar_label.pack(pady=30, side="top")

# Sidebar buttons
buttons = ["Home", "History", "Profile"]
for button in buttons:
    loadimage = tk.PhotoImage(file=f'./desktop_app/assets/sidebar/buttons/{button.lower()}.png')
    btn = tk.Button(sidebar, image=loadimage, bg="#3C2937", activebackground='#3C2937', bd=0,
                    command=lambda b=button: switch_tag(b))
    btn.image = loadimage  # Prevent image from being garbage collected
    btn.pack(pady=30, padx=20, side="top", anchor="w")

    # Add the button to the dictionary
    button_dict[button] = btn

# Main content
main_content = tk.Frame(app, width=1440 - 120, height=775)
main_content.pack(side="left", fill="both")

switch_tag("Home")

app.mainloop()