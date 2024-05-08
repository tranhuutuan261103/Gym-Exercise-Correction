import tkinter as tk
from PIL import Image, ImageTk
from components.Home import Home
from components.History import History

class AppMain(tk.Toplevel):
    def __init__(self, camera, on_dashboard_close=None):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", on_dashboard_close)
        self.title("My Desktop App")
        self.geometry("1440x775")
        self.configure(bg='white')
        self.resizable(False, False)

        self.camera = camera
        self.button_dict = {}
        self.setup_ui()

    def setup_ui(self):
        # Sidebar
        self.sidebar = tk.Frame(self, width=120, bg="#3C2937", height=725)
        self.sidebar.pack(side="left", fill="y", expand=False)

        avatar = tk.PhotoImage(file="./desktop_app/assets/sidebar/avatar.png")
        self.avatar_label = tk.Label(self.sidebar, image=avatar, bg="#3C2937")
        self.avatar_label.image = avatar  # Keep a reference!
        self.avatar_label.pack(pady=30, side="top")

        # Sidebar buttons
        buttons = ["Home", "History", "Profile"]
        for button in buttons:
            loadimage = tk.PhotoImage(file=f'./desktop_app/assets/sidebar/buttons/{button.lower()}.png')
            btn = tk.Button(self.sidebar, image=loadimage, bg="#3C2937", activebackground='#3C2937', bd=0,
                            command=lambda b=button: self.switch_tag(b))
            btn.image = loadimage  # Prevent image from being garbage collected
            btn.pack(pady=30, padx=20, side="top", anchor="w")

            # Add the button to the dictionary
            self.button_dict[button] = btn

        # Main content
        self.main_content = tk.Frame(self, width=1440 - 120, height=775)
        self.main_content.pack(side="left", fill="both")

        self.switch_tag("Home")

    def switch_tag(self, tag):
        # Delete previous content in the main frame
        for widget in self.main_content.winfo_children():
            widget.destroy()

        # Insert new content based on the selected tag
        if tag == "Home":
            home = Home(self.main_content, self, self.camera)
            home.pack()
        elif tag == "History":
            history = History(self.main_content, self)
            history.pack(fill="both", expand=True)
        elif tag == "Profile":
            label = tk.Label(self.main_content, text="Profile Page", bg="white")
            label.pack()

        self.update_button_appearance(tag)

    def update_button_appearance(self, selected_button):
        for button, btn_widget in self.button_dict.items():
            if button == selected_button:
                load_image = tk.PhotoImage(file=f'./desktop_app/assets/sidebar/buttons/{button.lower()}_selected.png')
            else:
                load_image = tk.PhotoImage(file=f'./desktop_app/assets/sidebar/buttons/{button.lower()}.png')
            btn_widget.config(image=load_image)
            btn_widget.image = load_image