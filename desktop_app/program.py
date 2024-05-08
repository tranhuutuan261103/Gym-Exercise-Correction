import tkinter as tk
from tkinter import messagebox
from app_main import AppMain
import cv2

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Workout Tracker')
        self.geometry('950x500')
        self.configure(bg='white')
        self.resizable(False, False)

        self.img = tk.PhotoImage(file='./desktop_app/assets/login/dumbbell.png')
        tk.Label(self, image=self.img, bg='white').place(x=0, y=0)

        self.frame = tk.Frame(self, width=350, height=350, bg='white')
        self.frame.place(x=520, y=70)

        self.heading = tk.Label(self.frame, text='Login', font=('Microsoft YaHei UI Light', 23, 'bold'), bg='white', fg='black')
        self.heading.place(x=150, y=10)

        # Username entry
        self.user = tk.Entry(self.frame, width=25, fg='black', bg='white', border=0, font=('Microsoft YaHei UI Light', 15))
        self.user.place(x=50, y=70)
        self.user.insert(0, 'Username')
        self.user.bind('<FocusIn>', self.on_enter)
        self.user.bind('<FocusOut>', self.on_leave)
        tk.Frame(self.frame, width=300, height=2, bg='black').place(x=50, y=100)

        # Password entry
        self.password = tk.Entry(self.frame, width=25, fg='black', bg='white', border=0, font=('Microsoft YaHei UI Light', 15))
        self.password.place(x=50, y=170)
        self.password.insert(0, 'Password')
        self.password.bind('<FocusIn>', self.on_enter)
        self.password.bind('<FocusOut>', self.on_leave)
        tk.Frame(self.frame, width=300, height=2, bg='black').place(x=50, y=200)

        # Login Button
        tk.Button(self.frame, text='Login', font=('Microsoft YaHei UI Light', 15, 'bold'), bg='black', fg='white', border=0, width=10, height=1, command=self.login).place(x=135, y=250)

        self.camera = cv2.VideoCapture(0)

    def on_enter(self, e):
        if e.widget == self.user and self.user.get() == 'Username':
            self.user.delete(0, tk.END)
            self.user.config(fg='black')
        elif e.widget == self.password and self.password.get() == 'Password':
            self.password.delete(0, tk.END)
            self.password.config(fg='black')

    def on_leave(self, e):
        if e.widget == self.user and self.user.get() == '':
            self.user.insert(0, 'Username')
        elif e.widget == self.password and self.password.get() == '':
            self.password.insert(0, 'Password')

    def login(self):
        if self.user.get() == 'admin' and self.password.get() == 'admin':
            self.withdraw()

            def on_dashboard_close():
                app.destroy()
                self.deiconify()

            app = AppMain(self.camera, on_dashboard_close)
        else:
            messagebox.showerror('Error', 'Invalid Username or Password')

if __name__ == "__main__":
    app = App()
    app.mainloop()