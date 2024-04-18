import tkinter as tk
from tkinter import ttk
from tkinter import Toplevel, PhotoImage, filedialog
from PIL import Image, ImageTk
import cv2
import requests
from io import BytesIO
from datetime import datetime
from services.Histories import get_histories

class History(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.curent_selected_history = None

        # Title label
        self.label_title = tk.Label(self, text="History                                                              ", font=("Arial", 46, "bold"), fg="#3C2937")
        self.label_title.grid(row=0, columnspan=2, pady=(20, 10), padx=20, sticky="w")

        # Main container
        self.fram_main = ttk.Frame(self)
        self.fram_main.grid(row=1, column=0, sticky='nsew')
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create the history list frame and tab control
        self.create_history_list()
        self.create_tabs()

        # Make the History frame fill the entire space of the parent container
        self.pack(fill='both', expand=True)

    def create_history_list(self):
        # History list frame on the left
        self.frame_history = ttk.Frame(self.fram_main, relief='solid', borderwidth=1, style='TFrame')
        self.frame_history.grid(row=0, column=0, sticky='nsew', padx=(10, 5), pady=10)
        self.fram_main.grid_rowconfigure(0, weight=1)
        self.fram_main.grid_columnconfigure(0, weight=3)  # 30%

        # Sample histories data
        self.histories = get_histories()

        # Add history entries to the frame
        for history in self.histories:
            history_frame = ttk.Frame(self.frame_history, relief='solid', borderwidth=1)
            history_frame.pack(fill='both', pady=10, padx=10)

            # Bind the click event to a method
            history_frame.bind("<Button-1>", lambda event, h=history: self.on_history_click(h))

            title = ttk.Label(history_frame, text=history["ExcerciseName"], font=("Arial", 16, "bold"))
            title.pack(anchor='w')

            date = ttk.Label(history_frame, text=history["Datetime"], font=("Arial", 12))
            date.pack(anchor='w')

    def on_history_click(self, history):
        # This method will be called with the history item data
        # You can then update the error details frame with this data
        self.curent_selected_history = history
        self.update_error_sumary_tab()
        self.update_error_detail_tab()

    def create_tabs(self):
        # Tab control on the right
        tab_control = ttk.Notebook(self.fram_main)
        tab_control.grid(row=0, column=1, sticky='nsew', padx=(5, 10), pady=10)
        self.fram_main.grid_columnconfigure(1, weight=7)  # 70%

        # Create each tab
        self.create_error_sumary_tab(tab_control, "SUMMARY")
        self.create_error_detail_tab(tab_control, "DETAIL")

    def create_error_sumary_tab(self, parent, error_title):
        self.frame_summary = ttk.Frame(parent)
        parent.add(self.frame_summary, text=error_title)
        label = ttk.Label(self.frame_summary, text="Select a history to view the error details.", foreground="gray")
        label.pack(pady=10)

    def update_error_sumary_tab(self):
        # Clear the current content
        for widget in self.frame_summary.winfo_children():
            widget.destroy()

        error_detail = self.get_error_details_for_current_history()

        label_name = ttk.Label(self.frame_summary, text=f"{error_detail['ExcerciseName']}", font=("Arial", 16, "bold"))
        label_name.pack(anchor='w', padx=10, pady=10)

        label_date = ttk.Label(self.frame_summary, text=f"{error_detail['Datetime']}", font=("Arial", 12))
        label_date.pack(anchor='w', padx=10)

        label_duration = ttk.Label(self.frame_summary, text=f"Duration: {error_detail['Duration']} seconds", font=("Arial", 12))
        label_duration.pack(anchor='w', padx=10)

        label = ttk.Label(self.frame_summary, text=f"There are {error_detail['ErrorTotalCount']} errors found.", foreground="red", justify='left')
        label.pack(pady=10)

        if self.curent_selected_history:
            for error in error_detail['SpecificErrorFrames']:
                error_label = ttk.Label(self.frame_summary, text=f"{error['ErrorType']}: {error['Count']}")
                error_label.pack(anchor='w', padx=10, pady=5)

    def create_error_detail_tab(self, parent, error_title):
        # Container frame for the detailed error reports
        self.detail_frame = ttk.Frame(parent)
        parent.add(self.detail_frame, text=error_title)
        label = ttk.Label(self.detail_frame, text="Select a history to view the error details.", foreground="gray")
        label.pack(pady=10)

    def update_error_detail_tab(self):
        # Clear the current content
        for widget in self.detail_frame.winfo_children():
            widget.destroy()

        error_detail = self.get_error_details_for_current_history()

        label = ttk.Label(self.detail_frame, text=f"There are {error_detail['ErrorTotalCount']} errors found.", foreground="red")
        label.pack(pady=10)

        if self.curent_selected_history:
            for error in error_detail['SpecificErrorFrames']:
                # Each error gets its own frame within the detail frame
                error_report_frame = ttk.Frame(self.detail_frame, relief='solid', borderwidth=1)
                error_report_frame.pack(fill='x',  pady=5, padx=10)

                # Labels for timestamp, score, and description
                timestamp_label = ttk.Label(error_report_frame, text=f'{error["ErrorType"]}: ', font=("Arial", 12))
                timestamp_label.grid(row=0, column=0, sticky="w", padx=5)

                count_label = ttk.Label(error_report_frame, text=error["Count"], font=("Arial", 12, "bold"), foreground="red")
                count_label.grid(row=0, column=1, sticky="w")

                error_detail_image_container = ttk.Frame(error_report_frame)
                error_detail_image_container.grid(row=1, column=0, columnspan=2, sticky="w", padx=5)

                for error_detail_images in error['ImageUrl']:
                    try:
                        photo = load_image_from_url(error_detail_images)
                        image_placeholder = tk.Label(error_detail_image_container, image=photo)
                        image_placeholder.photo = photo  # keep a reference!
                        image_placeholder.grid(row=0, column=0, sticky="w", padx=5)
                    except Exception as e:
                        print(f"An error occurred: {e}")
            

    def get_error_details_for_current_history(self):
        return self.curent_selected_history
    
def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    image = Image.open(image_data)
    image = image.resize((200, 120))
    return ImageTk.PhotoImage(image)