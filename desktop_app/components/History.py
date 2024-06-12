import tkinter as tk
from tkinter import ttk
from tkinter import Toplevel, PhotoImage, filedialog
from PIL import Image, ImageTk
import cv2
import requests
from io import BytesIO
from datetime import datetime
from services.Histories import get_histories
import threading
from tkinter.ttk import * 

class History(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.current_selected_history = None
        self.current_selected_frame_history = None

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
        self.create_styles()

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
            self.frame_history_item = ttk.Frame(self.frame_history, relief='solid', borderwidth=1)
            self.frame_history_item.pack(fill='both', pady=10, padx=10)

            # Bind the click event to a method
            self.frame_history_item.bind("<Button-1>", lambda event, h=history, history_frame=self.frame_history_item: self.on_history_click(h, history_frame))

            title = ttk.Label(self.frame_history_item, text=history["ExcerciseName"], font=("Arial", 16, "bold"))
            title.pack(anchor='w')
            title.bind("<Button-1>", lambda event, h=history, history_frame=self.frame_history_item: self.on_history_click(h, history_frame))  # Bind click to the label

            date = ttk.Label(self.frame_history_item, text=history["Datetime"], font=("Arial", 12))
            date.pack(anchor='w')
            date.bind("<Button-1>", lambda event, h=history, history_frame=self.frame_history_item: self.on_history_click(h, history_frame))  # Bind click to the label

    def create_styles(self):
        self.s = Style()
        self.s.configure('My.TFrameA', background='#0000ff')
        self.s.configure('My.TFrame', background='#f0f0f0')

    def on_history_click(self, history, history_frame):
        # This method will be called with the history item data
        # You can then update the error details frame with this data
        self.current_selected_history = history
        self.current_selected_frame_history = history_frame
        
        # for frame in self.frame_history.winfo_children():
        #     frame.config(style='My.TFrame')
        # history_frame.config(style='My.TFrameA')
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

        if self.current_selected_history:
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

        if self.current_selected_history:
            for error in error_detail['SpecificErrorFrames']:
                # Each error gets its own frame within the detail frame
                error_report_frame = ttk.Frame(self.detail_frame, relief='solid', borderwidth=1)
                error_report_frame.pack(fill='x', pady=5, padx=10)

                # Labels for timestamp, score, and description
                timestamp_label = ttk.Label(error_report_frame, text=f'{error["ErrorType"]}: ', font=("Arial", 12))
                timestamp_label.grid(row=0, column=0, sticky="w", padx=5)

                count_label = ttk.Label(error_report_frame, text=error["Count"], font=("Arial", 12, "bold"), foreground="red")
                count_label.grid(row=0, column=1, sticky="w")

                error_detail_image_container = ttk.Frame(error_report_frame)
                error_detail_image_container.grid(row=1, column=0, columnspan=2, sticky="w", padx=5)

                for i in range(len(error['ImageUrl']) // 3 + 1):
                    error_detail_image_container_row = ttk.Frame(error_detail_image_container)
                    for j in range(3):
                        if i * 3 + j >= len(error['ImageUrl']):
                            break
                        error_detail_images = error['ImageUrl'][i * 3 + j]
                        
                        def callback(photo, container=error_detail_image_container_row):
                            if photo:
                                image_placeholder = tk.Label(container, image=photo)
                                image_placeholder.photo = photo  # keep a reference!
                                image_placeholder.pack(side="left", padx=5)
                        
                        load_image_from_url(error_detail_images, callback)
                    
                    error_detail_image_container_row.pack(side="top", padx=5)
            

    def get_error_details_for_current_history(self):
        return self.current_selected_history
    
def load_image_from_url(url, callback):
    def task():
        try:
            response = requests.get(url)
            response.raise_for_status()
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            image = image.resize((200, 120))
            photo = ImageTk.PhotoImage(image)
            callback(photo)
        except requests.exceptions.RequestException as e:
            # print(f"HTTP Request error: {e}")
            pass
        except Exception as e:
            # print(f"An error occurred while loading image: {e}")
            pass
        callback(None)

    thread = threading.Thread(target=task)
    thread.start()