import pickle

def extract_parts_of_file(image_path):
    """
    Trích xuất ra các phần của 1
    """
    file_name, file_extension = image_path.split(".")
    return file_name, file_extension


def get_current_time_string():
    from datetime import datetime

    current_time = datetime.now()
    return f"{current_time.year}{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}{current_time.microsecond}"

def load_model(file_name):
    with open(file_name, "rb") as file:
        model = pickle.load(file)
        return model

def save_model(model, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(model, file)

def get_color_for_landmarks(label):
    if label == "C":
        return ((255, 165, 0), (255, 140, 0))
    elif label == "W":
        return ((29, 62, 199), (1, 143, 241))