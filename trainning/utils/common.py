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