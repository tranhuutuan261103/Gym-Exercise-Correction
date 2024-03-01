def extract_parts_of_file(image_path):
    """
    Trích xuất ra các phần của 1
    """
    file_name, file_extension = image_path.split(".")
    return file_name, file_extension