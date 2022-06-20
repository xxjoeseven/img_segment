import os
from glob import glob


def get_img_list(folder_path, format):
    """Search given path for .tif files

    Args: 
        folder_path: string path of the directory with the images
        format: string image format extension such as tif or gif etc

    Return:
        a list of the images path.
    """

    # Set original working directory
    original_cwd = os.getcwd()
    # Change working directory to image folder
    os.chdir(folder_path)
    # Get full path of image folder
    new_cwd = os.getcwd()
    # Get list of all tif images path in the folder
    list_img = glob(os.path.join(new_cwd, f"*.{format}"))
    # Set the working directory back to original
    os.chdir(original_cwd)

    return list_img

def make_dir(path):
    """
    """

    if not os.path.exists(path):
        os.makedirs(path)

def get_filename(file_path):
    """
    """

    file_name = os.path.basename(file_path).split('.')[0]

    return file_name
