import json
import os

from PIL import Image
from tqdm import tqdm


class ImageListCreator:
    def __init__(self, min_size):
        self.min_size = min_size

    def create(self, directory, output_directory, list_name):
        image_list = self.get_images_from(directory)
        ImageListCreator.save_list(image_list=image_list, list_name=list_name,
                                   output_directory=output_directory)

    def get_images_from(self, directory):
        files = os.listdir(directory)

        def path_of(file): return os.path.join(directory, file)

        return [path_of(file) for file in tqdm(files) if
                self.is_an_image_of_proper_size(file=path_of(file))]

    def is_an_image_of_proper_size(self, file):
        if self.is_file_an_image(file=file):
            return self.is_image_of_proper_size(image=file)
        return False

    @staticmethod
    def is_file_an_image(file):
        return any(file.endswith(extension) for extension in
                   ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def is_image_of_proper_size(self, image):
        image = Image.open(image, mode='r')
        return image.width >= self.min_size and image.height >= self.min_size

    @staticmethod
    def save_list(image_list, list_name, output_directory):
        output_file = os.path.join(output_directory, list_name)
        with open(output_file, mode='w') as json_image_list:
            json.dump(image_list, json_image_list)
