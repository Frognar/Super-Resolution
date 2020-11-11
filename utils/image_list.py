import json
import os

from PIL import Image


class ImageListCreator:
    def __init__(self, min_size):
        self._min_size = min_size

    def create(self, directory, output_directory, list_name):
        image_list = self._get_image_list(directory=directory)
        ImageListCreator._save_list_to_file(image_list=image_list, list_name=list_name,
                                            output_directory=output_directory)

    def _get_image_list(self, directory):
        file_list = os.listdir(directory)
        image_list = self._get_list_contains_only_images(directory=directory, file_list=file_list)
        return image_list

    def _get_list_contains_only_images(self, directory, file_list):
        def image_path(file): return os.path.join(directory, file)

        return [image_path(file_name) for file_name in file_list if
                self._is_file_an_image_of_proper_size(file_name=image_path(file_name))]

    def _is_file_an_image_of_proper_size(self, file_name):
        if self._is_file_an_image(file_name=file_name):
            return self._is_image_of_proper_size(image_name=file_name)
        return False

    @staticmethod
    def _is_file_an_image(file_name):
        return any(file_name.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def _is_image_of_proper_size(self, image_name):
        image = Image.open(image_name, mode='r')
        return image.width >= self._min_size and image.height >= self._min_size

    @staticmethod
    def _save_list_to_file(image_list, list_name, output_directory):
        output_file = os.path.join(output_directory, list_name)
        with open(output_file, mode='w') as json_image_list:
            json.dump(image_list, json_image_list)
