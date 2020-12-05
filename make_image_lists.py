import os

from utils import ImageListCreator


def main():
    image_list_creator = ImageListCreator(min_size=256)
    create_train_list(image_list_creator)
    create_test_lists(image_list_creator)


def create_train_list(image_list_creator):
    create_image_list(image_list_creator, './data/train/', 'train')


def create_image_list(image_list_creator, directory, data_set):
    image_list_creator.create(
        directory=directory,
        output_directory='./data/',
        list_name=f'{data_set}.json'
    )


def create_test_lists(image_list_creator):
    for test_set in os.listdir('./data/test'):
        create_test_list(image_list_creator, test_set)


def create_test_list(image_list_creator, test_set):
    directory = f'./data/test/{test_set}/'
    if is_directory(directory):
        create_image_list(image_list_creator, directory, test_set)


def is_directory(directory):
    return os.path.isdir(directory)


if __name__ == '__main__':
    main()
