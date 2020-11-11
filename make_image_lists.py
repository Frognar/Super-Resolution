from utils.image_list import ImageListCreator

if __name__ == '__main__':
    image_list_creator = ImageListCreator(min_size=200)
    image_list_creator.create(directory='./data/train/', output_directory='./data/', list_name='train.json')
    image_list_creator.create(directory='./data/val/', output_directory='./data/', list_name='val.json')
    image_list_creator.create(directory='./data/test/BSDS100/', output_directory='./data/', list_name='bsds100.json')
    image_list_creator.create(directory='./data/test/Manga109/', output_directory='./data/', list_name='manga109.json')
    image_list_creator.create(directory='./data/test/Set5/', output_directory='./data/', list_name='set5.json')
    image_list_creator.create(directory='./data/test/Set14/', output_directory='./data/', list_name='set14.json')
    image_list_creator.create(directory='./data/test/Urban100/', output_directory='./data/', list_name='urban100.json')
