import numpy as np
from imageFinder import ImageFinder
import matplotlib.pyplot as plt

THRESHOLD = np.arange(0.3, 1.6, 0.1, dtype='float32')
DATA_DIR = './Data/subset/'

def main():
    imf = ImageFinder()
    image_list = imf.load_data()

    precision_list = []
    recall_list = []
    for t in THRESHOLD:
        # Get matching list from first image
        match_list = imf.calc_euclidean_distance(image_list[0], threshold=t)
        precision_list.append(imf.get_precision())
        recall_list.append(imf.get_recall())
        imf = ImageFinder()
    print(precision_list)
    print(recall_list)

    # celeb_img_id_dict = imf.get_file_id_dict()

    # precision_list = []
    # recall_list = []
    # img_id_list = []
    # for i in range(0,10):
    #     image = random.choice(image_list)
    #     image_file_name = image.filename.split('./Data/subset/')[1]
    #     img_id_list.append(celeb_img_id_dict[image_file_name]) # append all ID
    #     for t in THRESHOLD:
    #         # Get matching list from first image
    #         match_list = imf.calc_euclidean_distance(image_list[0], threshold=t)
    #         precision_list.append(imf.get_precision())
    #         recall_list.append(imf.get_recall())
    #         imf = ImageFinder()
    # print(precision_list)
    # print(recall_list)

    # fig, axs = plt.subplots(2)
    # axs[0].plot(THRESHOLD, )
    # axs[1].plot(THRESHOLD, -y)

    # plt.plot(THRESHOLD, , 'b', label='ID:{0}'.format())
    # plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.show()

if __name__ == '__main__':
    main()