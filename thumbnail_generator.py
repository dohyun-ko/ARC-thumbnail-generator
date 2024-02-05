
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ARCDataset:
    train_path = './ARC_training/'
    eval_path = './ARC_evaluation/'

    # load training data
    def load_train_data(self, shuffle = False, jcode = True):
        train = []
        test = []
        j_codes = []

        json_files = glob.glob(os.path.join(self.train_path, '*.json'))

        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                train.append(data['train'])
                test.append(data['test'])

        if shuffle:
            np.random.seed = 777
            np.random.shuffle(train)
            np.random.shuffle(test)

        for i in range(len(json_files)):
            j_codes.append(json_files[i].split('/')[-1].split('.')[0])
        
        if jcode:
            return train, test, j_codes
        else:
            return train, test

    # load evaluation data
    def load_eval_data(self, shuffle = False, jcode = True):
        train = []
        test = []
        j_codes = []

        json_files = glob.glob(os.path.join(self.eval_path, '*.json'))
        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                train.append(data['train'])
                test.append(data['test'])

        if shuffle:
            np.random.seed = 777
            np.random.shuffle(train)
            np.random.shuffle(test)

        for i in range(len(json_files)):
            j_codes.append(json_files[i].split('/')[-1].split('.')[0])
        
        if jcode:
            return train, test, j_codes
        else:
            return train, test
    
    # make data np.array
    def flatten_data(self, datas):
        result = []
        for data in datas:
            for pair in data:
                result.append([np.array(pair['input']), np.array(pair['output'])])
        return result
    
    # jcode to index
    def jtoi(self, jcode, j_codes):
        if jcode in j_codes:
            return j_codes.index(jcode)
        else: 
            return None
        
    # index to jcode
    def itoj(self, index, j_codes):
        if index < len(j_codes):
            return j_codes[index]
        else:
            return None
    

settings = json.load(open('settings.json', 'r'))
colors_rgb = settings['colors_rgb']

def convert_data(datas):
    result = []

    for data in datas:
        if type(data) == dict:
            d_input = [[np.uint8(colors_rgb[value]) for value in line] for line in data['input']]
            d_output = [[np.uint8(colors_rgb[value]) for value in line] for line in data['output']]
        elif type(data) == list or type(data) == np.ndarray:
            d_input = [[np.uint8(colors_rgb[value]) for value in line] for line in data[0]]
            d_output = [[np.uint8(colors_rgb[value]) for value in line] for line in data[1]]
        else:
            raise Exception("Invalid data type : ", type(data))
        result.append([np.array(d_input), np.array(d_output)])
    return result

def convert_img2color(img):
    result = []
    for i in range(img.shape[0]):
        result.append([])
        for j in range(img.shape[1]):
            if img[i][j] != 0 and img[i][j] != 1 and img[i][j] != 2 and img[i][j] != 3 and img[i][j] != 4 and img[i][j] != 5 and img[i][j] != 6 and img[i][j] != 7 and img[i][j] != 8 and img[i][j] != 9:
                result[i].append(np.uint8([245, 245, 245]))
            else:
                result[i].append(np.uint8(colors_rgb[int(img[i][j])]))
    return np.array(result)

# plot the entire problem
def plot_data(datas, title=None, path=None):
    datas = convert_data(datas)
    num_data = len(datas)
    fig, axs = plt.subplots(num_data, 2, figsize=(8, 4 * num_data))
    fig.subplots_adjust(wspace=0.5, hspace=0.3)

    if title:
        fig.suptitle(title)

    if num_data == 1:
        axs[0].imshow(datas[0][0])
        axs[1].imshow(datas[0][1])
        axs[0].axis('off')
        axs[1].axis('off')
        plt.savefig(path, bbox_inches='tight')
        # plt.show()
        return
    else:
        for i in range(num_data):
            plt.axis('off')
            axs[i, 0].imshow(datas[i][0])
            axs[i, 1].imshow(datas[i][1])
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
        plt.savefig(path, bbox_inches='tight')
        # plt.show()

# plot individual image
def plot_img(img, title=None, block=True, transfrom=True):
    if type(img) == list:
        img = np.array(img)

    plt.figure(figsize=(8, 8))
    plt.xticks(np.arange(0, img.shape[1], 1))
    plt.yticks(np.arange(0, img.shape[0], 1))
    if transfrom:
        img = convert_img2color(img)
    if title:
        plt.title(title)
    plt.imshow(img)
    plt.show(block=block)







if __name__ == '__main__':
    arc = ARCDataset()

    # train
    train, test, j_codes = arc.load_train_data(shuffle = False, jcode = True)

    # create thumbnails for all 400 problems
    # if path not exist, create it
    if not os.path.exists('./thumbnails'):
        os.mkdir('./thumbnails')
    # make directory for train and its test and answer
    if not os.path.exists('./thumbnails/train'):
        os.mkdir('./thumbnails/train')
    if not os.path.exists('./thumbnails/train/test_and_answer'):
        os.mkdir('./thumbnails/train/test_and_answer')
    

    # create thumbnails for train data
    for i in range(len(train)):
        plot_data(train[i], path='./thumbnails/train/' + j_codes[i] + '.png')

    for i in range(len(test)):
        plot_data(test[i], path='./thumbnails/train/test_and_answer/' + j_codes[i] + '.png')


    ########################################################################################
    ########################################################################################
    
    # eval
    train, test, j_codes = arc.load_eval_data(shuffle = False, jcode = True)

    # make directory for eval and its test and answer
    if not os.path.exists('./thumbnails/eval'):
        os.mkdir('./thumbnails/eval')
    if not os.path.exists('./thumbnails/eval/test_and_answer'):
        os.mkdir('./thumbnails/eval/test_and_answer')
    
    # create thumbnails for eval data
    for i in range(len(train)):
        plot_data(train[i], path='./thumbnails/eval/' + j_codes[i] + '.png')

    for i in range(len(test)):
        plot_data(test[i], path='./thumbnails/eval/test_and_answer/' + j_codes[i] + '.png')