
from os import path
from matplotlib import pyplot as plt


IMAGE_PATH = path.join(__file__, '../images')


def save_fig_to_images(name):
    plt.savefig(path.abspath(path.join(IMAGE_PATH, name)))
