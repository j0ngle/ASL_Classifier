import matplotlib.pyplot as plt

def show_image(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()