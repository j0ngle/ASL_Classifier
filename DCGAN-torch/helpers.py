import matplotlib.pyplot as plt

def show_image(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

def save_graph(title, x_label, y_label, epoch, list1, list1_label, list2=None, list2_label=None):
    # title = title + "_at_epoch_{:04}".format(epoch)

    plt.figure(figsize=(10, 3)) 
    plt.title(title)
    plt.plot(list1, label=list1_label)
    plt.plot(list2, label=list2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    # plt.show()

    #TODO: Save to folder bc right now it isn't working for some reason
    filename = "DCGAN-torch/metrics/" + title + "_epoch_{:04}.png".format(epoch)
    # dir = os.path.join("metrics/"+filename)
    
    plt.savefig(filename)