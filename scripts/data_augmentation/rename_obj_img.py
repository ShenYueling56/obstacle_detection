import os



if __name__ == '__main__':
    path = "/media/shenyl/Elements/sweeper/dataset/0716/objects/new_dogshit/"
    result = "/media/shenyl/Elements/sweeper/dataset/0716/objects/new_dogshit_rename/"
    fileList = os.listdir(path)
    print("num: ")
    print(len(fileList))
    idx = 7
    for f in fileList:
        # print(path + f)
        print(result+"%06d.png"%idx)
        os.rename(path + f, result+"%06d.png"%idx)
        idx = idx + 1
