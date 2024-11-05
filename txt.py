import os

def write_name(np, tx):
    # npz文件路径
    files = os.listdir(np)
    # txt文件路径
    f = open(tx, 'w')
    for i in files:
        # name = i.split('\\')[-1]
        name = i[:-4] + '\n'
        f.write(name)


# write_name('./data/RITEyes/train_npz', './lists/lists_RITEyes/train.txt')
# write_name('./data/REFUGE/val_npz', './lists/lists_REFUGE/val.txt')

