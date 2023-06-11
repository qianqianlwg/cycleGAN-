# 获取第一个参数，即数据集名称
FILE=$1

# 检查数据集名称是否有效
if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "mini" && $FILE != "mini_pix2pix && $FILE != "mini_colorization" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi
//首先获取第一个参数，即数据集名称。然后，它检查数据集名称是否有效。如果数据集名称无效，则输出一条错误消息并退出。如果数据集名称有效，则输出数据集名称


# 如果数据集是 Cityscapes，则输出一条消息并退出
if [[ $FILE == "citysc" ]]; then
    echo "Due to license issue, we cannot provide the Cityscapes dataset from our repository. Please download the Cityscapes dataset from https://cityscapes-dataset.com, and use the script ./datasets/prepare_cityscapes_dataset.py."
    echo "You need to download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip. For further instruction, please read ./datasets/prepare_cityscapes_dataset.py"
    exit 1
fi

# 输出数据集名称
echo "Specified [$FILE]"

# 下载数据集
URL=http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/

wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
//脚本使用 wget 命令下载数据集，并使用 unzip 命令解压缩数据集。下载完成后，脚本会删除 ZIP 文件