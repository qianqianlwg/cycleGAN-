""" 通用测试脚本用于图像到图像转换。

一旦您使用train.py训练好您的模型,您就可以使用这个脚本来测试模型。
它将从'--checkpoints_dir'加载一个保存的模型,并将结果保存到'--results_dir'。

它首先根据选项创建模型和数据集。它会硬编码一些参数。
然后,它为'--num_test'张图像运行推理,并将结果保存到HTML文件中。

例子(您需要先训练模型或者从我们的网站下载预训练模型):
    测试CycleGAN模型(双向):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    测试CycleGAN模型(单向):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    选项'--model test'用于仅为一侧生成CycleGAN结果。
    这个选项会自动设置'--dataset_mode single',它只加载一组图像。
    相反,使用'--model cycle_gan'需要加载和生成双向结果,有时是不必要的。结果将保存到./results/。
    使用'--results_dir <目录路径保存结果>'指定结果目录。

    测试pix2pix模型:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

查看更多测试选项:options/base_options.py 和 options/test_options.py。
查看训练和测试提示:https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
查看常见问题:https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print(' 警告:未找到wandb包。选项"--use_wandb" 将导致错误.')



if __name__ == '__main__':
    opt = TestOptions().parse()  # 获取测试选项
    # 硬编码一些参数用于测试
    opt.num_threads = 0  # 测试代码只支持num_threads = 0
    opt.batch_size = 1  # 测试代码只支持batch_size = 1
    opt.serial_batches = True  # 禁用数据洗牌;注释此行如果需要在随机选择的图像上获得结果。
    opt.no_flip = True  # 不翻转;注释此行如果需要翻转图像的结果。
    opt.display_id = -1  # 没有visdom显示;测试代码将结果保存到HTML文件。
    dataset = create_dataset(opt)  # 根据opt.dataset_mode 和其他选项创建数据集
    model = create_model(opt)  # 根据opt.model 和其他选项创建模型
    model.setup(opt)  # 正常设置:加载和打印网络;创建调度程序

    # 初始化logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name,
                               config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # 创建网站
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # 定义网站目录
    if opt.load_iter > 0:  # 默认加载iter是0
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # 使用eval模式测试。这只影响像批归一化和丢弃这样的层。
    # 对于[pix2pix]:我们在原始的pix2pix中使用批归一化和丢弃。您可以尝试使用和不使用eval()模式。
    # 对于[CycleGAN]:它不应该影响CycleGAN,因为
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # 仅将我们的模型应用于opt.num_test张图像。
            break
        model.set_input(data)  # 从数据加载器中解包数据
        model.test()           # 运行推理
        visuals = model.get_current_visuals() # 获取图像结果
        img_path = model.get_image_paths()    # 获取图像路径
        if i % 5 == 0: # 每5个保存一次图像到HTML文件
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # 保存HTML
