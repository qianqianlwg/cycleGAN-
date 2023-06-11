from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """
    这个TestModel可以用于仅生成一个方向的CycleGAN结果。
    这个模型将自动设置'--dataset_mode single',它只加载一个集合中的图像。

    有关更多详细信息,请参阅测试说明。
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于数据集的选项,并重写现有选项的默认值。

        参数:
            parser - 原始选项解析器
            is_train (bool) - 是否处于训练阶段还是测试阶段。您可以使用此标志添加训练专用或测试专用选项。

        返回修改后的解析器。

        该模型仅可在测试时使用。它需要'--dataset_mode single'。
        您需要使用'--model_suffix'选项指定网络。
        """
        assert not is_train,'TestModel不能在训练时使用'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix',type=str,default='',help='在checkpoints_dir中,[epoch]_net_G[model_suffix].pth将被加载为生成器。')
        return parser

    def __init__(self, opt):
        """初始化pix2pix类。

        参数:
            opt (Option类)-- 存储所有实验标志; 必须是BaseOptions的子类
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失
        self.loss_names = []
        # 指定要保存/显示的图像
        self.visual_names = ['real', 'fake']
        # 指定要保存到磁盘的模型
        self.model_names = ['G' + opt.model_suffix]  # 只需要生成器。
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # 将模型分配给self.netG_[suffix],以便可以加载
        # 请参阅<BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # 在self中存储netG。

    def set_input(self, input):
        """从dataloader解包输入数据并执行必要的预处理步骤。

        参数:
            input:包含数据本身及其元数据信息的字典。

        我们需要使用'single_dataset'数据集模式。它只从一个域加载图像。
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """运行前向传递。"""
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """测试模型无优化。"""
        pass
