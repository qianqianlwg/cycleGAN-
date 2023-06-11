from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """包含测试选项

 它还包括在BaseOptions中定义的共享选项。
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # 定义共享选项
        parser.add_argument('--results_dir', type=str, default='./results/', help='保存结果在这里.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='结果图像的宽高比')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test等')
        # 训练和测试时Dropout和Batchnorm有不同的行为。
        parser.add_argument('--eval', action='store_true', help='在测试时使用eval模式。')
        parser.add_argument('--num_test', type=int, default=50, help='运行多少测试图像')
        # 重写默认值
        parser.set_defaults(model='test')
        # 为避免裁剪,load_size应与crop_size相同
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser

