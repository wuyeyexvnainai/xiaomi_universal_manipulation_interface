"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
# 变量解释器，允许你在配置文件中使用特殊的标记，这些标记在配置文件加载时会被解析和替换为具体的值。
OmegaConf.register_new_resolver("eval", eval, replace=True)
# 当你运行train.py并使用Hydra装饰器时，Hydra会在diffusion_policy/config目录下查找.yaml或其他指定的配置文件。
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    # 使用OmegaConf.resolve来解析配置中的所有占位符。
    OmegaConf.resolve(cfg)
    # 调用在配置文件中的_target_属性，获取配置中指定的目标类。
    cls = hydra.utils.get_class(cfg._target_)
    # 创建BaseWorkspace的实例，该实例等于cls类【该类继承于base类】，其中的属性由配置文件来配置他
    workspace: BaseWorkspace = cls(cfg)
    # /home/dell/code/dfpolicy/dfpolicy/xiaomi_universal_manipulation_interface/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py
    # run的是class TrainDiffusionUnetImageWorkspace(BaseWorkspace)的run函数
    workspace.run()

if __name__ == "__main__":
    main()
