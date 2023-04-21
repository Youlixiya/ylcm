import wandb
import warnings
from ylcm import unconditional_oxfordflowers102_cmconfig_dict, CMConfig, get_config, Consistency
wandb.login()
warnings.filterwarnings('ignore')
config = get_config(unconditional_oxfordflowers102_cmconfig_dict, CMConfig)
cm = Consistency(config)
cm.train()