from huggingface_hub import notebook_login
import wandb
wandb.login()
notebook_login()
import warnings
warnings.filterwarnings('ignore')
from ylcm import unconditional_oxfordflowers102_cmconfig_dict, CMConfig, get_config
config = get_config(unconditional_oxfordflowers102_cmconfig_dict, CMConfig)
print(config)
from ylcm import Consistency
cm = Consistency(config)
cm.train()