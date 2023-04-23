from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn import MSELoss, L1Loss
def get_loss_fn(loss_fn_type):
    if loss_fn_type in ["LearnedPerceptualImagePatchSimilarity(net_type='vgg')", 'MSELoss()', 'L1Loss()']:
        return eval(loss_fn_type)
