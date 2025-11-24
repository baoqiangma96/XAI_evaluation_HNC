import monai
import torch



def build_model():
    """
    Build a 3D MONAI DenseNet121 model for single-output prediction.
    The XAI app will automatically load weights from .safetensors.
    """
    input_channel = 1   # change to match your training setup (e.g. 1 for CT, 2 for CT+PET)

    return monai.networks.nets.DenseNet121(
        spatial_dims=3,
        in_channels=input_channel,
        out_channels=1
    ) 

'''
def build_model():
    """
    Build a 3D MONAI DenseNet121 model for single-output prediction,
    with an added normalization layer to scale output to [0, 1].
    """
    input_channel = 1   # change to match your setup (e.g., 3 for CT+PT+GTV)
    min_val, max_val = -1.0833, 7.9800  # use your known risk range

    class MinMaxNormalize(torch.nn.Module):
        def forward(self, x):
            return (x - min_val) / (max_val - min_val )

    model = monai.networks.nets.DenseNet121(
        spatial_dims=3,
        in_channels=input_channel,
        out_channels=1
    )

    return torch.nn.Sequential(model, MinMaxNormalize()) 
'''