import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This file defines the model architectures
"""
SOS_token = 0
EOS_token = 1


class multSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]


class Unet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Unet, self).__init__()
        # construct unet structure

        unet_block = UNetBlock(ngf * 8,
                               ngf * 8,
                               input_nc=None,
                               submodule=[RnnGeneratorBlock(input_size=ngf * 8, hidden_size=ngf * 8),
                                          RnnDecoderBlock(input_size=ngf * 8, hidden_size=ngf * 8)],
                               innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UNetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UNetBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UNetBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UNetBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        self.model = UNetBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                               outermost=True)  # add the outermost layer

    def forward(self, input):
        return self.model(input)


class RnnGeneratorBlock(nn.Module):
    def init_gru_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    def __init__(self, input_size, hidden_size):
        super(RnnGeneratorBlock, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)  # Why is this biderctional?
        self.init_gru_weights()
    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(1))  # Reshape to (batch, length of RNN sequence, features)
        output, hidden = self.gru(x)
        return output, hidden


class RnnDecoderBlock(nn.Module):
    def __init__(self, input_size, hidden_size, sobmodule=None):
        super(RnnDecoderBlock, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size*2, hidden_size*2, batch_first=True)
        self.conv = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(1024)
        self.out = nn.Linear(hidden_size*2, hidden_size)
        self.pool = nn.MaxPool2d(kernel_size=(1, 4))
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, encoder_outputs, encoder_hidden):
        batch_size = encoder_outputs.size(0)

        decoder_input = encoder_outputs

        decoder_hidden = encoder_hidden.view(2, batch_size, 512).reshape(1, batch_size, 1024)
        decoder_outputs = []

        for i in range(4):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        concat_outputs = torch.cat(decoder_outputs, dim=1)
        norm_outputs = self.batchnorm(concat_outputs.view(batch_size, 1024, 4, 4))
        pooled_outputs = self.pool(norm_outputs)
        conv_outputs = self.conv2(pooled_outputs)
        decoder_outputs = self.relu(conv_outputs).view(batch_size, 512, 2, 2)
        return decoder_outputs

    def forward_step(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden


class UNetBlock(nn.Module):
    """
    Defines one layer in the U-Net architecture
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UNetBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UNetBlock, self).__init__()
        self.outermost = outermost
        use_bias = False

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + submodule + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = multSequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Discriminator(nn.Module):
    """
    Implements a PatchGAN discriminator
    """
    
    def __init__(self, input_nc, ndf=64, n_layers=3):
        """
        Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()

        # Define the convolutional layers
        kw = 4  # Kernel size
        padw = 1  # Padding size

        # First layer
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        # Intermediate layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output layer
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # Output a single-channel prediction map

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
