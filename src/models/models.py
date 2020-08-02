from torch import nn
import torch
from torch.autograd import Variable
import torch

# class STHeaviside(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         y = torch.zeros(x.size()).type_as(x)
#         y[x >= 0] = 1
#         return y

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output
    

    
class VAE(nn.Module):

    def __init__(self, num_latents, num_params=2, output_dist=None):
        super(VAE, self).__init__()
        self.num_latents = num_latents
        self.num_params = num_params
        self.input_size = 128

        self.output_dist = output_dist
        self.num_output_params = 1
        if self.output_dist == 'normal':
            self.num_output_params = 2
        elif self.output_dist == 'bernoulli':
            self.num_output_params = 1
        else:
            self.num_output_params = 1

        # self.act = nn.ReLU(inplace=True)
        self.act = nn.LeakyReLU(inplace=True)

        # encoder:
        self.conv0 = nn.Conv2d(1, 32, 4, 2, 1)  # 64 x 64
        self.bn0 = nn.BatchNorm2d(32)


        self.conv1 = nn.Conv2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_6_mu = nn.Conv2d(512, self.num_latents, 1)
        self.conv_6_logsigma = nn.Conv2d(512, self.num_latents, 1)

        # decoder:
        self.deconv6 = nn.ConvTranspose2d(self.num_latents, 512, 1, 1, 0)  # 1 x 1
        self.debn6 = nn.BatchNorm2d(512)
        self.deconv5 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.debn5 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.debn4 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.debn3 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.debn2 = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(32, 32, 4, 2, 1) # 64 x 64
        
        
        self.debn1 = nn.BatchNorm2d(32)
        #self.deconv0 = nn.ConvTranspose2d(32, self.num_output_params, 4, 2, 1) # 64 x 64
        
        self.deconv0_mu = nn.ConvTranspose2d(32, 1, 4, 2, 1) # 64 x 64
        self.debn0_mu = nn.BatchNorm2d(1)
        self.deconv0_logsigma = nn.ConvTranspose2d(32, 1, 4, 2, 1) # 64 x 64
        self.debn0_logsigma = nn.BatchNorm2d(1)



    def encode(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.input_size, self.input_size)

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act(x)
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act(x)
        #x = self.conv_6(x)
        #x = x.view(batch_size, self.num_latents * self.num_params)
        #return x

        mu = self.conv_6_mu(x)
        logsigma = self.conv_6_logsigma(x)

        return (mu, logsigma)



    # def reparameterize_bernoulli(self, params):
    #     presigm_ps = params
    #     logp = torch.nn.functional.logsigmoid(presigm_ps)
    #     logq = torch.nn.functional.logsigmoid(-presigm_ps)

    #     #l = self._sample_logistic(logp.size()).type_as(presigm_ps)
    #     eps = 1e-8
    #     u = Variable(torch.rand(logp.size()))
    #     l = torch.log(u + eps) - torch.log(1 - u + eps)
    #     l = l.to(params.get_device())
        
    #     z = logp - logq + l
    #     b = STHeaviside.apply(z)
    #     # return b if self.stgradient else b.detach()
    #     return b #if self.stgradient else b.detach()


    def reparameterize(self, mu, logsigma):

        #mu = params[:,:int(params.size(1) / 2)]
        #logsigma = params[:,:int(-params.size(1) / 2)]
        #mu = params.select(-1, 0)
        #logsigma = params.select(-1, 1)
        std_z = torch.randn(mu.size()).type_as(mu.data)
        sampled = std_z * torch.exp(logsigma) + mu
        return sampled


    def decode(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_latents, 1, 1)

        x = self.deconv6(x)
        x = self.debn6(x)
        x = self.act(x)
        x = self.deconv5(x)
        x = self.debn5(x)
        x = self.act(x)
        x = self.deconv4(x)
        x = self.debn4(x)
        x = self.act(x)
        x = self.deconv3(x)
        x = self.debn3(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.debn2(x)
        x = self.act(x)
        x = self.deconv1(x)
        x = self.debn1(x)
        x = self.act(x)

        # x = self.deconv0(x)        
        # x = x.view(batch_size, self.num_output_params * self.input_size * self.input_size)
        # return x

        mu = self.deconv0_mu(x)
        logsigma = self.deconv0_logsigma(x)

        return (mu, logsigma)


    def forward(self, x):

        batch_size = x.size(0)
        
        #z_params = self.encode(x).view(batch_size, self.num_latents, self.num_params)
        (z_mu, z_logsigma) = self.encode(x)#.view(batch_size, self.num_latents, self.num_params)
        z_mu = z_mu.view(batch_size, self.num_latents)
        z_logsigma = z_logsigma.view(batch_size, self.num_latents)
        z = self.reparameterize(z_mu, z_logsigma)

        #x_params = self.decode(z).view(batch_size, self.input_size * self.input_size, self.num_output_params)
        (x_mu, x_logsigma) = self.decode(z)#.view(batch_size, self.input_size * self.input_size, self.num_output_params)
        x_mu = x_mu.view(batch_size, self.input_size * self.input_size)
        x_logsigma = x_logsigma.view(batch_size, self.input_size * self.input_size)

        if self.output_dist == 'normal':
            x = self.reparameterize(x_mu, x_logsigma)
            x_mu = x_mu.view(batch_size, self.input_size, self.input_size)
            x_logsigma = x_logsigma.view(batch_size, self.input_size, self.input_size)
            # x = self.act(x)
        elif self.output_dist == 'bernoulli':
            x = self.reparameterize_bernoulli(x_mu)
            # x = self.act(x)
        elif self.output_dist == 'fake_normal':
            x_logsigma = torch.zeros_like(x_logsigma)
            x = self.reparameterize(x_mu, x_logsigma)
            # x = self.act(x)
            x_mu = x_mu.view(batch_size, self.input_size, self.input_size)
            x_logsigma = x_logsigma.view(batch_size, self.input_size, self.input_size)
        else:
            #x = x_params
            x = x_mu

        x = x.view(batch_size, self.input_size, self.input_size)
        return x, (x_mu, x_logsigma), z, (z_mu, z_logsigma), None


# class CNN(nn.Module): 

#     def __init__(self, out_size=21):
#         super(CNN, self).__init__()

#         self.input_size = 128
#         self.out_size = out_size

#         self.act = nn.LeakyReLU(inplace=True)

#         # self.fc1 = nn.Linear(self.in_size, 1000)
#         # self.bn1 = nn.BatchNorm1d(1000)
#         # self.fc2 = nn.Linear(1000, 10000)
#         # self.bn2 = nn.BatchNorm1d(10000)
#         # self.fc3 = nn.Linear(10000, self.out_size*3)

#     def forward(self, x):

#         batch_size = x.size(0)
#         x = x.view(batch_size, 1, self.input_size, self.input_size)
        
#         # x = self.fc1(x)
#         # x = self.bn1(x)
#         # x = self.act(x)

#         # x = self.fc2(x)
#         # x = self.bn2(x)
#         # x = self.act(x)

#         # x = self.fc3(x)

#         x = x.view(batch_size, self.out_size, 3)
#         return None, (None, None), None, (None, None), x

# class FullyConnected(nn.Module): 

#     def __init__(self, vae: VAE, out_size=21):
#         super(FullyConnected, self).__init__()

#         self.vae = vae
#         # self.vae.eval()
#         # self.vae.requires_grad = False
#         # self.vae.requires_grad = False
#         self.out_size = out_size

#         self.act = nn.LeakyReLU(inplace=True)

#         self.fc1 = nn.Linear(self.vae.num_latents, 100)
#         self.bn1 = nn.BatchNorm1d(100)
        
#         self.fc2 = nn.Linear(100, 1000)
#         self.bn2 = nn.BatchNorm1d(1000)

#         self.fc3 = nn.Linear(1000, 1000)
#         self.bn3 = nn.BatchNorm1d(1000)

#         # self.fc4 = nn.Linear(1000, 1000)
#         # self.bn4 = nn.BatchNorm1d(1000)
        
#         self.fc5 = nn.Linear(1000, 100)
#         self.bn5 = nn.BatchNorm1d(100)
        
#         self.fc6 = nn.Linear(100, self.out_size*3)

#     def forward(self, x):

#         batch_size = x.shape[0]

#         with torch.no_grad():
#             self.vae.eval()
#             (z_mu, z_logsigma) = self.vae.encode(x)
#             z_mu = z_mu.view(batch_size, self.vae.num_latents)
#             z_logsigma = z_logsigma.view(batch_size, self.vae.num_latents)
#             z = self.vae.reparameterize(z_mu, z_logsigma)
        
#         x = self.fc1(z)
#         x = self.bn1(x)
#         x = self.act(x)

#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = self.act(x)

#         x = self.fc3(x)
#         x = self.bn3(x)
#         x = self.act(x)

#         # x = self.fc4(x)
#         # x = self.bn4(x)
#         # x = self.act(x)

#         x = self.fc5(x)
#         x = self.bn5(x)
#         x = self.act(x)

#         x = self.fc6(x)

#         x = x.view(batch_size, self.out_size, 3)
#         return None, (None, None), z, (z_mu, z_logsigma), x

class FullyConnected(nn.Module): 

    def __init__(self, in_size=10, out_size=21):
        super(FullyConnected, self).__init__()

        # self.vae = vae
        # self.vae.eval()
        # self.vae.requires_grad = False
        # self.vae.requires_grad = False
        self.out_size = out_size
        self.in_size = in_size

        self.act = nn.LeakyReLU(inplace=True)

        self.fc1 = nn.Linear(self.in_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        
        # self.fc2 = nn.Linear(1000, 1000)
        # self.bn2 = nn.BatchNorm1d(1000)

        # self.fc3 = nn.Linear(1000, 1000)
        # self.bn3 = nn.BatchNorm1d(1000)

        # self.fc4 = nn.Linear(1000, 1000)
        # self.bn4 = nn.BatchNorm1d(1000)
        
        # self.fc5 = nn.Linear(100, 100)
        # self.bn5 = nn.BatchNorm1d(100)
        
        self.fc6 = nn.Linear(100, self.out_size*3)

    def forward(self, x):

        batch_size = x.shape[0]

        # with torch.no_grad():
        #     self.vae.eval()
        #     (z_mu, z_logsigma) = self.vae.encode(x)
        #     z_mu = z_mu.view(batch_size, self.vae.num_latents)
        #     z_logsigma = z_logsigma.view(batch_size, self.vae.num_latents)
        #     z = self.vae.reparameterize(z_mu, z_logsigma)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)

        # x = self.fc2(x)
        # x = self.bn2(x)
        # x = self.act(x)

        # x = self.fc3(x)
        # x = self.bn3(x)
        # x = self.act(x)

        # x = self.fc4(x)
        # x = self.bn4(x)
        # x = self.act(x)

        # x = self.fc5(x)
        # x = self.bn5(x)
        # x = self.act(x)

        x = self.fc6(x)

        x = x.view(batch_size, self.out_size, 3)
        # return None, (None, None), z, (z_mu, z_logsigma), x
        return None, (None, None), None, (None, None), x

class JointModel(nn.Module): 

    def __init__(self, vae: VAE, fc):
        super(JointModel, self).__init__()

        self.vae = vae
        self.fc = fc
        
        self.num_latents = self.vae.num_latents
        self.num_params = self.vae.num_params
        self.input_size = self.vae.input_size

        self.output_dist = self.vae.output_dist
        self.num_output_params = self.vae.num_output_params

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, x):
        return self.vae.decode(x)

    def reparameterize(self, mu, logsigma):
        return self.vae.reparameterize(mu, logsigma)

    def forward(self, x):

        x, (x_mu, x_logsigma), z, (z_mu, z_logsigma), _ = self.vae(x)
        _, (_, _), _, (_, _), joints = self.fc(z)

        return x, (x_mu, x_logsigma), z, (z_mu, z_logsigma), joints
