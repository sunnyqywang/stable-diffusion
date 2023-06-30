import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", supervision_factor=1.0, nll_factor=1.0):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.supervision_factor = supervision_factor
        self.nll_factor = nll_factor

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def calculate_adaptive_weight2(self, nll_loss, g_loss, last_layer1=None, last_layer2=None):
        g_grads = torch.autograd.grad(g_loss, last_layer2, retain_graph=True)[0]
        nll_grads = torch.autograd.grad(nll_loss, last_layer1, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors=None, optimizer_idx=None,
                global_step=None, supervisions=None, last_layer=None, supervisor_layer=None, cond=None, split="train",
                weights=None):
        image_dim = reconstructions.shape[1]
        inputs = inputs[:,:image_dim,:,:]
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        if self.nll_factor > 0:
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        else:
            weighted_nll_loss = torch.sum(rec_loss) / rec_loss.shape[0]
            nll_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        if self.kl_weight > 0:
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        else:
            kl_loss = torch.tensor(0.0)

        if self.supervision_factor > 0.0:
            assert supervisor_layer is not None

            s_loss = torch.abs(supervisions[0].contiguous()-supervisions[1].contiguous())
            s_loss = torch.sum(s_loss) / s_loss.shape[0]
            try:
                s_weight = self.calculate_adaptive_weight2(nll_loss, s_loss,
                     last_layer1=last_layer, last_layer2=supervisor_layer)
            except:
                s_weight=torch.tensor(0.0)
        else:
            s_weight = torch.tensor(0.0)
            s_loss = torch.tensor(0.0)

        if self.disc_factor > 0:
            # now the GAN part
            if optimizer_idx == 0:
                # generator update
                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(reconstructions.contiguous())
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                g_loss = -torch.mean(logits_fake)

                if self.disc_factor > 0.0:
                    try:
                        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                        # print(d_weight)
                        # nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
                    except RuntimeError:
                        assert not self.training
                        d_weight = torch.tensor(0.0)
                else:
                    d_weight = torch.tensor(0.0)

                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                loss = (1-self.supervision_factor) * (weighted_nll_loss + self.kl_weight * kl_loss + \
                       d_weight * disc_factor * g_loss) + \
                       self.supervision_factor * s_weight * s_loss

                log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                       "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                       "{}/rec_loss".format(split): rec_loss.detach().mean(),
                       "{}/d_weight".format(split): d_weight.detach(),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/g_loss".format(split): g_loss.detach().mean(),
                       "{}/s_weight".format(split): s_weight.detach().mean(),
                       "{}/s_loss".format(split): s_loss.detach().mean(),
                       }
                return loss, log

            if optimizer_idx == 1:
                # second pass for discriminator update
                if cond is None:
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(reconstructions.contiguous().detach())
                else:
                    logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                       "{}/logits_real".format(split): logits_real.detach().mean(),
                       "{}/logits_fake".format(split): logits_fake.detach().mean()
                       }
                return d_loss, log
        else:
            d_weight = torch.tensor(0.0)
            disc_factor = torch.tensor(0.0)
            g_loss = torch.tensor(0.0)

        loss = (1 - self.supervision_factor) * (weighted_nll_loss + self.kl_weight * kl_loss + \
                d_weight * disc_factor * g_loss) + \
                self.supervision_factor * s_weight * s_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/logvar".format(split): self.logvar.detach(),
               "{}/kl_loss".format(split): kl_loss.detach().mean(),
               "{}/nll_loss".format(split): nll_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               "{}/d_weight".format(split): d_weight.detach(),
               "{}/disc_factor".format(split): torch.tensor(disc_factor),
               "{}/g_loss".format(split): g_loss.detach().mean(),
               "{}/s_weight".format(split): s_weight.detach().mean(),
               "{}/s_loss".format(split): s_loss.detach().mean(),
               }

        return loss, log