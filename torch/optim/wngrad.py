import torch
from .optimizer import Optimizer


class Wngrad(Optimizer):
# Modified from Adam
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        initial_accumulator_value (float, optional): initial value of the adaptive sum (default: 1.0)

    WNGrad: Learn the Learning Rate in Gradient Descent: https://arxiv.org/abs/1803.02865
    """

    def __init__(self, params, lr=1e-2,  eps=1e-8, weight_decay=0,momentum=0,initial_accumulator_value=1.0):
        defaults = dict(lr=lr, momentum=momentum,  eps=eps, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value)
        super(Wngrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Wngrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        #print self.param_groups
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                #print grad
                if grad.is_sparse:
                    raise RuntimeError('WNGrad does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if len(grad.size())==4:
                        state['square_wn'] = torch.ones_like(p.data.view(grad.size()[0]*grad.size()[1],-1)).mul_(initial_accumulator_value)
                    else:
                        state['square_wn'] = torch.ones_like(p.data).mul_(initial_accumulator_value)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                square_var = torch.zeros_like(p.data)
                square_wn = state['square_wn']
                sqrlr = group['lr']**2
                check = group['check']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                if  len(grad.size())==4:
                    # Note you could aslo try this:
                    # grad_expand = grad.view(grad.size()[0],-1)
                    grad_expand = grad.view(grad.size()[0]*grad.size()[1],-1)
                    grad_sqaure = grad_expand.norm(dim=1,p=2)**2
                    square_wn.addcdiv_(sqrlr,grad_sqaure.unsqueeze(1),square_wn)
                    wn = square_wn.view(*grad.shape)
                elif  len(grad.size())==2:
                    grad_sqaure = grad.norm(dim=1,p=2)**2
                    square_wn.addcdiv_(sqrlr,grad_sqaure.unsqueeze(1),square_wn)
                    wn = square_wn
                else:
                    wn = square_wn

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, wn)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, wn.add_(group['eps']))
        return loss
