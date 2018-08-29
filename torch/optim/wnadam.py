import torch
from .optimizer import Optimizer


class  Wnadam(Optimizer):
# Modified from Adam
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1.0)
        momentum (float, optional): momentum factor (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        initial_accumulator_value (float, optional): initial value of the adaptive sum (default: 1.0)

    WNGrad: Learn the Learning Rate in Gradient Descent: https://arxiv.org/abs/1803.02865
    """

    def __init__(self, params, lr=1.0, beta=0.9, eps=1e-8,
                 weight_decay=0,  initial_accumulator_value=1.0， amsgrad=False):
        defaults = dict(lr=lr, beta=beta, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, initial_accumulator_value=initial_accumulator_value)
        super(Wnadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('WNAdam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    if len(grad.size())==4:
                        state['square_wn'] = torch.ones_like(p.data.view(grad.size()[0]*grad.size()[1],-1)).mul_(initial_accumulator_value)
                    else:
                        state['square_wn'] = torch.ones_like(p.data).mul_(initial_accumulator_value)#
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta = group['beta']

                state['step'] += 1
                check = group['check']
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta).add_(1 - beta, grad)
                square_wn = state['square_wn']
                if  len(grad.size())==4:
                    grad_expand = grad.view(grad.size()[0]*grad.size()[1],-1)
                    grad_sqaure = grad_expand.norm(dim=1,p=2)**2
                    square_wn.addcdiv_(group['lr']**2,grad_sqaure.unsqueeze(1),square_wn)
                    wn = square_wn.view(*grad.shape)
                elif  len(grad.size())==2:
                    grad_sqaure = grad.norm(dim=1,p=2)**2
                    square_wn.addcdiv_(group['lr']**2,grad_sqaure.unsqueeze(1),square_wn)
                    wn = square_wn
                else:
                    wn = square_wn

                exp_avg_sq = wn

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq#.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq#.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta ** state['step']
                step_size = group['lr']  / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
