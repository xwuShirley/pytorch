import torch
from .optimizer import Optimizer

class Adagradnorm(Optimizer):
# Modified from Adagrad
    """Implements Adagradnorm algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1.0)
        momentum (float, optional): momentum (default: 0.0)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        initial_accumulator_value (float, optional): initial value of the adaptive sum (default: 1e-2)
     .. _AdaGrad stepsizes: Sharp convergence over nonconvex landscapes, from any 
        initialization: https://arxiv.org/pdf/1806.01811.pdf
     .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html

    """

    def __init__(self, params, lr=1.0, momentum=0.0, lr_decay=0, weight_decay=0, initial_accumulator_value=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        defaults = dict(lr=lr, momentum=momentum,lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(Adagradnorm, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                grad = p.data
                state = self.state[p]
                state['step'] = 0
                if len(grad.size())==4:
                    state['sum'] = torch.ones_like(grad.view(grad.size()[0],-1)).mul_(initial_accumulator_value)
                else:
                    state['sum'] = torch.ones_like(grad).mul_(initial_accumulator_value)
                if group['momentum'] > 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

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
                state = self.state[p]
                state['step'] += 1
                
                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                # grad.is_sparse TODO not yet implement
                
                if  len(grad.size()) == 4:
                    output_channel, input_channel, filter_h, filter_w = grad.size()
                    # Note you could also try:
                    # grad_sqaure = grad.view(output_channel*input_channel,-1).norm(dim=1,p=2)**2
                    grad_sqaure = grad.view(output_channel,-1).norm(dim=1,p=2)**2
                    state['sum'].add_(grad_sqaure.unsqueeze(1))
                elif len(grad.size()) == 2:
                    output_channel, input_channel = grad.size()
                    grad_sqaure = grad.norm(dim=1,p=2)**2
                    state['sum'].add_(grad_sqaure.unsqueeze(1))
                else:
                    ##  a tricky way to find the learning rate to update the bias term
                    try:
                        state['sum'].add_(grad_sqaure.view(grad.size()[0],-1).mean(dim=1))
                        state['sum'].addcmul_(1.0, grad, grad)
                    except:
                        state['sum'].addcmul_(1.0, grad, grad)


                std = state['sum'].sqrt().add_(1e-10).view(*grad.shape)

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(1-group['momentum'], grad)
                    p.data.addcdiv_(-clr, buf, std)
                else:
                    p.data.addcdiv_(-clr, grad, std)

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    p.data.add_(-clr*group['weight_decay'], grad)

        return loss
