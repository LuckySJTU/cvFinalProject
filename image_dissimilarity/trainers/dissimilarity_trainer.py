import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append("..")
from image_dissimilarity.util import trainer_util
from image_dissimilarity.models.dissimilarity_model import DissimNet, DissimNetPrior

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='sum',ignore_index=-100,weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index=ignore_index
        self.weight=weight

    def reduce_loss(self, loss, reduction='sum'):
        return loss.mean() if reduction == 'sum' else loss.sum() if reduction == 'sum' else loss

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction,weight=self.weight,ignore_index=self.ignore_index)
        return self.linear_combination(loss / n, nll, self.epsilon)

# class LabelSmoothingCrossEntropy(nn.Module):
#     """
#     NLL loss with label smoothing.
#     """
#     def __init__(self, smoothing=0.1,ignore_index=-100,weight=None):
#         """
#         Constructor for the LabelSmoothing module.
#         :param smoothing: label smoothing factor
#         """
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         assert smoothing < 1.0
#         self.smoothing = smoothing
#         self.confidence = 1. - smoothing
#         self.ignore_index=ignore_index
#         self.weight=weight

#     def forward(self, x, target):
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()


class DissimilarityTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, config, seed=0):
        
        trainer_util.set_seed(seed)
        
        cudnn.enabled = True
        self.config = config
        
        if config['gpu_ids'] != "-1":
            self.gpu = 'cuda'
        else:
            self.gpu = None
        
        if config['model']['prior']:
            self.diss_model = DissimNetPrior(**config['model']).cuda(self.gpu)
        elif 'vgg' in config['model']['architecture']:
            self.diss_model = DissimNet(**config['model']).cuda(self.gpu)
        else:
            raise NotImplementedError()

        # get pre-trained model
        pretrain_config = config['diss_pretrained']
        if pretrain_config['load']:
            epoch = pretrain_config['which_epoch']
            save_ckpt_fdr = pretrain_config['save_folder']
            ckpt_name = pretrain_config['experiment_name']

            print('Loading pretrained weights from %s (epoch: %s)' % (ckpt_name, epoch))
            model_path = os.path.join(save_ckpt_fdr, ckpt_name, '%s_net_%s.pth' % (epoch, ckpt_name))
            model_weights = torch.load(model_path)
            self.diss_model.load_state_dict(model_weights, strict=False)
            # NOTE: For old models, there were some correlation weights created that were not used in the foward pass. That's the reason to include strict=False
            
        print('Printing Model Parameters')
        print(self.diss_model.parameters)
        
        lr_config = config['optimizer']
        lr_options = lr_config['parameters']
        if lr_config['algorithm'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.diss_model.parameters(), lr=lr_options['lr'],
                                             weight_decay=lr_options['weight_decay'],)
        elif lr_config['algorithm'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.diss_model.parameters(),
                                              lr=lr_options['lr'],
                                              weight_decay=lr_options['weight_decay'],
                                              betas=(lr_options['beta1'], lr_options['beta2']))
        else:
            raise NotImplementedError
        
        if lr_options['lr_policy'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=lr_options['patience'], factor=lr_options['factor'])
        else:
            raise NotImplementedError
        
        self.old_lr = lr_options['lr']
        
        if config['training_strategy']['class_weight']:
            if not config['training_strategy']['class_weight_cityscapes']:
                if config['train_dataloader']['dataset_args']['void']:
                    label_path = os.path.join(config['train_dataloader']['dataset_args']['dataroot'], 'labels_with_void_no_ego/')
                else:
                    label_path = os.path.join(config['train_dataloader']['dataset_args']['dataroot'], 'labels/')
                    
                full_loader = trainer_util.loader(label_path, batch_size='all')
                print('Getting class weights for cross entropy loss. This might take some time.')
                class_weights = trainer_util.get_class_weights(full_loader, num_classes=2)
            else:
                if config['train_dataloader']['dataset_args']['void']:
                    class_weights = [1.54843156, 8.03912212]
                else:
                    class_weights = [1.46494611, 16.5204619]
            print('Using the following weights for each respective class [0,1]:', class_weights)
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, weight=torch.FloatTensor(class_weights).to("cuda")).cuda(self.gpu)
            # self.criterion=LabelSmoothingCrossEntropy(ignore_index=255, weight=torch.FloatTensor(class_weights).to("cuda")).cuda(self.gpu)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(self.gpu)
            # self.criterion=LabelSmoothingCrossEntropy(ignore_index=255).cuda(self.gpu)
        
    def run_model_one_step(self, original, synthesis, semantic, label):
        self.optimizer.zero_grad()
        predictions = self.diss_model(original, synthesis, semantic)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
        # print(model_loss)
        model_loss.backward()
        self.optimizer.step()
        self.model_losses = model_loss
        self.generated = predictions
        return model_loss, predictions
        
    def run_validation(self, original, synthesis, semantic, label):
        predictions = self.diss_model(original, synthesis, semantic)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
        return model_loss, predictions

    def run_model_one_step_prior(self, original, synthesis, semantic, label, entropy, mae, distance):
        self.optimizer.zero_grad()
        predictions = self.diss_model(original, synthesis, semantic, entropy, mae, distance)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
        model_loss.backward()
        self.optimizer.step()
        self.model_losses = model_loss
        self.generated = predictions
        return model_loss, predictions

    def run_validation_prior(self, original, synthesis, semantic, label, entropy, mae, distance):
        self.diss_model.cuda(self.gpu)
        predictions = self.diss_model(original.cuda(), synthesis.cuda(), semantic.cuda(), entropy.cuda(), mae.cuda(), distance.cuda())
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda(self.gpu))
        return model_loss, predictions

    def get_latest_losses(self):
        return {**self.model_loss}

    def get_latest_generated(self):
        return self.generated

    def save(self, save_dir, epoch, name):
        if not os.path.isdir(os.path.join(save_dir, name)):
            os.mkdir(os.path.join(save_dir, name))
        
        save_filename = '%s_net_%s.pth' % (epoch, name)
        save_path = os.path.join(save_dir, name, save_filename)
        torch.save(self.diss_model.state_dict(), save_path)  # net.cpu() -> net

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.config['training_strategy']['niter']:
            lrd = self.config['optimizer']['parameters']['lr'] / self.config['training_strategy']['niter_decay']
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
            
    def update_learning_rate_schedule(self, val_loss):
        self.scheduler.step(val_loss)
        lr = [group['lr'] for group in self.optimizer.param_groups][0]
        print('Current learning rate is set for %f' %lr)

if __name__ == "__main__":
    import yaml
    
    config = '../configs/default_configuration.yaml'
    gpu_ids = 0

    with open(config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config['gpu_ids'] = gpu_ids
    trainer = DissimilarityTrainer(config)
