import torch

from abc import ABC, abstractmethod

class BaseImgGenModel(ABC):
    '''
    Base class for image generation models. 
    All image generation models should inherit from this class and implement the abstract methods.
    '''
    def __init__(self, config):
        self.config = config
        self.model = None          # 子类在 super().__init__() 后自行赋值
        self.device = torch.device(config.get('device', 'cuda'))

    def train(self, dataloader):
        self._train_step(dataloader)
        return self                    # 支持链式 save_ckpt
    
    def inference(self, z, save_path=None):
        img = self._inference_step(z)  # 子类实现
        if save_path:
            self._save_image(img, save_path)
        return img

    # === 统一出口 ===
    def save_ckpt(self, path):
        torch.save({'model': self.model.state_dict(), 'config': self.config}, path)

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
    
    # 需要强制子类实现的方法
    @abstractmethod
    def _train_step(self, dataloader): 
        raise NotImplementedError

    @abstractmethod
    def _inference_step(self, z): 
        raise NotImplementedError

