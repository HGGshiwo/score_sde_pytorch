from torchrl.trainers import Trainer, ReplayBufferTrainer,  LogReward
import torch.nn as nn
import torch
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor, ActorValueOperator, NormalParamExtractor
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict.nn import TensorDictModule
from dataclasses import dataclass, asdict
from torch.optim import Adam
from env import create_env
from torchrl.record.loggers.tensorboard import TensorboardLogger
from datetime import datetime
import os

class Encoder(nn.Module):
    def __init__(self, img_size: int=32, model_dim: int=128):
        super().__init__()
        self.emb = nn.Embedding(2, model_dim)
        cnn, input_dim = [], 3
        while img_size != 2:
            cnn.append(nn.Conv2d(input_dim, input_dim*2, 3, 1, 1))
            cnn.append(nn.ReLU())
            cnn.append(nn.MaxPool2d(2))
            img_size //= 2
            input_dim *= 2
     
        self.cnn = nn.Sequential(
            *cnn,
            nn.Flatten(),
            nn.Linear(input_dim*4, model_dim)
        )
        self.res1 = nn.Linear(model_dim, model_dim)
        self.res2 = nn.Linear(model_dim, model_dim)

        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm1d(model_dim)

    def forward(self, z, error_norm, step_rejected):
        error_norm = self.bn1(error_norm[:, None])
        emb = self.emb(step_rejected.long()) * error_norm
        z = self.bn2(z)
        z = self.cnn(z)
        z = self.res1(z + emb)
        z = nn.functional.relu(z)
        z = self.res2(z + emb)
        z = nn.functional.relu(z)
        return z
    
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, extract_norm=False) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.norm_extractor = NormalParamExtractor() if extract_norm else nn.Identity()

    def forward(self, hidden):
        x = self.bn(hidden)
        x = self.fc(x)
        x = self.norm_extractor(x)
        return x
    
def create_ac(config):
    encoder = TensorDictModule(
        module=Encoder(config.img_size, config.model_dim).to(cfg.device), 
        in_keys=["z", "error_norm", "step_rejected"], 
        out_keys=["hidden"]
    )
    actor = TensorDictModule(
        module=Decoder(config.model_dim, 2, extract_norm=True).to(cfg.device), 
        in_keys=["hidden"], 
        out_keys=["loc", "scale"]
    )
    critic = TensorDictModule(
        module=Decoder(config.model_dim, 1).to(cfg.device),
        in_keys=["hidden"],
        out_keys=["state_value"]
    )
    policy = ProbabilisticActor(
        actor,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=Normal,
        return_log_prob=True,
    )
    return ActorValueOperator(encoder, policy, critic)

@dataclass
class Config:
    total_frames: int = 1e6
    n_optim: int = 4
    env_num: int = 128
    device: torch.device = torch.device('cuda')
    log_interval: int = 1e4
    save_interval: int = 1e4
    batch_size: int = 32
    exp_name: str|None = None
    img_size: int = 32
    model_dim: int = 128

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)


if __name__ == "__main__":
    
    kwargs = {
        "total_frames": 1e6,
        "n_optim": 4,
        "env_num": 128,
        "device": torch.device('cuda'),
        "log_interval": 1,
        "save_interval": 1000, 
        "batch_size": 32,
        "exp_name": None,
        "img_size": 32,
        "model_dim": 128    
    }
    cfg = Config.from_dict(kwargs)

    ac = create_ac(cfg)
    loss_module = ClipPPOLoss(
        ac.get_policy_operator(), 
        ac.get_value_operator(),
        normalize_advantage=True,
    )

    env = create_env(cfg)
    collector = SyncDataCollector(
        frames_per_batch=cfg.env_num,
        create_env_fn=env,
        policy=ac.get_policy_operator(),
        device=cfg.device,
    )
    optimizer = Adam(loss_module.parameters())

    replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=1e4))
    rb_trainer = ReplayBufferTrainer(
        replay_buffer=replay_buffer, 
        batch_size=cfg.batch_size,
        device=cfg.device,
        flatten_tensordicts=True
    )

    exp_name = cfg.exp_name
    if cfg.exp_name is None:
        exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cfg.exp_name = exp_name
        parent_dir = os.path.join("exp/rl", exp_name)
        os.makedirs(parent_dir, exist_ok=True)    
        import yaml
        with open(os.path.join("exp/rl", exp_name, "config.yaml"), "w") as f:
            yaml.dump(asdict(cfg), f)
            
    logger = TensorboardLogger(
        exp_name=exp_name,
        log_dir=os.path.join("exp/rl", exp_name, "log"),
    )

    trainer = Trainer(
        collector=collector,
        total_frames=cfg.total_frames,
        frame_skip=1,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        optim_steps_per_batch=cfg.n_optim,
        log_interval=cfg.log_interval,
        save_trainer_file=os.path.join("exp/rl", exp_name, "trainer.pt"),
        save_trainer_interval=cfg.save_interval,
        clip_norm=1.0,
    )

    if os.path.exists(trainer.save_trainer_file):
        trainer.load_from_file(trainer.save_trainer_file)


    trainer.register_op("batch_process", rb_trainer.extend)
    trainer.register_op("process_optim_batch", rb_trainer.sample)
    trainer.register_op("post_loss", rb_trainer.update_priority)
    
    log_reward = LogReward(reward_key=("next", "reward"))
    trainer.register_op("pre_steps_log", log_reward)
    
    try:
        trainer.train() 
    except KeyboardInterrupt:
        trainer.save_trainer(force_save=True)
        