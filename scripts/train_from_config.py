import os
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


os.environ['HYDRA_FULL_ERROR'] = "1"
@hydra.main(config_path="../configs", config_name="config_fsav994", version_base=None)
def main(cfg:DictConfig):
    pl.seed_everything(333) # seed for reproducibility

    ## DATA ##
    dm = hydra.utils.call(cfg.datamodule)
    dm.setup("train")
    # train_dl, val_dl = dm.train_dataloader(), dm.val_dataloader()
    # print(f"{next(iter(val_dl))[0].shape=}")

    litmodel = hydra.utils.call(cfg.litmodel)
    for p in litmodel.l2o.reg_net.parameters(): 
        print(p.mean())
    print(litmodel)
    # for k, p in litmodel.named_parameters():
    #     print(f"{k}: {p.numel()}")
    # print(sum(p.numel() for p in litmodel.parameters() if p.requires_grad))

    trainer = hydra.utils.call(cfg.trainer)
    trainer.fit(litmodel, dm)

if __name__ == "__main__": 
    main()