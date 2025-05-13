import hydra
from omegaconf import DictConfig, OmegaConf
from utils.utils_model import * 


@hydra.main(config_path="./configs/", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  

    if cfg.task == "train_cv":
        train_cv(cfg)
    elif cfg.task == "train_full":
        train_full(cfg)
    elif cfg.task == "test_full":
        test_full(cfg)


if __name__ == "__main__":
    main()

