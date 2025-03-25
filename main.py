import hydra
from omegaconf import DictConfig, OmegaConf
from utils_model import * 


@hydra.main(config_path="/scratch_tmp/users/k21066795/BreastAgeNet/configs/", config_name="config")
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

