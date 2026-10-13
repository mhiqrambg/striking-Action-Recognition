from src import load_config
from src.data import process_all_videos
from src.training.trainer import train, train_final_model


def main():
    cfg = load_config("configs/default.yaml")
    process_all_videos(cfg, save_skeleton_video=False)
    summary = train(cfg, scenario_name="B", use_learnable_pe=True, use_focal_loss=True)
    print(summary)
    deploy_path = train_final_model(cfg, scenario_name="B", use_learnable_pe=True, use_focal_loss=True)
    print(f"Deployable model saved to: {deploy_path}")


if __name__ == "__main__":
    main()
