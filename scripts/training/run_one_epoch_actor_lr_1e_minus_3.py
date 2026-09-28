from one_epoch_runner_common import run_one_epoch_runner


RUN_NAME = "one_epoch_actor_lr_1e_minus_3"
RUN_DIR = "runs/one_epoch_actor_lr_1e_minus_3"
LR_LABEL = "1e-3"
ACTOR_LR = 1.0e-3


if __name__ == "__main__":
    run_one_epoch_runner(
        run_name=RUN_NAME,
        run_dir=RUN_DIR,
        lr_label=LR_LABEL,
        actor_lr=ACTOR_LR,
    )
