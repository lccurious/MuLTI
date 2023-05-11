import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from multi.dataset import vector_autoregression
from multi.models import multi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent Processes Identification from Multi-view Data - MLP Mixing"
    )
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=10,
        help="Number of batches to average evaluation performance at the end.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--act-fct",
        type=str,
        default="leaky_relu",
        help="Activation function in mixing network g.",
    )
    parser.add_argument(
        "--c-param",
        type=float,
        default=0.05,
        help="Concentration parameter of the conditional distribution.",
    )
    parser.add_argument(
        "--m-param",
        type=float,
        default=1.0,
        help="Additional parameter for the marginal (only relevant if it is not uniform).",
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--n-mixing-layer",
        type=int,
        default=3,
        help="Number of layers in nonlinear mixing network g.",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Dimensionality of the latents."
    )
    parser.add_argument(
        "--num-views", type=int, default=2,
        help="Number of different View"
    )
    parser.add_argument(
        "--n-shared", type=int, default=4,
        help="Dimensionality of the shared between views"
    )
    parser.add_argument(
        "--num-regimes", type=int, default=1,
        help="Number of different regimes"
    )
    parser.add_argument(
        "--causal-type", type=str, default="linear", choices=("linear", "nonlinear")
    )
    parser.add_argument(
        "--causal-con-order", type=int, default=1, choices=(0, 1, 2)
    )
    parser.add_argument(
        "--time-lag", type=int, default=2, help="Number of time lags for auto-regression."
    )
    parser.add_argument(
        "--length", type=int, default=8, help="Length of generated time series"
    )
    parser.add_argument(
        "--space-type", type=str, default="box", choices=("box", "sphere", "unbounded")
    )
    parser.add_argument(
        "--m-p",
        type=int,
        default=0,
        help="Type of ground-truth marginal distribution. p=0 means uniform; "
        "all other p values correspond to (projected) Lp Exponential",
    )
    parser.add_argument(
        "--c-p",
        type=int,
        default=2,
        help="Exponent of ground-truth Lp Exponential distribution.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--p",
        type=int,
        default=2,
        help="Exponent of the assumed model Lp Exponential distribution.",
    )
    parser.add_argument("--batch-size", type=int, default=3200)
    parser.add_argument("--n-log-steps", type=int, default=2500)
    parser.add_argument("--n-steps", type=int, default=100001)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--beta1", type=float, default=0.01)
    parser.add_argument("--beta2", type=float, default=0.01)
    parser.add_argument("--beta3", type=float, default=0.01)
    args = parser.parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    return args


def cli_main():

    # Parse command line arguments
    args = parse_args()
    seed_everything(args.seed, workers=True)
    
    # Data parameters
    var_datamodule = vector_autoregression.LinearVAR(args)

    # Model parameters
    model = multi.MuLTI(args)

    # Training parameters
    logger = TensorBoardLogger(f"{args.save_dir}/tensorboard", 
                               default_hp_metric=False,
                               name="MuLTI")
    trainer = Trainer(deterministic=True,
                      logger=logger,
                      default_root_dir=args.save_dir,
                      max_steps=args.n_steps,
                      accelerator='gpu', devices=1)
    trainer.fit(model, datamodule=var_datamodule)


if __name__ == '__main__':
    cli_main()
