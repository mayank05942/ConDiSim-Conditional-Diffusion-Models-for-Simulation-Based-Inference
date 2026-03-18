import torch

from model_architecture import ReverseDiffusionModel


# Backup: original ConDiSim-sized configs (large model)
# BENCHMARK_CONFIGS = {
#
#     "two_moons": {
#         "theta_dim": 2,
#         "y_dim": 2,
#         "hidden_dim": 64,
#         "num_blocks": 4,
#     },
#     "gaussian_mixture": {
#         "theta_dim": 2,
#         "y_dim": 2,
#         "hidden_dim": 64,
#         "num_blocks": 4,
#     },
#     "gaussian_linear": {
#         "theta_dim": 10,
#         "y_dim": 10,
#         "hidden_dim": 64,
#         "num_blocks": 6,
#     },
#     "gaussian_linear_uniform": {
#         "theta_dim": 10,
#         "y_dim": 10,
#         "hidden_dim": 64,
#         "num_blocks": 6,
#     },
#     "slcp": {
#         "theta_dim": 5,
#         "y_dim": 8,
#         "hidden_dim": 128,
#         "num_blocks": 6,
#     },
#     "slcp_with_distractors": {
#         "theta_dim": 5,
#         "y_dim": 100,
#         "hidden_dim": 128,
#         "num_blocks": 6,
#     },
#     "bernoulli_glm": {
#         "theta_dim": 10,
#         "y_dim": 10,
#         "hidden_dim": 128,
#         "num_blocks": 6,
#     },
#     "bernoulli_glm_raw": {
#         "theta_dim": 10,
#         "y_dim": 100,
#         "hidden_dim": 128,
#         "num_blocks": 6,
#     },
#     "sir": {
#         "theta_dim": 2,
#         "y_dim": 10,
#         "hidden_dim": 64,
#         "num_blocks": 4,
#     },
#     "lotka_volterra": {
#         "theta_dim": 4,
#         "y_dim": 20,
#         "hidden_dim": 128,
#         "num_blocks": 6,
#     },
# }

# New attempt: configs closer to Simformer size (~180k params)
BENCHMARK_CONFIGS = {

    "two_moons": {
        "theta_dim": 2,
        "y_dim": 2,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "gaussian_mixture": {
        "theta_dim": 2,
        "y_dim": 2,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "gaussian_linear": {
        "theta_dim": 10,
        "y_dim": 10,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "gaussian_linear_uniform": {
        "theta_dim": 10,
        "y_dim": 10,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "slcp": {
        "theta_dim": 5,
        "y_dim": 8,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "slcp_with_distractors": {
        "theta_dim": 5,
        "y_dim": 100,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "bernoulli_glm": {
        "theta_dim": 10,
        "y_dim": 10,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "bernoulli_glm_raw": {
        "theta_dim": 10,
        "y_dim": 100,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "sir": {
        "theta_dim": 2,
        "y_dim": 10,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
    "lotka_volterra": {
        "theta_dim": 4,
        "y_dim": 20,
        "hidden_dim": 40,
        "num_blocks": 4,
    },
}
def main() -> None:
    for benchmark, cfg in BENCHMARK_CONFIGS.items():
        theta_dim = cfg["theta_dim"]
        y_dim = cfg["y_dim"]
        hidden_dim = cfg["hidden_dim"]
        num_blocks = cfg["num_blocks"]

        model = ReverseDiffusionModel(
            theta_dim=theta_dim,
            y_dim=y_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("-" * 60)
        print(f"benchmark={benchmark}")
        print(f"theta_dim={theta_dim}, y_dim={y_dim}")
        print(f"hidden_dim={hidden_dim}")
        print(f"num_blocks={num_blocks}")
        print(f"Total parameters     : {total_params}")
        print(f"Trainable parameters : {trainable_params}")


if __name__ == "__main__":
    main()
