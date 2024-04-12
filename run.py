from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch

from turngpt.model import TurnGPT
from turngpt.plot_utils import plot_trp


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="runs/TurnGPT/TurnGPT_1fsd1oan/epoch=4_val_loss=2.6804.ckpt",
    )
    parser.add_argument(
        "-t", "--text", type=str, default="hello there how are you doing today"
    )
    args = parser.parse_args()
    return args


turn_list = [
    [
        "Hello there how can i help you",
        "thanks for asking what is the way to the store",
    ],
]

if __name__ == "__main__":

    args = get_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    model = TurnGPT.load_from_checkpoint(args.checkpoint)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    with torch.inference_mode():
        t = model.tokenizer(turn_list)
        out = model(
            torch.tensor(t["input_ids"], device=model.device),
            torch.tensor(t["speaker_ids"], device=model.device),
        )
        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = model.get_trp(out["probs"])

    out = model.string_list_to_trp(turn_list)

    figs = []
    for b in range(out["trp_probs"].shape[0]):
        proj = out["trp_proj"][b].cpu() if "trp_proj" in out else None
        fig_tmp, _ = plot_trp(
            trp=out["trp_probs"][b].detach().cpu(),
            proj=proj,
            # text=out["tokens"][b],
            unk_token=model.tokenizer.unk_token,
            eos_token=model.tokenizer.eos_token,
            plot=False,
        )
        figs.append(fig_tmp)
    plt.show()
