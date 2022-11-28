import os
from pathlib import Path

import torch.multiprocessing as mp

from main_pretrain import get_args_parser, main

def multigpu_main(rank, args):
    os.environ["RANK"] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    main(args)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Training with {args.world_size} GPUs")
    os.environ['WORLD_SIZE'] = str(args.world_size)
    mp.spawn(multigpu_main, args=(args,), nprocs=args.world_size)