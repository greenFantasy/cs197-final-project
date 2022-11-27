## Pre-training MAE on Chest Xrays
## Setup
### Creating the AMI
- Create AWS account and ask for project member to share AMI with your AWS id
- Go to EC2 and Launch Instance; then select "Browse more AMIs" option
- Set region to US East (N. Virginia)
- Click "My AMIs" and select the checkbox that reads "Shared with Me"
- Set instance type to g4dn.xlarge
- Set a key pair for this AMI (create a pair if necessary)
- Set storage to 1 x 1600 with gp3 as the storage type
- Then hit launch instance

### Connecting to AMI with VSCode via SSH
- Install the Remote SSH extension on VS Code
- In your command palette, select the option "Remote-SSH: Add new SSH host"
- Now run the command `ssh -i "[key pair path]" ec2-user@[Public IPv4 DNS]` where key pair path is your local path to the key pair that's associated with this AMI and the DNS is listed under Instance Summary on AWS
Connect to the instance in VSCode, and from that directory, clone the following repo: https://github.com/greenFantasy/cs197-final-project

Now, use conda to create a new environment:

```
conda update -n base conda
conda create --name=mae python=3.9
```

Now activate the new environment, we will now set up the dependencies

Run:

```
source activate mae
conda install pytorch torchvision -c pytorch
pip install tensorboard
conda update pillow
conda install six
conda install -c conda-forge timm=0.3.2
```

Now, we have to make a change to the timm package because it has an error. Run

```
python main_pretrain.py
```

And the stacktrace will point to a file that is erroring in the form of: `…/timm/models/layers/helpers.py`. Use vim to go into this file, and edit line 6 (containing the import of container_abcs) to the following:

```
import collections.abc as container_abcs
```

Now, you are ready to run! Retry running:

```
python main_pretrain.py
```

And we are training a masked autoencoder!

### Multi-GPU Training

We do a lot of our training on multiple GPUs on a single node (e.g. AWS p3.8xlarge instances). If you have multiple GPUs on your machine, you can use the `multigpu_pretrain.py` training script we developed. You can just run:

```
torchrun multigpu_pretrain.py --world_size=[# of gpus on node]
```

and you are now training using all the GPUs! You can confirm with the single GPU case that this runs a lot faster.