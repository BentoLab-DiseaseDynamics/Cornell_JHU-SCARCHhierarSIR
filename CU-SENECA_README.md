# Running on Cornell Seneca cluster

Read the docs: https://portal.cac.cornell.edu/TechDocs/clusters/seneca/#ssh

## Getting set up

### Setting up an account and tunneling into the Seneca cluster

1. Set up a CAC account to access the CAC portal.

2. In the "Request or Join a CAC Project" tab, ask permission to be added to project "Bento Lab Projects" by user "arb24_0001".

3. In the "Manage your CAC login credentials" tab, generate an SSH keypair. For Mac or Linux users, copy the private key to your `~/.ssh` folder.

4. Tunnel into Seneca cluster

```bash
ssh -i <path to the private key file> <NetID>@seneca-login1.cac.cornell.edu
```

On Mac, make an alias for this login by making a file `config` in your `~/.ssh` folder, in which you type,

```
Host cornell_seneca
    HostName seneca-login1.cac.cornell.edu
    User <netID>
    IdentityFile ~/.ssh/<private_key>
```

you can then tunnel into the Seneca cluster more simply,

```bash
ssh cornell_seneca
```

### Coupling Github to the Seneca cluster

1. Generate an ssh key on the Seneca cluster. When prompted to "Enter file in which to save the key", give the key a descriptive name like `github` and press Enter. The key will be saved in the `~/.ssh/` folder.

```bash
ssh-keygen -t ed25519 -C "<netID>@cornell.edu"
```

2. Display the public key and copy it's contents to the clipboard.

```bash
cat ~/.ssh/github.pub
```

3. Register the public key in your personal GH. Go to your personal GH account > Settings > SSH and GPG keys > Add new SSH Key and paste the public key.

4. Start the SSH agent and test the connection.

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github
```

Followed by,

```bash
ssh -T git@github.com
```

Which should print: "Hi <user>! You've successfully authenticated, but GitHub does not provide shell access.".

### Setting up the model and environment

1. Clone the repository you want to work in.

```bash
git clone git@github.com:BentoLab-DiseaseDynamics/Cornell_JHU-SCARCHhierarSIR.git
```

2. Install the conda environment. Note that you CANNOT pull packages from the `defaults` channel, you must use only `conda-forge`.

```bash
module load anaconda3
conda env create -f SCARCHhierarSIR_env.yml
```

3. Install the model inside the conda environment.

```bash
source /opt/ohpc/pub/software/anaconda3/etc/profile.d/conda.sh
conda activate BENTOLAB-SCARCH_HIERARSIR
unset PYTHONHOME
unset PYTHONPATH
pip install -e .
```

### Submitting jobs

An example job submission script is in the `~/scripts/operational` folder.

### Cluster Tips and Tricks

1. Monitor your job's progress:

```bash
squeue -u <netID>
```

2. Cancel your job:

```bash
scancel <job_ID>
```

3. See a history of your jobs:

```bash
sacct -u <netID> --format=JobID,State,NodeList,Elapsed,TotalCPU
```

4. Reset git repository on cluster to latest version:

```bash
git reset --hard && git clean -f -d
git pull origin
```

You might have to repeat,

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github
```

5. Copy files from the HPC to local computer (Linux/Mac):

    - Open a terminal where you want to place the files on your computer.
    - Run ```scp -r cornell_seneca:/<path/to/file> .``` (assuming you have made an alias for the cluster named `cornell_seneca` as outlined before)
    - If connections time out during a secure copy you can try adding the options: ```-o ServerAliveInterval=30 -o ServerAliveCountMax=10```
    - Run ```find . -type f -name "*.<ext>" -delete``` to find an remove large files with extension .<ext> to make downloading faster.
    
