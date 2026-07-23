## Running on Cornell Seneca cluster

### Setting up an account and loging in

1. Login to Seneca

```bash
ssh -i <path to the private key file> <NetID>@seneca-login1.cac.cornell.edu
```

### Submitting jobs

### Cluster Tips and Tricks

1. Monitor your job's progress:

```bash
squeue -u your_username
scancel --name=your_job_name
```

2. Cancel your job:

```bash
scancel <job_ID>
```

3. Reset git repository on cluster to latest version:

```bash
git reset --hard && git clean -f -d
git pull origin
```

4. Copy files from the HPC to local computer:

    - Open a terminal where you want to place the files on your computer.
    - Run ```scp -r <NetID>@seneca-login1.cac.cornell.edu:/<path/to/file> .```
    - If connections time out during a secure copy you can try adding the options: ```-o ServerAliveInterval=30 -o ServerAliveCountMax=10```
    - Run ```find . -type f -name "*.hdf5" -delete``` to find an remove .hdf5 backends to make downloading faster.
    
