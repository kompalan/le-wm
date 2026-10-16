# AGENTS.md

## Remote compute (Vast.ai)

> [!WARNING]
> **FUNDS ARE LIMITED. EVERY MINUTE OF GPU TIME COSTS REAL MONEY.**
>
> - **Use compute SPARINGLY and EFFECTIVELY.** Do not rent a GPU for anything that can be done locally (editing code, debugging imports, small smoke tests, plotting, reading logs).
> - **DESTROY THE INSTANCE IMMEDIATELY when the experiment is done**, including when it fails or crashes. Do not leave instances idle "just in case."
> - **NEVER have more than ONE instance rented at a time.** Before creating an instance, check that none exist. If one already exists, reuse it or destroy it first. Never launch parallel instances.

### Before renting

1. Make sure the code runs locally first (e.g. a few steps on CPU/MPS with a tiny config). Do not use paid GPU time to find bugs that a local run would find.
2. Have a clear plan: which command(s) to run, roughly how long they take, and what outputs/checkpoints need to be copied back.
3. Confirm no instance is already running:
   ```bash
   vastai show instances --raw   # must return []
   ```

### Choosing an offer

Pick the **cheapest** machine that can do the job. Do not over-provision GPU count, VRAM, or disk.

```bash
vastai search offers 'num_gpus=1 gpu_ram>=24 reliability>0.98 inet_down>200 direct_port_count>=1' -o 'dph+' --limit 10
```

`-o 'dph+'` sorts by price per hour, cheapest first. Change the filters only as far as the experiment actually needs.

### Renting

```bash
vastai create instance <offer_id> \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --disk <GB> --ssh --direct --label <short-label> --raw
```

Poll until `actual_status` is `running`, then wait ~15s before connecting:

```bash
vastai show instance <instance_id> --raw
```

### Connecting

The SSH key is `~/.ssh/claudes_key` (registered on Vast.ai). Get `ssh_host` and `ssh_port` from `vastai show instance <id> --raw`:

```bash
ssh -i ~/.ssh/claudes_key -o BatchMode=yes -p <ssh_port> root@<ssh_host>
```

Use `scp -i ~/.ssh/claudes_key -P <ssh_port> ...` (or `rsync -e "ssh -i ~/.ssh/claudes_key -p <ssh_port>"`) to move code and results.

### While running

- Run long jobs under `nohup` or `tmux` so an SSH disconnect does not kill them, and write logs to a file.
- Check progress at sensible intervals. If the run diverges, crashes, or is clearly not producing useful results, **stop it and destroy the instance**. Do not leave a broken run going.
- Push checkpoints/results somewhere durable (copy back locally or push to the HF Hub) **before** destroying, because destroying deletes all data on the instance.

### Destroying (MANDATORY)

As soon as results are copied off, destroy the instance:

```bash
echo y | vastai destroy instance <instance_id>
```

`vastai destroy instance` asks for `[y/N]` confirmation and **silently aborts** when run non-interactively, so the `echo y |` is required. **Always verify** the instance is gone:

```bash
vastai show instances --raw   # must return []
```

If this does not return `[]`, the instance is still billing. Fix that before doing anything else.

### Checklist

- [ ] Code verified locally before renting
- [ ] `vastai show instances --raw` returned `[]` before creating (max **one** instance)
- [ ] Cheapest suitable offer chosen
- [ ] Results/checkpoints copied off the instance
- [ ] Instance destroyed **immediately** after the experiment (or on failure)
- [ ] `vastai show instances --raw` returned `[]` after destroying
