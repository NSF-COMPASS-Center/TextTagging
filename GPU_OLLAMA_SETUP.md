# GPU box setup checklist (Ollama tagging)

Everything needed to run `test_llm_tagger.py` / `runner.py` / `run_batch.py`
with `--provider ollama` on a GPU machine you `ssh` into.

## 1. Get the code over

- [ ] Commit + push the current local work on `taggers` (uncommitted right now:
      `Document_Parsing/chunk_filter.py`, `test_llm_tagger.py`, and the diffs in
      `Document_Parsing/__init__.py`, `Extraction/semantic_baseline.py`,
      `run_batch.py`, `runner.py`) — the GPU box needs these to run the tagger.
- [ ] On the GPU box: `git clone`/`git pull`, `git checkout taggers`

## 2. Data & config that `.gitignore` excludes (copy these manually, e.g. `scp -r` / `rsync`)

Everything below is untracked on purpose (`output/`, `tags/`, `papers/`,
`*.zip`, `run_cfs*.yaml`, `run_loss_survival*.yaml`, `run.example.yaml`,
`tag_config.json` are all in `.gitignore`), so a fresh clone on the GPU box
won't have them:

- [ ] `tags/` — needed for every paper type you'll tag (`tags/{cfs,uv,loss_survival}/llm_tags.json` at minimum; the others if you'll also run extraction)
- [ ] `papers/` (or the zips `cfs_markdown.zip` / `loss_survival_markdown.zip`, then `unzip` them into `papers/<type>/figures_with_markdown/`)
- [ ] Any `run_*.yaml` configs you use with `run_batch.py`
- [ ] `tag_config.json` if anything downstream reads it
- [ ] `.env` — **only needed if extraction (`--extract`) still calls OpenAI/Gemini**; pure Ollama tagging doesn't need API keys at all

## 3. Python env

- [ ] `python3 -m venv venv && source venv/bin/activate`
- [ ] `pip install -r requirements.txt` (pulls in the `ollama` client package already)

## 4. Ollama on the GPU box

- [ ] Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
- [ ] Confirm the GPU is visible to Ollama (`nvidia-smi` shows the card; Ollama's
      own logs on startup should say it detected a GPU, not "CPU only")
- [ ] Start the server: `ollama serve` (or check it's already running as a systemd service — `systemctl status ollama`)
- [ ] Pull the model: `ollama pull gpt-oss:20b` (~14 GB download — check disk space first)
- [ ] Sanity check it's up: `curl http://localhost:11434/api/tags` should list `gpt-oss:20b`
- [ ] Sanity check GPU offload during a real call: `ollama ps` while a request is running should show the model with `100% GPU` (not `CPU`)

## 5. If you're driving this from your laptop instead of running directly on the GPU box

Only needed if you `ssh` in just to keep Ollama running but want to launch
`test_llm_tagger.py` from elsewhere:

- [ ] `ssh -L 11434:localhost:11434 <gpu-host>` to tunnel Ollama's port locally
- [ ] Pass `--host http://localhost:11434` (or `tagging_llm_host` in a YAML config) so the client on your laptop hits the tunnel

Simplest path: just run everything (venv + script) directly on the GPU box over
the ssh session — then Ollama's default `http://localhost:11434` works with no
extra flags.

## 6. Run it

- [ ] `source venv/bin/activate`
- [ ] `python test_llm_tagger.py --provider ollama --model gpt-oss:20b --paper-type cfs`
- [ ] Check `output/test_llm_tagger.json` (or wherever `--output` points) for the tagged chunks + reasoning

### Running the `experiments/*/llm/` configs

- [ ] `python run_batch.py --config experiments/<paper_type>/<method>/llm/sent<N>.yaml --input papers/<paper_type>/.../*.md`
- [ ] Do **not** pass `--ground-truth` — the eval report only runs when that flag is
      given, so leaving it off skips the ground-truth accuracy step entirely and
      keeps the GPU run to tagging + extraction + the per-paper
      `<pdf_name>_comparison.csv`/`.html` side-by-side, which lands in each paper's
      `output_dir` regardless of ground truth.
