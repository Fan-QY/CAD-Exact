# -----------------------------------------------------------------------------
# Batched inference for FUSED CADRecode models. Drop-in replacement for
# demo_CADLLM_fused.py with --fast --skip-iou; produces only .py + .stl per sample
# (no IoU/CD/PNG/STEP). Use --batch_size N to tune throughput.
# -----------------------------------------------------------------------------

import os
import json
import signal
import argparse
from typing import Dict, Any, List

import numpy as np
import trimesh
import cadquery as cq
from local.data_process import *
from local.run_setup import attn_impl
from tqdm import tqdm


class ExecTimeout(Exception):
    pass


def _exec_alarm_handler(signum, frame):
    raise ExecTimeout("exec timeout")


# kill any sample whose CadQuery exec hangs longer than this (CadQuery can loop forever on bad geometry)
EXEC_TIMEOUT_SEC = 30
signal.signal(signal.SIGALRM, _exec_alarm_handler)

# A predicted program can be syntactically valid and still ask OCCT to build a
# pathological solid; tessellating that at a 0.001 tolerance allocates without
# bound. The timeout never fires, because the process is not slow -- it is busy
# allocating -- so the job is OOM-killed instead and loses every sample the batch
# had not yet written. Three of eight cluster cells died this way, one with 128GB.
#
# The cap has to be relative. RLIMIT_AS covers the whole process and CUDA reserves
# tens of gigabytes of address space for its context, so a flat ceiling would stop
# the model from loading at all. The limit is therefore raised to "what this
# process already maps, plus room for one tessellation" for the duration of exec()
# and restored afterwards, which bounds the damage a single bad program can do.
# RLIMIT_AS rather than RLIMIT_DATA: OCCT allocates through mmap.
EXEC_MEM_HEADROOM_GB = float(os.environ.get("EXEC_MEM_HEADROOM_GB", "24"))


def _vm_size_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.readline().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


class exec_memory_cap:
    """Bound the address space one exec() may add. No-op where unsupported."""

    def __enter__(self):
        self.prev = None
        if EXEC_MEM_HEADROOM_GB <= 0:
            return self
        try:
            import resource
            vm = _vm_size_bytes()
            if vm <= 0:
                return self
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            cap = vm + int(EXEC_MEM_HEADROOM_GB * (1 << 30))
            if hard != resource.RLIM_INFINITY:
                cap = min(cap, hard)
            if soft == resource.RLIM_INFINITY or cap < soft:
                resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
                self.prev = (soft, hard)
        except Exception:
            self.prev = None
        return self

    def __exit__(self, *exc):
        if self.prev is not None:
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, self.prev)
            except Exception:
                pass
        return False


import torch
from torch import nn
from transformers import (
    AutoTokenizer, Qwen2ForCausalLM, Qwen2Model, PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutputWithPast


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points):
        x = points
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        x = self.projection(x)
        return x


class CADRecode(Qwen2ForCausalLM):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                point_cloud=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                cache_position=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None or past_key_values.get_seq_length() == 0:
            assert inputs_embeds is None
            inputs_embeds = self.model.embed_tokens(input_ids)
            point_embeds = self.point_encoder(point_cloud).bfloat16()
            mask = (attention_mask == -1)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = point_embeds.reshape(-1, point_embeds.size(-1))
            attention_mask = attention_mask.clone()
            attention_mask[mask] = 1
            input_ids = None
            position_ids = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs['point_cloud'] = kwargs['point_cloud']
        return model_inputs


def get_args():
    parser = argparse.ArgumentParser(description='Batched inference for FUSED CADRecode models')
    parser.add_argument('--datainfo', required=True, type=str,
                        help='dir containing test_mesh_paths.json')
    parser.add_argument('--savedir', required=True, type=str,
                        help='dir to write per-sample <id>/{<id>.txt, <id>.stl}')
    parser.add_argument('--modeldir', required=True, type=str,
                        help='dir containing checkpoint-<N>-fused/')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='checkpoint number to load (loads checkpoint-{N}-fused)')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size for parallel inference (default: 4)')
    parser.add_argument('--n_points', default=256, type=int)
    parser.add_argument('--pc_seed', default=42, type=int,
                        help="seed for the surface sampling that produces the input point "
                             "cloud. Training resamples every visit, so the model is meant "
                             "to be invariant to it; varying this measures how much of the "
                             "prediction is determined by the sample rather than the shape.")
    parser.add_argument('--pc_resample', default=0, type=int,
                        help="draw this many points from a .npy cloud, seeded by --pc_seed, "
                             "before farthest-point sampling. 0 disables it. Non-zero is what "
                             "makes repeated runs differ, i.e. what best-of-N needs.")
    parser.add_argument('--max_new_tokens', default=1024, type=int)
    parser.add_argument('--no_mesh', action='store_true',
                        help='write only <id>.txt; skip exec/tessellate/STL (PEM-only runs)')
    return parser.parse_args()


def load_point_cloud(datapath: str, n_points: int, seed: int = 42,
                     resample: int = 0) -> np.ndarray:
    """Reduce one sample to an n_points cloud. Raises on failure.

    Two input forms. A mesh is scaled by 1e-3 first -- that is what turns the
    corpus's integer coordinates into the frame the model trained in -- and then
    surface sampled. A .npy is taken as points ALREADY in that frame; scaling
    them here would apply the transform twice. Whoever wrote the .npy owns the
    normalisation, and an external benchmark whose normalisation differs will
    look like a model that generalises badly, so that step is deliberately not
    hidden in here.
    """
    if datapath.endswith(".npy"):
        pts = np.load(datapath).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"expected (N,3) points, got {pts.shape}")
        if not np.isfinite(pts).all():
            raise ValueError("non-finite coordinates in point cloud")
        if len(pts) < n_points:
            raise ValueError(f"only {len(pts)} points, need {n_points}")
        # sample_farthest_points starts from index 0 and consumes no numpy
        # randomness, so seeding alone leaves the 256-point cloud bit-identical
        # across seeds -- best-of-N built on --pc_seed silently returns N copies
        # of one prediction. Drawing a seeded subset of the input first is what
        # actually varies it, and it is also what CAD-Recode's test-time
        # sampling does: "each from a different sampling of the input point
        # cloud". resample=0 keeps the deterministic single-candidate path.
        if resample:
            rng = np.random.RandomState(seed)
            k = min(resample, len(pts))
            if k < n_points:
                raise ValueError(f"resample {k} below n_points {n_points}")
            pts = pts[rng.choice(len(pts), size=k, replace=False)]
        np.random.seed(seed)
        _, ids = sample_farthest_points(torch.tensor(pts).unsqueeze(0), K=n_points)
        return np.asarray(pts[ids[0].numpy()])

    gt_mesh = trimesh.load_mesh(datapath)
    gt_mesh.apply_scale(1e-3)
    np.random.seed(seed)
    return mesh_to_point_cloud(gt_mesh, n_points=n_points, n_pre_points=8192)


def main():
    args = get_args()
    print("=" * 60)
    print("Batched FUSED Inference")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    n_points = args.n_points
    test_mesh_json = os.path.join(args.datainfo, 'test_mesh_paths.json')
    mesh_map = read_json(test_mesh_json)

    if torch.cuda.is_available():
        device = "cuda"
        attn_implementation = attn_impl()
        print(f"Using GPU. visible count: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        attn_implementation = None
        print("Using CPU.")

    fused_model_dir = os.path.join(args.modeldir, f'checkpoint-{args.checkpoint}-fused')
    if not os.path.exists(fused_model_dir):
        raise FileNotFoundError(f"Fused model not found: {fused_model_dir}")
    print(f"Loading fused model from: {fused_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        fused_model_dir,
        pad_token='<|im_end|>',
        padding_side="left",
    )
    model = CADRecode.from_pretrained(
        fused_model_dir,
        torch_dtype="auto",
        attn_implementation=attn_implementation,
        device_map=device,
    )
    model = model.eval().to(device)
    print("✓ Model loaded")

    samples = list(mesh_map.keys())
    remaining = []
    for sample in samples:
        txt_file = os.path.join(args.savedir, sample, sample + '.txt')
        if not os.path.exists(txt_file):
            remaining.append(sample)
    print(f"Total: {len(samples)} | done: {len(samples) - len(remaining)} | remaining: {len(remaining)}")

    bos_id = tokenizer('<|im_start|>')['input_ids'][0]
    eos_id = tokenizer("<|endoftext|>")["input_ids"][0]
    pad_id = tokenizer.pad_token_id

    n_ok = 0
    n_fail = 0
    pbar = tqdm(range(0, len(remaining), args.batch_size),
                desc=f"Inferring (B={args.batch_size})")

    for batch_start in pbar:
        batch_samples = remaining[batch_start: batch_start + args.batch_size]

        # --- Stage 1: load point clouds, drop loaders that fail ---
        valid_samples = []
        valid_pcs = []
        for sample in batch_samples:
            try:
                pc = load_point_cloud(mesh_map[sample], n_points, args.pc_seed,
                                      args.pc_resample)
                valid_samples.append(sample)
                valid_pcs.append(pc)
            except Exception as e:
                n_fail += 1
                tqdm.write(f"[LOAD FAIL] {sample}: {type(e).__name__}: {str(e)[:80]}")

        if not valid_samples:
            continue

        # --- Stage 2: batched generate ---
        n = len(valid_samples)
        single_ids = [pad_id] * n_points + [bos_id]
        single_mask = [-1] * n_points + [1]
        input_ids = torch.tensor([single_ids] * n).to(model.device)
        attention_mask = torch.tensor([single_mask] * n).to(model.device)
        point_clouds = torch.tensor(np.stack(valid_pcs).astype(np.float32)).to(model.device)

        with torch.no_grad():
            batch_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                point_cloud=point_clouds,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                do_sample=False,
            )

        # --- Stage 3: per-sample decode + exec + save ---
        decoded = tokenizer.batch_decode(batch_ids)
        for sample, py_string in zip(valid_samples, decoded):
            savesampledir = os.path.join(args.savedir, sample)
            os.makedirs(savesampledir, exist_ok=True)

            begin = py_string.find('<|im_start|>') + 12
            end = py_string.find('<|endoftext|>')
            if end == -1:
                end = None
            py_string = py_string[begin: end]

            with open(os.path.join(savesampledir, sample + '.txt'), "w", encoding="utf-8") as f:
                f.write(py_string)

            if args.no_mesh:
                # PEM reads the emitted program, not the solid. Executing and
                # tessellating is what the run spends its wall clock on: on the
                # loft split the GPU sits at 21% while OCCT meshes 32 smooth
                # lofts per batch, so skipping it turns a 45-minute run into a
                # decode-bound one. Nothing that needs a mesh (validity, CD,
                # TVF, IoU) can be scored from a --no_mesh run.
                continue

            try:
                local_ns: Dict[str, Any] = {}
                signal.alarm(EXEC_TIMEOUT_SEC)
                try:
                    with exec_memory_cap():
                        exec(py_string, globals(), local_ns)
                        compound = local_ns['r'].val()
                        vertices, faces = compound.tessellate(0.001, 0.1)
                        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
                    mesh.export(os.path.join(savesampledir, sample + '.stl'))
                finally:
                    signal.alarm(0)
                n_ok += 1
            except ExecTimeout:
                n_fail += 1
                tqdm.write(f"[TIMEOUT] {sample}: exec exceeded {EXEC_TIMEOUT_SEC}s")
            except MemoryError:
                n_fail += 1
                tqdm.write(f"[MEMCAP] {sample}: exec exceeded "
                           f"{EXEC_MEM_HEADROOM_GB} GiB of extra address space")
            except Exception as e:
                n_fail += 1
                tqdm.write(f"[EXEC FAIL] {sample}: {type(e).__name__}: {str(e)[:80]}")

        pbar.set_postfix({'ok': n_ok, 'fail': n_fail})

    print()
    print("=" * 60)
    print(f"DONE. ok={n_ok}, fail={n_fail}, total processed={n_ok + n_fail}")
    print(f"Output: {args.savedir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
