from pathlib import Path
import hashlib
import json
import os
import random
import re
import time
import urllib.request

MODEL = 'unum-cloud/uform-gen2-qwen-500m'
REVISION = '78dc2e4d600def7698d5fb3733bea4e22dd2f3f9'
TOKENIZER = 'Qwen/Qwen1.5-0.5B-Chat'
TOKENIZER_REVISION = '4d14e384a4b037942bb3f3016665157c8bcb70ea'
TASKS = {'caption': 'Describe this image.', 'protest': 'Does this image contain protest?',
         'police': 'Does this image contain police?', 'violence': 'Does this image contain protester violence?'}
FRAMES = {'baseline': '',
          'civic_frame': 'Context supplied with the photograph: This gathering is described as a peaceful demonstration.\n',
          'disorder_frame': 'Context supplied with the photograph: This gathering is described as a violent riot.\n'}
SAMPLE = [
    dict(id='protest_march', url='https://upload.wikimedia.org/wikipedia/commons/c/c3/1963_march_on_washington.jpg',
         source='https://commons.wikimedia.org/wiki/File:1963_march_on_washington.jpg', credit='Rowland Scherman / U.S. Information Agency / National Archives', event='March on Washington 1963'),
    dict(id='protest_memorial', url='https://upload.wikimedia.org/wikipedia/commons/7/75/1963_march_lincoln_memorial.jpg',
         source='https://commons.wikimedia.org/wiki/File:1963_march_lincoln_memorial.jpg', credit='U.S. Information Agency / National Archives', event='March on Washington 1963'),
    dict(id='protest_police', url='https://upload.wikimedia.org/wikipedia/commons/4/45/Civil_rights_march_on_Washington%2C_D.C._police_line.jpg',
         source='https://commons.wikimedia.org/wiki/File:Civil_rights_march_on_Washington,_D.C._police_line.jpg', credit='Marion S. Trikosko / Library of Congress', event='March on Washington 1963'),
    dict(id='protest_suffrage', url='https://cdn.loc.gov/service/pnp/ggbain/11300/11383v.jpg',
         source='https://www.loc.gov/item/2014691470/', credit='Bain News Service / Library of Congress', event='Suffrage parade catalogued 1913'),
    dict(id='control_boardwalk', url='https://cdn.loc.gov/service/pnp/cph/3c00000/3c07000/3c07800/3c07839v.jpg',
         source='https://www.loc.gov/item/93506594/',
         credit='George Grantham Bain Collection / Library of Congress', event='Coney Island 1908'),
    dict(id='control_station', url='https://cdn.loc.gov/service/pnp/fsa/8c33000/8c33100/8c33198v.jpg',
         source='https://www.loc.gov/item/2017820721/',
         credit='John Collier Jr. / Library of Congress', event='Grand Central Terminal 1941'),
]

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def fetch(url, limit=12_000_000):
    request = urllib.request.Request(url, headers={'User-Agent': 'DissertationFeasibility/1.0 (academic image analysis)'})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Download exceeds size limit: ' + url)
    return data

def prepare_inputs(root):
    """Download the photographs, model and tokenizer."""
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    for name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
        os.environ.pop(name, None)
    from PIL import Image, ImageOps
    from huggingface_hub import snapshot_download
    root = Path(root).resolve()
    data = root / 'data'
    (data / 'originals').mkdir(parents=True, exist_ok=True)
    (data / 'images').mkdir(exist_ok=True)
    manifest_path = data / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for item in manifest:
            if digest(root / item['path']) != item['sha256']:
                raise ValueError(f"Image has changed: {item['id']}")
    else:
        manifest = []
        for spec in SAMPLE:
            item = dict(spec)
            url = item['url']
            original = data / 'originals' / (item['id'] + '.jpg')
            if not original.exists():
                original.write_bytes(fetch(url))
            processed = data / 'images' / (item['id'] + '.jpg')
            with Image.open(original) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                image.save(processed, quality=92)
            item.update(path=processed.relative_to(root).as_posix(), download_url=url,
                        sha256=digest(processed), original_sha256=digest(original),
                        rights='Public-domain / no-known-restrictions archival sources selected in the earlier pilot; retain and review linked source rights.',
                        preprocessing='EXIF orientation, RGB, fit within 768 pixels, JPEG quality 92, no crop')
            manifest.append(item)
            print('Prepared', item['id'], flush=True)
        save_json(manifest_path, manifest)
    model_dir = root / '.models/uform'
    tokenizer_dir = root / '.models/qwen_tokenizer'
    snapshot_download(MODEL, revision=REVISION, local_dir=model_dir,
                      allow_patterns=['*.py', '*.json', '*.safetensors', 'README.md'])
    snapshot_download(TOKENIZER, revision=TOKENIZER_REVISION, local_dir=tokenizer_dir,
                      allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'])
    files = [p for directory in (model_dir, tokenizer_dir) for p in directory.iterdir() if p.is_file()]
    lock = {'model': MODEL, 'revision': REVISION, 'tokenizer': TOKENIZER,
            'tokenizer_revision': TOKENIZER_REVISION,
            'files': [{'path': p.relative_to(root).as_posix(), 'sha256': digest(p)} for p in files]}
    save_json(root / 'model_lock.json', lock)
    print('Inputs ready.', flush=True)


def prepare_extra_images(root, sample):
    """Explicitly download optional photographs into a separate, reusable manifest."""
    from PIL import Image, ImageOps
    root = Path(root).resolve()
    manifest_path = root / 'data/intersectional_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if [item['id'] for item in manifest] != [item['id'] for item in sample]:
            raise ValueError('The optional image selection changed; preserve the existing manifest first.')
        for item, spec in zip(manifest, sample):
            if any(item.get(key) != value for key, value in spec.items()):
                raise ValueError(f"Optional source metadata changed: {item['id']}")
            if digest(root / item['path']) != item['sha256']:
                raise ValueError(f"Image has changed: {item['id']}")
            if digest(root / item['original_path']) != item['original_sha256']:
                raise ValueError(f"Original image has changed: {item['id']}")
        return manifest
    manifest = []
    for spec in sample:
        if not re.fullmatch(r'[a-z0-9_]+', spec['id']):
            raise ValueError('Use simple lowercase image IDs')
        original = root / 'data/intersectional_originals' / (spec['id'] + '.jpg')
        processed = root / 'data/intersectional_images' / (spec['id'] + '.jpg')
        original.parent.mkdir(parents=True, exist_ok=True)
        processed.parent.mkdir(parents=True, exist_ok=True)
        if not original.exists():
            original.write_bytes(fetch(spec['url']))
        with Image.open(original) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            image.save(processed, quality=92)
        item = dict(spec)
        item.update(path=processed.relative_to(root).as_posix(),
                    original_path=original.relative_to(root).as_posix(),
                    download_url=spec['url'], sha256=digest(processed),
                    original_sha256=digest(original),
                    preprocessing='EXIF orientation, RGB, fit within 768 pixels, JPEG quality 92, no crop')
        manifest.append(item)
        print('Prepared', item['id'], flush=True)
    save_json(manifest_path, manifest)
    return manifest


def parse_answer(text):
    """Conservative yes/no extraction; other answers remain available for human review."""
    if not isinstance(text, str):
        return 'unparseable'
    match = re.match(r'^\s*(yes|no)(?=$|[\s.,!?:;])', text, re.I)
    uncertainty = (
        r'\b(?:maybe|uncertain|unclear|unsure|ambiguous|possibly|perhaps)\b',
        r"\b(?:cannot|can't|could not|couldn't)\s+(?:determine|tell|say|be sure)\b",
        r'\bnot\s+(?:sure|certain|clear)\b',
        r'\bno\s+(?:way|means)\s+to\s+(?:tell|determine|know)\b',
        r'\b(?:yes\s+(?:or|and)\s+no|no\s+(?:or|and)\s+yes)\b',
    )
    if not match or any(re.search(pattern, text, re.I) for pattern in uncertainty):
        return 'unparseable'
    return match.group(1).lower()


def print_output(row):
    heading = f"{row['image_id']} | {row['task']} | {row['condition']}"
    if 'identity' in row:
        heading += f" | {row['identity']} | {row['source']} | {row['prompt_style']}"
    if row['repeat']:
        heading += f" | repeat {row['repeat']}"
    print(f"\n{heading}\n{row['prompt']}")
    if row['status'] == 'error':
        print(f"Error: {row['error']}", flush=True)
    else:
        print(row['raw_text'], flush=True)
        if row.get('truncated'):
            print('[Token limit reached]', flush=True)


def run_experiment(root, results_name='results_t4', limit=0, *, jobs=None,
                   extra_manifest_path=None, design=None,
                   device='cuda:0', dtype='float16'):
    """Run on one CUDA GPU; preserve the trial design and resume identical plans only."""
    if not isinstance(limit, int) or limit < 0:
        raise ValueError('limit must be 0 (all calls) or a positive number')
    if dtype not in {'float16', 'float32'}:
        raise ValueError('For T4 use float16, or float32 for a precision comparison; not bfloat16.')

    root = Path(root).resolve()
    os.environ.update(
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(root / '.cache/huggingface'),
        HF_MODULES_CACHE=str(root / '.cache/modules'), TOKENIZERS_PARALLELISM='false',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',
    )
    import importlib.metadata
    import platform
    import torch
    from PIL import Image
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoConfig, AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('This branch targets a CUDA GPU. Use the CPU version on master for CPU runs.')
    if not torch.cuda.is_available():
        raise RuntimeError(
            'No CUDA GPU is available. In Colab select Runtime > Change runtime type > T4 GPU. '
            'Restart the session after changing hardware or installing dependencies. '
            'Keep Colab\'s CUDA-enabled torch/torchvision pair; do not install CPU wheels.'
        )
    device = torch.device('cuda', device.index if device.index is not None else 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    compute_dtype = {'float16': torch.float16, 'float32': torch.float32}[dtype]
    hardware = dict(
        name=properties.name, compute_capability=list(torch.cuda.get_device_capability(device)),
        total_memory_gib=properties.total_memory / 1024**3,
        cuda_runtime=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
    )
    print(f'GPU: {hardware["name"]}; device: {device}; precision: {dtype}', flush=True)

    manifest = json.loads((root / 'data/manifest.json').read_text(encoding='utf-8'))
    if extra_manifest_path:
        manifest += json.loads((root / extra_manifest_path).read_text(encoding='utf-8'))
    lock = json.loads((root / 'model_lock.json').read_text(encoding='utf-8'))
    if not manifest or len({item['id'] for item in manifest}) != len(manifest):
        raise ValueError('The image manifest must contain unique image IDs')
    for item in manifest + lock['files']:
        if digest(root / item['path']) != item['sha256']:
            raise ValueError(f"File has changed: {item['path']}")

    custom_jobs = jobs is not None
    if custom_jobs:
        jobs = [dict(job) for job in jobs]
        required = {'image_id', 'task', 'condition', 'repeat', 'prompt', 'trial_id',
                    'identity', 'race_label', 'gender_label', 'source', 'prompt_style', 'image_set'}
        available = {item['id'] for item in manifest}
        if not jobs or any(not required.issubset(job) for job in jobs):
            raise ValueError('Intersectional trials need all factor fields and unique trial IDs')
        if any(job['image_id'] not in available or job['task'] not in TASKS for job in jobs):
            raise ValueError('A trial names an unknown image or task')
        if any(not isinstance(job['prompt'], str) or not job['prompt'].strip() for job in jobs):
            raise ValueError('Every trial needs a nonempty prompt')
        selected = {job['image_id'] for job in jobs}
        manifest = [item for item in manifest if item['id'] in selected]
    else:
        jobs = []
        for item in manifest:
            for task, question in TASKS.items():
                conditions = FRAMES if task == 'violence' else ['baseline']
                for condition in conditions:
                    jobs.append(dict(
                        image_id=item['id'], task=task, condition=condition,
                        repeat=0, prompt=FRAMES[condition] + question,
                    ))
        for task in ['protest', 'police', 'violence']:
            jobs.append(dict(
                image_id=manifest[0]['id'], task=task, condition='baseline',
                repeat=1, prompt=TASKS[task],
            ))
    for job in jobs:
        job.setdefault('trial_id', '__'.join(
            str(job[key]) for key in ['image_id', 'task', 'condition', 'repeat']
        ))
    if len({job['trial_id'] for job in jobs}) != len(jobs):
        raise ValueError('Trial IDs must be unique')
    random.Random(20261107).shuffle(jobs)

    generation = dict(
        do_sample=False, num_beams=1, max_new_tokens=48,
        use_cache=True, eos_token_id=151645,
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ['torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'Pillow',
                     'accelerate', 'huggingface-hub', 'safetensors']
    }
    plan = dict(
        jobs=jobs, model_lock=lock, images=manifest, generation=generation,
        threads=4, device=str(device), dtype=dtype, seed=20261107,
        hardware=hardware, attention_backend='sdpa_math',
        deterministic_algorithms=True, cublas_workspace_config=':4096:8',
        allow_tf32=False, cudnn_benchmark=False,
        implementation_sha256=digest(__file__),
        versions=versions, python=platform.python_version(),
    )
    if custom_jobs:
        plan['design'] = design or {}
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan['fingerprint'] = fingerprint

    output = root / results_name
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / 'raw_outputs.jsonl'
    existing = []
    if raw_path.exists():
        existing = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines() if line]
    if any(row['fingerprint'] != fingerprint for row in existing):
        raise ValueError('The experiment settings have changed. Choose a new results_name.')
    if any(row['status'] != 'ok' for row in existing):
        raise ValueError('This run contains a failed call. Fix the cause and choose a new results_name.')
    done = {row['trial_id'] for row in existing}
    if len(done) != len(existing) or not done.issubset({job['trial_id'] for job in jobs}):
        raise ValueError('The results contain duplicate or unknown calls')
    pending = [job for job in jobs if job['trial_id'] not in done]
    if limit:
        pending = pending[:limit]
    save_json(output / 'experiment_plan.json', plan)
    if not pending:
        print('All responses are already saved.')
        return

    torch.set_num_threads(4)
    torch.manual_seed(20261107)
    torch.cuda.manual_seed_all(20261107)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    model_dir = root / '.models/uform'
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    config.text_decoder_name_or_path = str(root / '.models/qwen_tokenizer')
    model = AutoModel.from_pretrained(
        str(model_dir), config=config, trust_remote_code=True,
        local_files_only=True, torch_dtype=compute_dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    processor_cls = get_class_from_dynamic_module(
        'processing_uform_gen.VLMProcessor', str(model_dir), local_files_only=True,
    )
    processor = processor_cls(config)
    if any(parameter.device != device for parameter in model.parameters()):
        raise RuntimeError('All model parameters must reside on the selected CUDA GPU.')
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    session_peak_allocated_gib = torch.cuda.memory_allocated(device) / 1024**3
    lookup = {item['id']: item for item in manifest}

    with raw_path.open('a', encoding='utf-8') as handle:
        for job in pending:
            item = lookup[job['image_id']]
            row = dict(job, fingerprint=fingerprint, image_sha256=item['sha256'])
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            try:
                with Image.open(root / item['path']) as source:
                    image = source.convert('RGB')
                inputs = processor(text=[job['prompt']], images=[image], return_tensors='pt')
                # Cast image tensors only: token IDs and attention masks must remain integers.
                inputs = {
                    key: value.to(device=device, dtype=compute_dtype if value.is_floating_point() else value.dtype)
                    for key, value in inputs.items()
                }
                # The math SDPA path supports T4; no FlashAttention package is required.
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                    tokens = model.generate(
                        **inputs, **generation, pad_token_id=processor.tokenizer.pad_token_id,
                    )
                ids = tokens[0, inputs['input_ids'].shape[1]:].detach().cpu().tolist()
                torch.cuda.synchronize(device)
                answer = processor.decode(ids, skip_special_tokens=True)
                row.update(
                    status='ok', raw_text=answer, generated_tokens=len(ids),
                    parsed_label=None if job['task'] == 'caption' else parse_answer(answer),
                    truncated=len(ids) >= generation['max_new_tokens']
                    and bool(ids) and ids[-1] != generation['eos_token_id'],
                )
            except Exception as error:
                row.update(status='error', raw_text='', parsed_label='unparseable', truncated=False,
                           elapsed_seconds=time.perf_counter() - tick,
                           error=f'{type(error).__name__}: {error}')
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                print_output(row)
                raise
            row['elapsed_seconds'] = time.perf_counter() - tick
            row['gpu_peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 1024**3
            row['gpu_peak_reserved_gib'] = torch.cuda.max_memory_reserved(device) / 1024**3
            session_peak_allocated_gib = max(session_peak_allocated_gib, row['gpu_peak_allocated_gib'])
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
            print_output(row)
            del inputs, tokens
    save_json(output / f'session_{time.time_ns()}.json', dict(
        device=str(device), dtype=dtype, hardware=hardware, model_load_seconds=load_seconds,
        gpu_peak_allocated_gib=session_peak_allocated_gib,
        elapsed_seconds=time.perf_counter() - started,
        calls_this_session=len(pending), fingerprint=fingerprint,
    ))
    del model, processor
    torch.cuda.empty_cache()
