import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import fcntl

import optuna
from optuna.trial import TrialState
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import train as train_module

MIN_AGE_FIXED = 18


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna超参搜索（最小化验证集MAE）')
    parser.add_argument('--study-name', type=str, default='hparam_search', help='Optuna study名称')
    parser.add_argument('--n-trials', type=int, default=30, help='总trial数量')
    parser.add_argument('--max-epochs', type=int, default=20, help='搜索阶段最大训练轮数')
    parser.add_argument('--gpus', nargs='*', type=int, default=None, help='可用GPU列表')
    parser.add_argument('--n-workers', type=int, default=None, help='并行worker数量（默认=GPU数）')
    parser.add_argument('--timeout', type=int, default=None, help='搜索超时（秒）')
    parser.add_argument('--storage', type=str, default=None, help='Optuna sqlite存储路径')
    parser.add_argument('--pruner-warmup-epochs', type=int, default=5, help='剪枝warmup轮数')
    parser.add_argument('--pruner-startup-trials', type=int, default=5, help='剪枝startup trials')
    parser.add_argument('--base-params', type=str, default=None, help='基础参数JSON路径')
    parser.add_argument('--worker', action='store_true', help='仅运行worker模式')
    opt_args, train_args = parser.parse_known_args()
    return opt_args, train_args


def resolve_gpus(explicit_gpus):
    if explicit_gpus:
        return explicit_gpus
    env_gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env_gpus:
        return [int(x) for x in env_gpus.split(',') if x.strip()]
    count = torch.cuda.device_count()
    if count <= 0:
        return [None]
    return list(range(count))


def make_storage_url(storage_path):
    return f"sqlite:///{storage_path}"


def load_base_params(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_base_params(args, base_params):
    for key, value in base_params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.min_age = MIN_AGE_FIXED


def build_objective(base_args, max_epochs):
    def objective(trial):
        args = copy.deepcopy(base_args)
        args.no_save = True
        args.epochs = max_epochs
        args.ensemble = False
        args.min_age = MIN_AGE_FIXED
        args.patience = max_epochs + 1

        args.lr = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
        args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        args.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 96])
        args.optimizer = trial.suggest_categorical('optimizer', ['adamw', 'sgd'])
        args.dropout = trial.suggest_float('dropout', 0.0, 0.7)
        args.scheduler = trial.suggest_categorical('scheduler', ['cosine', 'step', 'plateau', 'none'])

        if args.scheduler == 'cosine':
            args.warmup_epochs = trial.suggest_int('warmup_epochs', 0, 10)
            args.eta_min = trial.suggest_float('min_lr', 1e-7, 1e-5, log=True)
        elif args.scheduler == 'step':
            args.step_size = trial.suggest_categorical('step_size', [5, 10, 20])
            args.gamma = trial.suggest_float('gamma', 0.1, 0.9)
        elif args.scheduler == 'plateau':
            args.lr_patience = trial.suggest_int('lr_patience', 2, 10)
            args.lr_factor = trial.suggest_float('lr_factor', 0.1, 0.9)
            args.lr_min_lr = trial.suggest_float('min_lr', 1e-7, 1e-5, log=True)

        def reporter(epoch, val_mae):
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                trial.set_user_attr('fail_reason', 'pruned')
                raise optuna.TrialPruned(f'Pruned at epoch {epoch}')

        try:
            result = train_module.train(args, reporter=reporter)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = str(exc).lower()
            if 'out of memory' in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trial.set_user_attr('fail_reason', 'oom')
                raise
            trial.set_user_attr('fail_reason', f'runtime_error: {str(exc)}')
            raise
        except Exception as exc:
            trial.set_user_attr('fail_reason', f'exception: {str(exc)}')
            raise

        if not result or 'best_val_mae' not in result:
            trial.set_user_attr('fail_reason', 'missing_metric')
            raise RuntimeError('训练未返回有效的best_val_mae')

        best_val_mae = result['best_val_mae']
        if not torch.isfinite(torch.tensor(best_val_mae)):
            trial.set_user_attr('fail_reason', 'nan_or_inf')
            raise RuntimeError('best_val_mae为NaN/Inf')

        trial.set_user_attr('best_epoch', result.get('best_epoch'))
        trial.set_user_attr('train_time', result.get('train_time'))
        return best_val_mae

    return objective


def worker_loop(gpu_id, opt_args, train_args, storage_url, log_path):
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    output_root = Path('outputs') / 'hparam_search' / opt_args.study_name
    output_root.mkdir(parents=True, exist_ok=True)
    parser = train_module.create_arg_parser()
    base_args = parser.parse_args(train_args)
    base_params = load_base_params(opt_args.base_params)
    apply_base_params(base_args, base_params)

    objective = build_objective(base_args, opt_args.max_epochs)
    sampler = optuna.samplers.TPESampler(seed=base_args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=opt_args.pruner_startup_trials,
        n_warmup_steps=opt_args.pruner_warmup_epochs
    )
    study = optuna.create_study(
        study_name=opt_args.study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True
    )

    start_time = time.time()
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'[worker {gpu_id}] start\n')
    while True:
        if opt_args.timeout and (time.time() - start_time) >= opt_args.timeout:
            break
        if not acquire_trial_slot(study, output_root, opt_args.n_trials):
            break
        study.optimize(objective, n_trials=1, catch=(Exception,))
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f'[worker {gpu_id}] done\n')


def acquire_trial_slot(study, output_root, max_trials):
    budget_path = output_root / 'trial_budget.json'
    lock_path = output_root / 'trial_budget.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'w', encoding='utf-8') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        issued = 0
        if budget_path.exists():
            with open(budget_path, 'r', encoding='utf-8') as f:
                try:
                    issued = int(json.load(f).get('issued', 0))
                except Exception:
                    issued = 0
        waiting_state = getattr(TrialState, 'WAITING', None)
        running_states = [TrialState.RUNNING]
        if waiting_state is not None:
            running_states.append(waiting_state)
        total_trials = len(study.get_trials(deepcopy=False, states=(
            TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, *running_states
        )))
        issued = max(issued, total_trials)
        if issued >= max_trials:
            return False
        issued += 1
        with open(budget_path, 'w', encoding='utf-8') as f:
            json.dump({'issued': issued}, f)
        return True


def save_results(study, output_root):
    results = []
    for trial in study.trials:
        results.append({
            'number': trial.number,
            'value': trial.value,
            'state': trial.state.name,
            'params': trial.params,
            'user_attrs': trial.user_attrs,
            'fail_reason': trial.user_attrs.get('fail_reason')
        })
    results_path = output_root / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    try:
        df = study.trials_dataframe(attrs=('number', 'value', 'state', 'params', 'user_attrs'))
        df['fail_reason'] = df['user_attrs'].apply(lambda x: x.get('fail_reason') if isinstance(x, dict) else None)
        df.to_csv(output_root / 'results.csv', index=False)
        ranked = df[df['state'] == 'COMPLETE'].sort_values('value')
        ranked.to_csv(output_root / 'results_ranked.csv', index=False)
    except Exception:
        pass


def summarize_param_effects(completed_trials):
    metrics = {}
    if not completed_trials:
        return metrics

    values = [t.value for t in completed_trials]
    params = [t.params for t in completed_trials]
    keys = sorted({k for p in params for k in p})

    for key in keys:
        col = [p.get(key) for p in params]
        if all(isinstance(x, (int, float)) for x in col):
            ranked = sorted(zip(col, values), key=lambda x: x[0])
            xs = [x for x, _ in ranked]
            ys = [y for _, y in ranked]
            n = len(xs)
            if n < 3 or max(xs) == min(xs):
                metrics[key] = {'type': 'numeric', 'spearman': 0.0, 'range': (min(xs), max(xs))}
                continue
            xranks = {v: i for i, v in enumerate(sorted(set(xs)))}
            yranks = {v: i for i, v in enumerate(sorted(set(ys)))}
            xranked = [xranks[v] for v in xs]
            yranked = [yranks[v] for v in ys]
            mean_x = sum(xranked) / n
            mean_y = sum(yranked) / n
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xranked, yranked))
            var_x = sum((x - mean_x) ** 2 for x in xranked)
            var_y = sum((y - mean_y) ** 2 for y in yranked)
            spearman = 0.0 if var_x == 0 or var_y == 0 else cov / (var_x ** 0.5 * var_y ** 0.5)
            metrics[key] = {'type': 'numeric', 'spearman': spearman, 'range': (min(xs), max(xs))}
        else:
            buckets = {}
            for v, y in zip(col, values):
                buckets.setdefault(v, []).append(y)
            means = {k: sum(v) / len(v) for k, v in buckets.items()}
            spread = max(means.values()) - min(means.values()) if means else 0.0
            metrics[key] = {'type': 'categorical', 'means': means, 'spread': spread}
    return metrics


def detect_high_perf_region(completed_trials, top_frac=0.1):
    if not completed_trials:
        return {}
    top_n = max(1, int(len(completed_trials) * top_frac))
    top_trials = sorted(completed_trials, key=lambda t: t.value)[:top_n]
    params = [t.params for t in top_trials]
    region = {}
    keys = sorted({k for p in params for k in p})
    for key in keys:
        vals = [p.get(key) for p in params]
        if all(isinstance(x, (int, float)) for x in vals):
            region[key] = {'min': min(vals), 'max': max(vals)}
        else:
            counts = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            region[key] = {'top_categories': sorted(counts.items(), key=lambda x: x[1], reverse=True)}
    return region


def summarize_failure_bias(trials, states):
    filtered = [t for t in trials if t.state in states]
    if not filtered:
        return {}
    params = [t.params for t in filtered]
    keys = sorted({k for p in params for k in p})
    bias = {}
    for key in keys:
        vals = [p.get(key) for p in params]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        bias[key] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return bias


def save_summary(study, output_root, base_params):
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    best = study.best_trial if completed else None
    param_effects = summarize_param_effects(completed)
    high_region = detect_high_perf_region(completed)
    fail_bias = summarize_failure_bias(study.trials, {TrialState.FAIL})
    prune_bias = summarize_failure_bias(study.trials, {TrialState.PRUNED})

    lines = [
        '# Optuna搜索总结',
        f'- min_age 固定为 {MIN_AGE_FIXED}',
        f'- 试验总数: {len(study.trials)}',
        f'- 完成: {len(completed)}',
        f'- 剪枝: {len(pruned)}',
        f'- 失败: {len(failed)}',
    ]
    if best:
        lines.append(f'- 最佳MAE: {best.value}')
        lines.append(f'- 最佳trial: {best.number}')
        lines.append(f'- 最佳参数: {json.dumps(best.params, ensure_ascii=False)}')
    if completed:
        strongest = sorted(
            [(k, abs(v.get('spearman', 0.0)), v) for k, v in param_effects.items()],
            key=lambda x: x[1],
            reverse=True
        )
        lines.append('## 参数影响（基于完成trial）')
        for k, score, meta in strongest:
            if meta['type'] == 'numeric':
                lines.append(f'- {k}: spearman={meta["spearman"]:.3f} range={meta["range"]}')
            else:
                lines.append(f'- {k}: spread={meta["spread"]:.3f} means={meta["means"]}')
        lines.append('## 高性能区域（Top10% trial）')
        lines.append(json.dumps(high_region, ensure_ascii=False))
        lines.append('## 失败/剪枝集中趋势')
        lines.append(f'- fail_top: {json.dumps(fail_bias, ensure_ascii=False)}')
        lines.append(f'- pruned_top: {json.dumps(prune_bias, ensure_ascii=False)}')
        lines.append('## short-run 排名可信度')
        lines.append('- 仅基于短训练轮数的相对排序；若有 full-run 结果需重新验证排名一致性。')
    if base_params:
        lines.append(f'- base_params: {json.dumps(base_params, ensure_ascii=False)}')

    summary_path = output_root / 'summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def save_best_params(study, output_root, base_params):
    best = study.best_trial
    merged = dict(base_params)
    merged.update(best.params)
    merged['min_age'] = MIN_AGE_FIXED
    out_path = output_root / 'best_params.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def main():
    opt_args, train_args = parse_args()
    output_root = Path('outputs') / 'hparam_search' / opt_args.study_name
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / 'search.log'
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('[main] start\n')

    storage_path = Path(opt_args.storage) if opt_args.storage else output_root / 'study.sqlite3'
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = make_storage_url(storage_path.resolve())

    gpus = resolve_gpus(opt_args.gpus)
    if opt_args.n_workers is None:
        opt_args.n_workers = len(gpus)
    opt_args.n_workers = max(1, min(opt_args.n_workers, len(gpus)))

    if opt_args.worker:
        worker_loop(gpus[0], opt_args, train_args, storage_url, log_path)
        return

    if opt_args.n_workers == 1:
        worker_loop(gpus[0], opt_args, train_args, storage_url, log_path)
    else:
        processes = []
        for i in range(opt_args.n_workers):
            gpu_id = gpus[i]
            cmd = [
                sys.executable, str(__file__),
                '--study-name', opt_args.study_name,
                '--n-trials', str(opt_args.n_trials),
                '--max-epochs', str(opt_args.max_epochs),
                '--pruner-warmup-epochs', str(opt_args.pruner_warmup_epochs),
                '--pruner-startup-trials', str(opt_args.pruner_startup_trials),
                '--storage', str(storage_path),
                '--gpus', str(gpu_id),
                '--base-params', opt_args.base_params if opt_args.base_params else '',
                '--worker'
            ]
            if opt_args.timeout is not None:
                cmd += ['--timeout', str(opt_args.timeout)]
            cmd += train_args
            p = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
            processes.append(p)
        for p in processes:
            p.wait()

    study = optuna.load_study(
        study_name=opt_args.study_name,
        storage=storage_url
    )
    base_params = load_base_params(opt_args.base_params)
    save_results(study, output_root)
    save_summary(study, output_root, base_params)
    save_best_params(study, output_root, base_params)

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('[main] done\n')


if __name__ == '__main__':
    main()
