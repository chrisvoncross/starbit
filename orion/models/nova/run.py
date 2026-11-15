"""Manifest first. Hours only after that, and only through Door."""

from __future__ import annotations

from pathlib import Path

import torch

from .ckpt import exists, load, load_scale, save
from .data.batch import input_width
from .data.scale import fit
from .data.walk import Ticks, open_walk
from .log import Log
from .memory import Memory
from .net.model import Net
from .ride import Ride
from .spec import Manifest, by_name
from .step import train
from .vary import Align


def run(
    manifest: Manifest,
    hours: tuple[str, ...],
    *,
    read,
    log_path: str | Path | None,
    ckpt_dir: str | Path,
    form=None,
    scale=None,
) -> Log:
    if manifest.steps < 1:
        raise ValueError("the manifest has no steps")
    log = Log(log_path)
    form = form or by_name(manifest.form)
    log.write("run", form=form.name, loss=manifest.loss, steps=int(manifest.steps), hours=len(tuple(hours)))

    def watched(token):
        log.write("hour", token=str(token))
        return read(token)

    root = Path(ckpt_dir)
    resumed = exists(root)
    if scale is None and resumed:
        scale = load_scale(root)
    if scale is None:
        log.write("fit", days=list(manifest.train_days))
        scale = fit(manifest.train_days, read=watched)
        log.write("scale", symbols=len(scale.symbols))
    net = Net(form, input_width())
    if torch.cuda.is_available():
        net.cuda()
    decay, held = [], []
    for param in net.parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or getattr(param, "_no_weight_decay", False):
            held.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay}, {"params": held, "weight_decay": 0.0}],
        lr=manifest.lr,
    )
    memory = Memory()
    resume = None
    seen = 0
    if resumed:
        resume, seen = load(root, net, memory, optimizer=optimizer)
        log.write("resume", hour=None if resume is None else int(resume.hour), step=seen)
    if seen >= manifest.steps:
        log.write("done", steps=seen)
        return log
    ticks = Ticks()
    total_loss = 0.0
    total_correct = 0
    total_labeled = 0
    total_rides = 0
    total_off_rides = 0
    ride = Ride()
    align = Align()
    group_sum = {name: 0.0 for name in ("use_ohlc", "use_book", "use_trade", "use_other", "pull_ohlc", "pull_book", "pull_trade", "pull_other")}
    group_n = {name: 0 for name in group_sum}
    with open_walk(
        hours,
        lookback=manifest.lookback,
        batch=manifest.batch,
        seed=manifest.seed,
        read=watched,
        resume=resume,
        log=log,
    ) as walk:
        for batch in walk:
            report = train(
                net,
                optimizer,
                batch,
                scale,
                memory,
                manifest.loss,
                ticks.of(walk.stream),
                ride,
                align,
            )
            seen += 1
            total_loss += report["loss"]
            total_correct += report["correct"]
            total_labeled += report["labeled"]
            total_rides += int(report.get("ride_steps", 0))
            for name in group_sum:
                if report.get(name) is not None:
                    group_sum[name] += report[name]
                    group_n[name] += 1
            total_off_rides += int(report.get("off_ride_steps", 0))
            point = walk.resume_state()
            log.write(
                "step",
                step=seen,
                loss=report["loss"],
                running_loss=total_loss / seen,
                acc=report["acc"],
                running_acc=None if total_labeled == 0 else total_correct / total_labeled,
                labeled=report["labeled"],
                running_labeled=total_labeled,
                correct=report["correct"],
                label_none=report["label_none"],
                label_low=report["label_low"],
                label_high=report["label_high"],
                pred_none=report["pred_none"],
                pred_low=report["pred_low"],
                pred_high=report["pred_high"],
                off=report.get("off"),
                running_off=None if total_labeled == 0 else (total_labeled - total_correct) / total_labeled,
                ride=report.get("ride"),
                running_ride=None if total_labeled == 0 else total_rides / total_labeled,
                off_ride=report.get("off_ride"),
                running_off_ride=None if total_labeled == total_correct else total_off_rides / (total_labeled - total_correct),
                ride_max=report.get("ride_max", 0),
                hit_max=report.get("hit_max", 0),
                logit_none=report.get("logit_none"),
                logit_low=report.get("logit_low"),
                logit_high=report.get("logit_high"),
                grad_head=report.get("grad_head"),
                grad_block=report.get("grad_block"),
                margin_std=report.get("margin_std"),
                feat_std=report.get("feat_std"),
                use_move=report.get("use_move"),
                use_ohlc=report.get("use_ohlc"),
                use_book=report.get("use_book"),
                use_trade=report.get("use_trade"),
                use_other=report.get("use_other"),
                pull_move=report.get("pull_move"),
                pull_ohlc=report.get("pull_ohlc"),
                pull_book=report.get("pull_book"),
                pull_trade=report.get("pull_trade"),
                pull_other=report.get("pull_other"),
                running_use_ohlc=None if group_n["use_ohlc"] == 0 else group_sum["use_ohlc"] / group_n["use_ohlc"],
                running_use_book=None if group_n["use_book"] == 0 else group_sum["use_book"] / group_n["use_book"],
                running_use_trade=None if group_n["use_trade"] == 0 else group_sum["use_trade"] / group_n["use_trade"],
                running_use_other=None if group_n["use_other"] == 0 else group_sum["use_other"] / group_n["use_other"],
                running_pull_ohlc=None if group_n["pull_ohlc"] == 0 else group_sum["pull_ohlc"] / group_n["pull_ohlc"],
                running_pull_book=None if group_n["pull_book"] == 0 else group_sum["pull_book"] / group_n["pull_book"],
                running_pull_trade=None if group_n["pull_trade"] == 0 else group_sum["pull_trade"] / group_n["pull_trade"],
                running_pull_other=None if group_n["pull_other"] == 0 else group_sum["pull_other"] / group_n["pull_other"],
                lag_n=report.get("lag_n"),
                lag0=report.get("lag0"),
                lag_prev=report.get("lag_prev"),
                lag_next=report.get("lag_next"),
                running_lag0=None if align.pairs == 0 else align.hit0 / align.pairs,
                running_lag_prev=None if align.pairs == 0 else align.hit_prev / align.pairs,
                running_lag_next=None if align.pairs == 0 else align.hit_next / align.pairs,
                hour=int(point.hour),
                rows=len(batch),
            )
            if seen % manifest.ckpt_every == 0 or seen == manifest.steps:
                save(root, net, memory, point, log, optimizer=optimizer, step=seen, scale=scale)
            if seen >= manifest.steps:
                break
    log.write("done", steps=seen)
    return log
