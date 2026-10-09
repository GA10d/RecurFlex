"""Select epochs on the inner holdout, then fit one final seed-42 subject model."""
import argparse
import hashlib
import time
import numpy as np
import torch
from common import HERE, protocol, record, Config, setup, load_json, save_json, sha256
from common import training_model, RecurFlex, objective, sample_batch, predict, inverse_targets, restore_1000hz, coverage, score


def tensor_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit(subject,phase,epochs=None):
    cfg=Config();setup()
    folder=HERE/"fits"/subject/phase
    if (folder/"fit.json").exists():
        saved=load_json(folder/"fit.json")
        assert saved["model_sha256"]==sha256(folder/"model.pt")
        return saved
    source=HERE/"inputs"/subject/phase
    x=torch.from_numpy(np.load(source/"x_train.npy").transpose(2,0,1).copy()).cuda()
    y=torch.from_numpy(np.load(source/"y_train.npy")).cuda()
    stats=load_json(source/"scalers.json")
    model=training_model(cfg.dropout).cuda()
    assert sum(v.numel() for v in model.parameters())==6808198
    initial_sha=tensor_hash(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    generator=torch.Generator(device="cuda").manual_seed(42)
    count=epochs if phase=="full" else cfg.max_epochs
    assert count and count>0
    validation=None
    if phase=="validation":validation=(np.load(source/"x_validation.npy"),np.load(source/"validation_truth.npy"))
    folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/"config.json",dict(subject=subject,phase=phase,epochs=count,settings=cfg.to_dict(),
        initial_state_sha256=initial_sha,parameters=6808198,scalers_sha256=sha256(source/"scalers.json")))
    save_json(folder/"scalers.json",stats)
    history=[];best=-float("inf");best_epoch=0;best_state=None;start=time.monotonic()
    for epoch in range(1,count+1):
        model.train();total=0.
        for step in range(cfg.windows_per_epoch//cfg.batch):
            bx,by=sample_batch(x,y,cfg.batch,generator,cfg.window)
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model(bx),by)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())
        row=dict(subject=subject,phase=phase,epoch=epoch,epochs=count,loss=total/128,elapsed=time.monotonic()-start)
        if validation is not None:
            vx,vy=validation
            val_pred=inverse_targets(predict(model,vx,"cuda",2048),stats)
            validation_r4=score(vy[225:-25],val_pred[225:-25])["r4"]
            row["validation_r4"]=validation_r4
            if validation_r4>best+1e-4:
                best=validation_r4;best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row)
        save_json(folder/"history.json",history);save_json(folder/"progress.json",row)
        print(row,flush=True)
        if validation is not None and epoch-best_epoch>=cfg.patience:break
    state=best_state if validation is not None else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    saved_epoch=best_epoch if validation is not None else len(history)
    torch.save(dict(state_dict=state,subject=subject,seed=42,epoch=saved_epoch),folder/"model.pt")
    result=dict(subject=subject,phase=phase,seed=42,epochs=len(history),selected_epoch=saved_epoch,
        initial_state_sha256=initial_sha,elapsed=time.monotonic()-start,model_sha256=sha256(folder/"model.pt"),parameters=6808198)
    if validation is not None:
        r=record(subject);dev=r["splits"]["train_full"][1];inner=r["splits"]["train_inner"][1]
        result.update(validation_r4=best,suggested_full_epochs=max(8,min(160,round(best_epoch*dev/inner))))
    save_json(folder/"fit.json",result)
    return result


def run(subject):
    validation=fit(subject,"validation")
    return fit(subject,"full",validation["suggested_full_epochs"])


@torch.inference_mode()
def infer(subject):
    setup();folder=HERE/"fits"/subject/"full"
    checkpoint=torch.load(folder/"model.pt",map_location="cpu",weights_only=True)
    assert checkpoint["subject"]==subject and checkpoint["seed"]==42
    model=RecurFlex().cuda();model.load_state_dict(checkpoint["state_dict"],strict=True)
    stats=load_json(folder/"scalers.json")
    x=np.load(HERE/"inputs"/subject/"full/x_test.npy")
    native=inverse_targets(predict(model,x,"cuda",2048),stats)
    a,b=record(subject)["splits"]["test"]
    full=restore_1000hz(native,b-a)
    output=HERE/"predictions";output.mkdir(exist_ok=True)
    path=output/f"{subject}.npz"
    np.savez_compressed(path,prediction_100hz=native,prediction_1000hz=full)
    save_json(output/f"{subject}_coverage.json",coverage(len(native)))
    return dict(subject=subject,seed=42,file=str(path.relative_to(HERE)),prediction_sha256=sha256(path),
        model_sha256=sha256(folder/"model.pt"),scalers_sha256=sha256(folder/"scalers.json"),
        test_raw_indices=[a,b],shape_100hz=list(native.shape),shape_1000hz=list(full.shape))


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--subject",required=True)
    run(parser.parse_args().subject)
