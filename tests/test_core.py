"""Unit tests on synthetic masks (no competition data needed).

Run with:  python -m pytest -q tests
"""
import numpy as np
import pandas as pd
import pytest

from filseg import rle
from filseg.metrics import Crop, evaluate, evaluate_frames, overlap_matrices, pq_from_overlaps
from filseg.postprocess import PostConfig, grow_labels, instances_from_prob, label_to_crops

H = W = 2048


def box(y0, x0, h, w):
    m = np.zeros((H, W), bool)
    m[y0:y0 + h, x0:x0 + w] = True
    return m


def test_rle_roundtrip():
    m = box(10, 20, 30, 40) | box(1000, 1500, 5, 300)
    counts = rle.encode(m)
    assert isinstance(counts, str)
    assert np.array_equal(rle.decode(counts), m)
    assert rle.area(counts) == m.sum()


def test_polygon_rasterisation_matches_area():
    poly = [100.0, 100.0, 200.0, 100.0, 200.0, 150.0, 100.0, 150.0, 100.0, 100.0]
    m = rle.polygon_to_mask(poly)
    assert abs(int(m.sum()) - 100 * 50) <= 2 * (100 + 50)


def test_overlap_matrix_iou():
    a, b = Crop.from_full(box(0, 0, 10, 10)), Crop.from_full(box(0, 5, 10, 10))
    iou, dice = overlap_matrices([a], [b])
    assert iou[0, 0] == pytest.approx(50 / 150)
    assert dice[0, 0] == pytest.approx(100 / 200)


def test_pq_perfect_and_empty():
    gt = {"010101-img": [Crop.from_full(box(0, 0, 20, 20)), Crop.from_full(box(100, 100, 20, 20))]}
    res, _ = evaluate(gt, {"img": gt["010101-img"]})
    assert res.pq == pytest.approx(1.0) and res.tp == 2
    res, _ = evaluate(gt, {})
    assert res.pq == 0 and res.fn == 2


def test_pq_definition_with_fp_fn_and_multiple_annotators():
    g1 = Crop.from_full(box(0, 0, 10, 10))
    g2 = Crop.from_full(box(500, 500, 10, 10))
    p1 = Crop.from_full(box(0, 0, 10, 8))           # IoU 0.8 with g1
    p2 = Crop.from_full(box(900, 900, 10, 10))      # false positive
    gt = {"010101-img": [g1, g2], "010102-img": [g1]}
    res, df = evaluate(gt, {"img": [p1, p2]})
    # annotator 1: TP(0.8), FN(g2), FP(p2); annotator 2: TP(0.8), FP(p2)
    assert (res.tp, res.fp, res.fn) == (2, 2, 1)
    assert res.pq == pytest.approx(1.6 / (2 + 0.5 * 2 + 0.5 * 1))
    assert len(df) == 2


def test_iou_exactly_half_is_not_a_match():
    g = Crop.from_full(box(0, 0, 10, 10))
    p = Crop.from_full(box(0, 0, 10, 5))            # IoU = 0.5 -> not a TP (strict >)
    res = pq_from_overlaps([(overlap_matrices([g], [p])[0], 1, 1)])
    assert res.tp == 0 and res.fp == 1 and res.fn == 1


def test_evaluate_frames_official_format():
    m = box(10, 10, 30, 30)
    gt = pd.DataFrame({"filament_id": ["010101-img_1"], "segmentation_rle": [rle.encode(m)]})
    pred = pd.DataFrame({"filament_id": ["img_1"], "segmentation_rle": [rle.encode(m)]})
    res, _ = evaluate_frames(gt, pred)
    assert res.pq == pytest.approx(1.0)


def test_postprocess_components_and_filters():
    prob = np.zeros((H, W), np.float32)
    prob[100:140, 100:300] = 0.9                    # large confident filament
    prob[500:504, 500:504] = 0.9                    # tiny blob -> removed by min_area
    prob[800:840, 800:1000] = 0.55                  # weak component -> removed by min_max_prob
    lab, stats = instances_from_prob(prob, PostConfig(threshold=0.5, min_area=50, min_max_prob=0.7))
    assert lab.max() == 1 and len(stats) == 1
    crops = label_to_crops(lab)
    assert crops[0].area == 40 * 200


def test_fill_holes_and_merge_distance():
    prob = np.zeros((H, W), np.float32)
    prob[100:140, 100:200] = 0.9
    prob[115:125, 140:160] = 0.0                    # hole
    prob[100:140, 205:300] = 0.9                    # second fragment, gap of 5 px
    lab, _ = instances_from_prob(prob, PostConfig(threshold=0.5, min_area=10))
    assert lab.max() == 2 and (lab[120, 150] > 0)
    lab, _ = instances_from_prob(prob, PostConfig(threshold=0.5, min_area=10, merge_distance=8))
    assert lab.max() == 1


def test_grow_labels_keeps_instances_disjoint():
    lab = np.zeros((64, 64), np.int32)
    lab[10:20, 10:20] = 1
    lab[10:20, 23:33] = 2
    g = grow_labels(lab, 3)
    assert set(np.unique(g)) == {0, 1, 2}
    assert (g == 1).sum() > 100 and (g == 2).sum() > 100


def test_scorer_parallel_matches_sequential():
    from filseg.tune import Scorer
    rng = np.random.default_rng(0)
    probs, gt = {}, {}
    for i in range(3):
        p = np.zeros((256, 256), np.float16)
        y, x = rng.integers(10, 200, 2)
        p[y:y + 20, x:x + 40] = 0.9
        probs[f"img{i}"] = p
        full = np.zeros((H, W), bool)
        full[y * 8:(y + 20) * 8, x * 8:(x + 40) * 8] = True
        gt[f"010101-img{i}"] = [Crop.from_full(full)]
    cfg = PostConfig(threshold=0.5, min_area=10)
    seq = Scorer(probs, gt, workers=1)(cfg)
    s = Scorer(probs, gt, workers=2)
    try:
        par = s(cfg)
    finally:
        s.close()
    assert (seq.tp, seq.fp, seq.fn) == (par.tp, par.fp, par.fn) == (3, 0, 0)
    assert seq.pq == pytest.approx(par.pq) and seq.pq > 0.9


def test_component_decomposition_equals_official_pq():
    from filseg.components import build
    from filseg.tune import Scorer
    rng = np.random.default_rng(1)
    probs, gt = {}, {}
    for i in range(3):
        p = np.zeros((256, 256), np.float16)
        entries = {f"01010{a}-img{i}": [] for a in (1, 2)}
        for _ in range(4):
            y, x = rng.integers(10, 220, 2)
            hh, ww = rng.integers(4, 20, 2)
            p[y:y + hh, x:x + ww] = rng.uniform(0.55, 0.95)
            for a, e in enumerate(entries):
                if rng.random() < 0.8:
                    full = np.zeros((H, W), bool)
                    full[y * 8 + a * 3:(y + hh) * 8, x * 8:(x + ww) * 8 + a * 5] = True
                    entries[e].append(Crop.from_full(full))
        probs[f"img{i}"] = p
        gt.update({e: c for e, c in entries.items() if c})
    shape = PostConfig(threshold=0.5, min_area=0)
    cs = build(probs, gt, shape, workers=1)
    for min_area, min_max in [(0, 0.0), (3000, 0.0), (0, 0.8), (6000, 0.7)]:
        keep = cs.rule(min_area=min_area, min_max_prob=min_max)
        pq_fast, tp, fp, fn = cs.pq(keep)
        ref = Scorer(probs, gt, workers=1)(PostConfig(threshold=0.5, min_area=min_area, min_max_prob=min_max))
        assert pq_fast == pytest.approx(ref.pq, abs=1e-9)
        assert (tp, fp, fn) == (ref.tp, ref.fp, ref.fn)


def test_small_instance_weights():
    from filseg.data import Entry
    from filseg.train import small_instance_weights
    small = [100.0, 100.0, 110.0, 100.0, 110.0, 110.0, 100.0, 110.0, 100.0, 100.0]
    big = [500.0, 500.0, 600.0, 500.0, 600.0, 600.0, 500.0, 600.0, 500.0, 500.0]
    e = Entry("010101-img", "010101", "img", "img.jpeg", [small, big], [1, 1])
    w = small_instance_weights([e], gamma=0.5, margin=3)
    assert w[105, 105] == pytest.approx((1200 / rle.polygon_to_mask(small).sum()) ** 0.5, rel=1e-5)
    assert w[105, 98] == w[105, 105]              # margin band
    assert w[550, 550] == 1.0 and w[50, 50] == 1.0


def test_tta_inverse_transforms():
    import torch
    from filseg.model import predict_batch
    x = torch.randn(2, 1, 32, 32)
    for tta in (1, 2, 4, 8):
        p = predict_batch(torch.nn.Identity(), x, tta=tta)
        assert torch.allclose(p, torch.sigmoid(x), atol=1e-6)


def test_compact_checkpoint_roundtrip(tmp_path):
    import torch
    from filseg.model import build_model, load_checkpoint, save_checkpoint
    torch.manual_seed(0)
    m = build_model("Unet", "resnet18", weights=None).eval()
    save_checkpoint(m, tmp_path / "m.pt", arch="Unet", encoder="resnet18", res=64)
    m2, meta = load_checkpoint(tmp_path / "m.pt")
    assert meta["res"] == 64 and meta["encoder"] == "resnet18"
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        d = (torch.sigmoid(m(x)) - torch.sigmoid(m2(x))).abs().max().item()
    assert d < 5e-3


def test_submission_check_flags_overlaps_and_duplicates():
    from filseg.submission import check, rows_from_label
    lab = np.zeros((H, W), np.int32)
    lab[10:20, 10:20] = 1
    lab[50:60, 50:60] = 2
    rows = rows_from_label("img", lab)
    assert [r[0] for r in rows] == ["img_1", "img_2"]
    ok = check(pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]), {"img", "other"})
    assert ok["problems"] == [] and ok["test_images_without_prediction"] == 1
    bad = pd.DataFrame(rows + [("img_3", rows[0][1])], columns=["filament_id", "segmentation_rle"])
    assert any("overlapping" in p for p in check(bad)["problems"])
    dup = pd.DataFrame(rows + [rows[0]], columns=["filament_id", "segmentation_rle"])
    assert any("duplicated" in p for p in check(dup)["problems"])


def test_crop_geometric_keeps_alignment():
    from filseg import augment
    rng = np.random.default_rng(0)
    img = np.zeros((512, 512), np.uint8)
    tgt = np.zeros((512, 512), np.uint8)
    img[200:260, 300:310] = 200
    tgt[200:260, 300:310] = 255
    for _ in range(20):
        a, t = augment.crop_geometric([img, tgt], 128, rng, focus=1, p_focus=1.0)
        assert a.shape == t.shape == (128, 128) and a.dtype == np.uint8
        assert t.max() > 0                       # focus crop contains the filament
        on = t > 127
        assert a[on].mean() > 150               # image and target stay aligned
    w = augment.window(img, 5, 5, 32)
    assert w.shape == (32, 32) and w[:11, :11].sum() == 0


def test_ema_warmup_tracks_model_early():
    import torch
    from torch.optim.swa_utils import AveragedModel
    from filseg.train import ema_with_warmup
    m = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(m.weight)
    ema = AveragedModel(m, multi_avg_fn=ema_with_warmup(0.999), use_buffers=True)
    for _ in range(20):
        m.weight.data += 1.0
        ema.update_parameters(m)
    # plain EMA(0.999) would still be ~0.2; with warm-up the average follows the trajectory
    assert ema.module.weight.item() > 8.0
    # networks with BatchNorm have integer buffers (num_batches_tracked) that must not break the update
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.BatchNorm2d(2))
    ema2 = AveragedModel(net, multi_avg_fn=ema_with_warmup(0.999), use_buffers=True)
    for _ in range(3):
        net(torch.randn(2, 1, 8, 8))
        ema2.update_parameters(net)
    assert int(ema2.module[1].num_batches_tracked) == 3


def test_disk_normalisation_and_prediction_path():
    import torch
    from filseg.model import image_stats, predict_images
    yy, xx = np.mgrid[:2048, :2048]
    img = np.zeros((2048, 2048), np.uint8)
    disk = (yy - 1024) ** 2 + (xx - 1024) ** 2 < 950 ** 2
    img[disk] = 120
    img[disk & (xx < 1024)] = 140
    m, s = image_stats(img, "disk")
    assert 120 / 255 < m < 140 / 255 and s > 0.01
    assert image_stats(img, "global") == (0.45, 0.225)
    p = predict_images([torch.nn.Identity()], [img], 64, torch.device("cpu"), tta=2, norm="disk")
    assert p.shape == (1, 64, 64) and p.dtype == np.float16


def test_component_subset_pq_and_nested_select():
    from filseg.components import build, nested_select
    from filseg.tune import Scorer
    rng = np.random.default_rng(2)
    probs, gt = {}, {}
    for i in range(6):
        p = np.zeros((256, 256), np.float16)
        for _ in range(3):
            y, x = rng.integers(10, 220, 2)
            hh, ww = rng.integers(4, 20, 2)
            p[y:y + hh, x:x + ww] = rng.uniform(0.55, 0.99)
            if rng.random() < 0.8:
                full = np.zeros((H, W), bool)
                full[y * 8:(y + hh) * 8, x * 8:(x + ww) * 8 + 4] = True
                gt.setdefault(f"010101-img{i}", []).append(Crop.from_full(full))
        probs[f"img{i}"] = p
    cs = build(probs, gt, PostConfig(threshold=0.5, min_area=0), workers=1)
    subset = ["img0", "img2", "img5"]
    keep = cs.rule(min_area=2000)
    ref = Scorer({k: probs[k] for k in subset}, gt, workers=1)(PostConfig(threshold=0.5, min_area=2000))
    pq, tp, fp, fn = cs.pq(keep, subset)
    assert pq == pytest.approx(ref.pq, abs=1e-9) and (tp, fp, fn) == (ref.tp, ref.fp, ref.fn)
    folds = {f"img{i}": i % 3 for i in range(6)}
    res = nested_select({"t0.5": cs}, folds, {"min_area": [0, 2000], "min_max_prob": [0.0, 0.9], "min_mean_prob": [0.0]})
    assert len(res["fold_choices"]) == 3 and 0.0 <= res["nested_pq"] <= 1.0
    assert res["final_rule"]["min_area"] in (0, 2000)


def test_cv_argument_split(capsys):
    from filseg import cv
    cv.main(["--folds", "1", "2", "--out", "o", "--max-hours", "2", "--dry-run", "--",
             "--encoder", "resnet34", "--bs", "8", "--predict-test"])
    out = capsys.readouterr().out
    assert "--fold 1 --out o" in out.replace("\\", "/") and "--fold 2" in out
    assert out.count("--encoder resnet34 --bs 8 --predict-test") == 2 and "--max-hours 1.000" in out


def test_pq_sparse_reduces_to_pq():
    from filseg.components import pq_sparse
    assert pq_sparse(60.0, 100, 40, 50, 1.0) == pytest.approx(60.0 / (100 + 20 + 25))
    assert pq_sparse(60.0, 100, 40, 50, 0.7) < pq_sparse(60.0, 100, 40, 50, 1.0)


def test_upsample_uint8_and_load_probs(tmp_path):
    import sys
    from pathlib import Path
    from filseg.postprocess import upsample
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_oof import load_probs
    u = np.full((4, 4), 255, np.uint8)
    assert upsample(u, 8).max() == pytest.approx(1.0) and upsample(u, 8).shape == (8, 8)
    assert upsample(np.full((4, 4), 0.5, np.float16), 4).max() == pytest.approx(0.5)
    np.savez_compressed(tmp_path / "a.npz", img1=np.full((4, 4), 200, np.uint8), img2=np.zeros((4, 4), np.uint8))
    np.savez_compressed(tmp_path / "b.npz", img1=np.full((8, 8), 100, np.uint8))   # other encoder, larger res
    np.savez_compressed(tmp_path / "c.npz", img3=np.full((4, 4), 50, np.uint8))    # another fold
    p = load_probs([tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"])
    assert set(p) == {"img1", "img2", "img3"} and p["img1"].dtype == np.uint8
    assert p["img1"].shape == (8, 8) and int(p["img1"][0, 0]) == 150 and int(p["img3"][0, 0]) == 50


def test_cached_arrays_roundtrip(tmp_path):
    import cv2
    from filseg.data import Entry
    from filseg.train import cached_arrays
    img_dir = tmp_path / "data" / "train" / "train_images"
    img_dir.mkdir(parents=True)
    yy, xx = np.mgrid[:2048, :2048]
    groups = {}
    for i in range(2):
        img = np.where((yy - 1024) ** 2 + (xx - 1024) ** 2 < 950 ** 2, 120 + 10 * i, 0).astype(np.uint8)
        cv2.imwrite(str(img_dir / f"img{i}.jpeg"), img)
        poly = [500.0, 500.0, 700.0, 500.0, 700.0, 560.0, 500.0, 560.0, 500.0, 500.0]
        groups[f"img{i}"] = [Entry(f"010101-img{i}", "010101", f"img{i}", f"img{i}.jpeg", [poly], [1])]
    msgs = []
    first = cached_arrays(tmp_path / "cache", tmp_path / "data", groups, 256, 0, 0.0, "disk", msgs.append)
    second = cached_arrays(tmp_path / "cache", tmp_path / "data", groups, 256, 0, 0.0, "disk", msgs.append)
    assert any("cached in" in m for m in msgs) and any("loading prepared arrays" in m for m in msgs)
    assert isinstance(first[0], np.memmap) and isinstance(second[0], np.memmap) and second[2] is None
    for a, b in zip(first, second):
        assert (a is None and b is None) or np.array_equal(np.asarray(a), np.asarray(b))
    assert first[1].max() == 255 and first[3].shape == (2, 2)
    other = cached_arrays(tmp_path / "cache", tmp_path / "data", {"img0": groups["img0"]}, 256, 0, 0.0, "disk",
                          msgs.append)
    assert len(other[0]) == 1          # a different image list never reuses the cache


def test_refine_extent_grow_and_shrink():
    from filseg.postprocess import instances_from_prob
    prob = np.zeros((128, 128), np.float32)
    prob[20:30, 20:60] = 0.9          # instance A core
    prob[20:30, 70:100] = 0.9         # instance B core
    prob[18:32, 15:105] = np.maximum(prob[18:32, 15:105], 0.3)   # faint halo joining A and B
    prob[80:84, 80:84] = 0.6          # weak instance C (vanishes when shrinking to 0.7)
    base, _ = instances_from_prob(prob, PostConfig(threshold=0.5, min_area=1))
    assert base.max() == 3
    grown, _ = instances_from_prob(prob, PostConfig(threshold=0.5, min_area=1, mask_threshold=0.2))
    assert grown.max() == 3 and (grown > 0).sum() > (base > 0).sum()
    assert (grown[18:32, 15:105] > 0).all()                          # halo fully assigned ...
    assert set(np.unique(grown[18:32, 15:105])) == {1, 2}            # ... to A or B, never shared
    shrunk, _ = instances_from_prob(prob, PostConfig(threshold=0.5, min_area=1, mask_threshold=0.7))
    assert shrunk.max() == 2 and set(np.unique(shrunk)) == {0, 1, 2}  # C dropped, labels renumbered
