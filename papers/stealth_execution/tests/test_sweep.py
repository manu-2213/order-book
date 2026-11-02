"""The sweep's job tagging and ranking."""

from order_book.execution import new_policy


def test_sweep_job_tag_and_ranking(tmp_path, monkeypatch):
    import json
    from papers.stealth_execution.scripts import sweep_rl
    monkeypatch.setattr(sweep_rl, "OUT", tmp_path)
    job = sweep_rl.SweepJob(lr=1e-4, entropy=0.01, epochs=4, size=0.20,
                            iterations=2, episodes=2, institutions=1, horizon=200,
                            burn_in=200, eval_every=2, eval_seeds=2, seed=0)
    assert job.tag == "lr1e-4_ent0.01_ep4_sz0.20"
    assert not job.is_done()
    job.out_dir.mkdir(parents=True)
    new_policy(seed=0).save(job.out_dir / "policy.pt")
    (job.out_dir / "history.json").write_text(json.dumps({
        "config": {"size": 0.20},
        "baseline_twap": {"shortfall": 2.82},
        "evals": [{"iteration": 2, "shortfall": 2.40, "completion": 0.99}],
        "history": [{"iteration": 1}, {"iteration": 2}],
    }))
    assert job.is_done()
    ranking = sweep_rl.rank_sweep(tmp_path / "rl" / "sweep")
    assert "0.20" in ranking and ranking["0.20"][0]["tag"] == job.tag
    assert "lr1e-4_ent0.01_ep4_sz0.20" in sweep_rl.format_ranking(ranking)
