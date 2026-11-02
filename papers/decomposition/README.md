# Decomposing the cost of a metaorder

How much of implementation shortfall is liquidity the order consumed, and how much is the market's reaction to it, split into the
inventory response of makers, the response to the moved price, detection and predation? Then: how each share moves with participation,
horizon, schedule randomness and order type across 12 classical executors and a learned one, and whether the reaction channels are
what turn book consumption into a concave impact profile.

`Market.hide` alone cannot say. It only blinds the detector, so a hidden executor still moves the mid and every reaction to that sits in
the "mechanical" bucket. Every run here steps **Arm 0** (no executor) and a **treated world** in lockstep on counter-based draws, with four
reaction channels on independent switches (`order_book/counterfactual.py`):

| Channel | On | Off |
|---|---|---|
| `I` inventory | makers skew on the inventory the executor pushed onto them | makers skew on their Arm-0 inventory |
| `P` price | the background lives on the true book and reads its public state | the background acts in Arm 0; its order flow is replayed on a copy of the book |
| `D` detection | the makers' flow detector sees the executor | it does not (`Market.hide`) |
| `A` predation | the anticipator trades, on a detector that sees the executor | it stays dormant |

All off is the **ghost arm**: the executor fills against a live copy of Arm 0's book that nobody else can see (`order_book/ghost.py`
states the matching rule). All on is the full A4 market. For one seed, with v(S) the shortfall when channels S exist and D what the same
fills would have cost at Arm 0's mid:

    IS = natural drift + mechanical + reaction,   mechanical = v({}) - D,   reaction = v(all) - v({}),   share = reaction / (IS - D)

The reaction is split along the nested ladder of the plan and, because ladder increments depend on the order channels join, by Shapley
values over the 2^4 = 16 worlds, which add up to the reaction exactly per seed. Four more worlds outside the factorial look inside it: the
price arm with chartists, makers or fundamentalists blind to the move, and an anticipator that trades but cannot see the executor.
Inference is over seeds with paired bootstrap CIs and Holm.

| Arm | Channels | Treated world |
|---|---|---|
| 0 | | no executor |
| 1 | `----` | ghost: copy of the book, the background acts on Arm 0 |
| 2 | `-P--` | true book, reservation on Arm-0 inventory, detector blind |
| 3 | `IP--` | true book and inventory, detector blind (what E3 calls invisible) |
| 4 | `IPD-` | detector on (A1) |
| 5 | `IPDA` | anticipator (A4) |

The plan's table attaches "inventory" to the step into Arm 2 and "chartists" to the step into Arm 3, but Arm 2 as written freezes
inventory, so those two labels are swapped relative to what changes. Both orders are paths through the same cells and both are reported.

## Running it

The full run goes to the cluster: see `hpc/README.md`.

```sh
# on the cluster
bash hpc/submit.sh decomposition smoke_test
bash papers/decomposition/slurm/submit_all.sh all

# anywhere: a seconds-long end-to-end check, written to *_quick folders
python -m papers.decomposition.scripts.decompose --quick
python -m papers.decomposition.scripts.detection_curve --quick
python -m papers.decomposition.scripts.crn_variance --quick
python -m papers.decomposition.scripts.replications --quick

# tables and notes, from the committed pilot or from the full run
python -m papers.decomposition.scripts.make_tables
python -m papers.decomposition.scripts.make_tables --root outputs/decomposition --deep e8_deep_cover \
    --heldout outputs/stealth_execution/rl_heldout/summary.json
cd papers/decomposition/paper && latexmk -pdf notes.tex
```

On the cluster each grid is an array of 1-hour tasks, one seed each: `--shard K` runs the K-th seed and `--merge` combines them and writes the summary, refusing if a run is missing. Run on one
machine, `--resume` keeps the runs already in `episodes.jsonl`, so an interrupted job continues. Strategy names also accept `pov@0.05` (fixed-rate POV), `rl@<weights>`, and `twap@size=0.5,timing=0.3,passive=0.25` (a dialled TWAP:
zero dials is `twap_market`, `passive=1` alone is `twap_passive`).

| What | Module | Script | Output in `outputs/decomposition/` |
|---|---|---|---|
| E8: Shapley values, shares, exponents, executor routes | `decomposition.py` | `decompose` | `e8_*/` |
| The paired-seed protocol | `decomposition.py` | `crn_variance` | `protocol/` |
| Detection curve, randomisation value | `detection_curve.py` | `detection_curve` | `detection_curve*/` |
| Shi-Cartlidge, Bucci et al., ABIDES-MARL | `decomposition.py` | `replications` | `replications/` |

## Pilot results

20 seeds (40 for the protocol), one horizon, only two executors in the deep-cover market: enough to show the instrument works, not the paper's
grid. Tables and discussion are in `paper/notes.pdf`, generated from `results/pilot/`. Those runs predate manifests; they ran on the code just before
the move into `papers/`, on simulator version 1.

- **Reaction dominates.** For TWAP in the deep-cover market its share of cost net of drift rises from 0.44 at 1.7% realised Q/V to 0.89 at 35%.
  Mechanical cost is 0.5-0.7 bps, close to the half spread.
- **Most of it is the price channel.** Inside it, makers re-centring is a large part, fundamentalists hold the price down and chartists are about
  zero for TWAP. Detection and predation are small: at most about 0.5 and 0.07 bps for TWAP in that market.
- **No evidence yet that reaction produces the concavity.** The exponent on mean impact is 0.95 in Arm 1 and 0.93 in Arm 5.
- **Counter-based draws** keep 26-39% of background order flow shared against 6-12% for stateful ones. Variance reduction improves on the median
  and in 22 of 30 cells; Sharma's condition holds in all 30 under both schemes.
- **Timing jitter defeats the learned detector and leaves leakage unchanged**, because makers read the flow detector and nothing reacts to A3. Testing
  Yang and Zhu properly needs makers wired to a precise detector.
- **The replication hooks do not cleanly confirm their sources**, and a price runaway at very large sizes is a common factor to resolve first.
- **PPO has no gain to decompose.** On unseen seeds it is behind TWAP (`papers/stealth_execution/README.md`).

## Read before quoting a number

- The **price channel is not chartists**: it holds every reaction to the true book. Use `price_split`.
- **Nominal Q/V overstates participation**, because the order is sized on burn-in volume while volume is still rising. Fits use `q_over_v`, realised
  against Arm 0's volume over the horizon.
- The ghost **reverts completely by construction**, so test reversion among arms with a permanent component.
- `cost` includes the constant half spread, so its exponent looks flat in Arm 1. Say which measure a claim rests on.
- Claims attributed to Buffalo et al., Klein et al., Sharma, Shi-Cartlidge, Bucci et al., Gomes-Waelbroeck, Yang-Zhu and ABIDES-MARL come from the
  research plan and have not been checked against the papers.
