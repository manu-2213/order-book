import pickle

from order_book.env import OrderBookEnv, burn_in
from tests.helpers import assert_reservations_consistent


def test_env_is_seeded_bounded_and_picklable():
    states = burn_in([0, 1], steps=200)
    env = OrderBookEnv(states, decision_interval=5, horizon=3)
    obs, _ = env.reset(seed=4)
    assert obs.shape == env.observation_space.shape
    results = [env.step(a) for a in (1, 3, 4)]
    trajectory = [r[1] for r in results]
    assert [r[3] for r in results] == [False, False, True]  # truncated at the horizon
    clone = pickle.loads(pickle.dumps(OrderBookEnv(states, decision_interval=5, horizon=3)))
    clone.reset(seed=4)
    assert [clone.step(a)[1] for a in (1, 3, 4)] == trajectory
    env.sim.check_conservation()
    assert_reservations_consistent(env.sim.market)
