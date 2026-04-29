MARKET = "u"  # "u" for america, "t" for taiwan
TV_SORT_WINDOW = "1M"
FINMIND_THREADS = 2
LOOKBACK_DAYS = 28
CHUNK_SIZE = 64
CANDIDATE_POOL_MULTIPLIER = 2

DIRECTION = "high_growth_low_valuation"

LAST_N = 10

Q = 8  # 4?
PEAK_CUTOFF_RATIO = 1 / 2
PEAK_PAIR_MODE = "first"
RESULT_LIMIT = 384
# Supported values:
# - "high_growth_low_valuation"
# - "low_growth_high_valuation"
# PEAK_PAIR_MODE:
# - "first": require both values to be non-null, then use only the first
# - "average": require both values to be non-null, then average both
