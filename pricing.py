"""Recombining binomial option-pricing model.

Supports European and American exercise and call, put, or custom state-based
payoffs. The terminal exercise payoff is g(S, K); American contracts may
exercise for that same payoff at any earlier tree node.
"""

import ast
import math
import operator


# Only arithmetic, comparisons, conditionals, and these math functions are
# supported in a user-provided payoff. No Python eval() is used.
_ALLOWED_FUNCTIONS = {
    "max": max,
    "min": min,
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
}
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_COMPARISONS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _calculate_expression(node, names):
    """Evaluate a small, restricted mathematical expression AST."""
    if isinstance(node, ast.Expression):
        return _calculate_expression(node.body, names)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _calculate_expression(node.left, names)
        right = _calculate_expression(node.right, names)
        if isinstance(node.op, ast.Pow) and abs(right) > 50:
            raise ValueError("Custom payoff powers must have exponent <= 50 in magnitude.")
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calculate_expression(node.operand, names)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.Compare) and all(type(op) in _COMPARISONS for op in node.ops):
        left = _calculate_expression(node.left, names)
        for op, comparator in zip(node.ops, node.comparators):
            right = _calculate_expression(comparator, names)
            if not _COMPARISONS[type(op)](left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.IfExp):
        test = _calculate_expression(node.test, names)
        return _calculate_expression(node.body if test else node.orelse, names)

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _ALLOWED_FUNCTIONS
        and not node.keywords
    ):
        args = [_calculate_expression(arg, names) for arg in node.args]
        return _ALLOWED_FUNCTIONS[node.func.id](*args)

    raise ValueError(
        "Unsupported payoff expression. Use S, K, arithmetic, comparisons, "
        "'a if condition else b', and supported math functions."
    )


def make_payoff(payoff_type="Call", payoff_expression=None):
    """Return a callable g(S, K) for a call, put, or custom expression.

    Custom examples:
        max(S - K, 0)   -- call
        max(K - S, 0)   -- put
        1 if S > K else 0 -- cash-or-nothing digital payoff
        abs(S - K)      -- straddle payoff at maturity
    """
    if payoff_type == "Call":
        return lambda stock, strike: max(stock - strike, 0.0)
    if payoff_type == "Put":
        return lambda stock, strike: max(strike - stock, 0.0)
    if payoff_type != "Custom":
        raise ValueError("Payoff type must be Call, Put, or Custom.")

    if not isinstance(payoff_expression, str) or not payoff_expression.strip():
        raise ValueError("Enter a custom payoff expression using S and K.")
    if len(payoff_expression) > 200:
        raise ValueError("Custom payoff expression must be at most 200 characters.")

    try:
        parsed = ast.parse(payoff_expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid custom payoff expression syntax.") from exc
    if sum(1 for _ in ast.walk(parsed)) > 80:
        raise ValueError("Custom payoff expression is too complex.")

    def payoff(stock, strike):
        try:
            result = _calculate_expression(
                parsed, {"S": stock, "K": strike, "pi": math.pi, "e": math.e}
            )
            if not isinstance(result, (bool, int, float)):
                raise ValueError("The payoff must evaluate to a number.")
            value = float(result)
            if not math.isfinite(value):
                raise ValueError("The payoff must be finite at every stock-price node.")
            return value
        except (ArithmeticError, OverflowError, TypeError) as exc:
            raise ValueError(f"Custom payoff is undefined at S={stock:.4g}: {exc}") from exc

    # Validate all expression branches structurally, even if a given S value
    # wouldn't take that branch (e.g., conditional digital payoffs).
    def validate(node):
        if isinstance(node, ast.Expression):
            return validate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return
        if isinstance(node, ast.Name) and node.id in ("S", "K", "pi", "e"):
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            validate(node.left)
            validate(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            validate(node.operand)
            return
        if isinstance(node, ast.Compare) and all(type(op) in _COMPARISONS for op in node.ops):
            validate(node.left)
            for value in node.comparators:
                validate(value)
            return
        if isinstance(node, ast.IfExp):
            validate(node.test)
            validate(node.body)
            validate(node.orelse)
            return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _ALLOWED_FUNCTIONS and not node.keywords):
            for arg in node.args:
                validate(arg)
            return
        raise ValueError("Unsupported custom payoff expression or function.")

    validate(parsed)
    return payoff


def black_scholes_price(
    S: float,
    K: float,
    sigma: float,
    r: float,
    maturity: float,
    payoff_type: str = "Call",
) -> float:
    """Return the Black–Scholes–Merton price of a European vanilla option.

    S is the current underlying price, K the strike, sigma the annualized
    volatility, r the continuously compounded annual risk-free rate, and
    maturity the remaining time to expiry in years. Assumes no dividends.
    Returns a per-share price, consistent with binomial_pricing().
    """
    if payoff_type not in ("Call", "Put"):
        raise ValueError("BSM benchmark supports only European calls and puts.")
    if not all(math.isfinite(v) for v in (S, K, sigma, r, maturity)):
        raise ValueError("BSM inputs must be finite.")
    if S <= 0 or K < 0 or sigma <= 0 or maturity < 0:
        raise ValueError("Require S > 0, K >= 0, sigma > 0 and maturity >= 0.")

    if maturity == 0:
        return max(S - K, 0.0) if payoff_type == "Call" else max(K - S, 0.0)
    if K == 0:
        return S if payoff_type == "Call" else 0.0

    root_time = math.sqrt(maturity)
    vol_time = sigma * root_time
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * maturity) / vol_time
    d2 = d1 - vol_time
    discount_strike = K * math.exp(-r * maturity)

    def standard_normal_cdf(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    if payoff_type == "Call":
        price = S * standard_normal_cdf(d1) - discount_strike * standard_normal_cdf(d2)
    else:
        price = discount_strike * standard_normal_cdf(-d2) - S * standard_normal_cdf(-d1)
    return max(0.0, price)  # Avoid microscopic negative round-off in deep OTM tails.


def binomial_pricing(
    T: int,
    S: float,
    K: float,
    sigma: float,
    r: float,
    dt: float,
    option_style: str = "European",
    payoff_type: str = "Call",
    payoff_expression: str | None = None,
    return_exercise: bool = False,
):
    """Compute stock, option-value, and delta trees.

    T counts levels, not steps: maturity is (T - 1) * dt.
    At level k, index j is the number of downward moves.
    Uses the Cox-Ross-Rubinstein (CRR) factors:
        u = exp(sigma * sqrt(dt)), d = exp(-sigma * sqrt(dt)).
    sigma is annualized volatility (decimal, e.g. 0.20 for 20%).
    p is the risk-neutral up probability.

    If return_exercise=True, also return a Boolean tree marking the nodes
    where immediate exercise is strictly preferable for an American option.
    The original six-argument call still returns the original three trees.
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 2:
        raise ValueError("T must be an integer of at least 2 levels.")
    if not all(math.isfinite(v) for v in (S, K, sigma, r, dt)):
        raise ValueError("Model parameters must be finite numbers.")
    if S <= 0:
        raise ValueError("Initial stock price S must be positive.")
    if K < 0:
        raise ValueError("Strike price K cannot be negative.")
    if sigma <= 0:
        raise ValueError("Annualized volatility sigma must be positive.")
    if dt <= 0:
        raise ValueError("Time interval dt must be positive.")
    if option_style not in ("European", "American"):
        raise ValueError("Option style must be European or American.")

    # CRR factors depend on volatility and the duration of one time step.
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    growth = math.exp(r * dt)
    if not (d < growth < u):
        raise ValueError(
            "No-arbitrage condition violated: require "
            "exp(-sigma*sqrt(dt)) < exp(r*dt) < exp(sigma*sqrt(dt))."
        )

    payoff = make_payoff(payoff_type, payoff_expression)
    p = (growth - d) / (u - d)
    q = 1 - p
    discount = math.exp(-r * dt)

    def create_tree(levels, fill=0.0):
        return [[fill] * (k + 1) for k in range(levels)]

    stock_values = create_tree(T)
    option_values = create_tree(T)
    delta_values = create_tree(T - 1)
    exercise_nodes = create_tree(T - 1, False)

    for k in range(T):
        for j in range(k + 1):
            stock_values[k][j] = S * u ** (k - j) * d ** j

    for j in range(T):
        option_values[T - 1][j] = payoff(stock_values[T - 1][j], K)

    for k in range(T - 2, -1, -1):
        for j in range(k + 1):
            continuation = discount * (
                p * option_values[k + 1][j]
                + q * option_values[k + 1][j + 1]
            )

            if option_style == "American":
                exercise_value = payoff(stock_values[k][j], K)
                exercise_nodes[k][j] = exercise_value > continuation + 1e-12
                option_values[k][j] = max(exercise_value, continuation)
            else:
                option_values[k][j] = continuation

            delta_values[k][j] = (
                option_values[k + 1][j] - option_values[k + 1][j + 1]
            ) / (
                stock_values[k + 1][j] - stock_values[k + 1][j + 1]
            )

    if return_exercise:
        return stock_values, option_values, delta_values, exercise_nodes
    return stock_values, option_values, delta_values



def exercise_time_distribution(
    stock_values,
    exercise_nodes,
    up_probability: float,
    strike: float,
    payoff_type: str = "Put",
    payoff_expression: str | None = None,
):
    """Exact risk-neutral distribution of the *first* optimal exercise time.

    Works for a recombining CRR tree with the American exercise policy supplied
    by binomial_pricing(return_exercise=True). No Monte Carlo or explicit
    enumeration of the 2**(T-1) paths is necessary.

    If terminal exercise payoff is positive, exercise at maturity. Otherwise
    count that surviving path as expiring without exercise. For the usual
    nonnegative call/put payoffs this has the standard option interpretation.

    The result contains the probability of exercise at each time step (index 0
    corresponds to today), plus the separate mass of paths expiring unexercised.
    """
    levels = len(stock_values)
    if levels < 2 or len(exercise_nodes) != levels - 1:
        raise ValueError("Stock and exercise trees have inconsistent numbers of levels.")
    if any(len(row) != i + 1 for i, row in enumerate(stock_values)):
        raise ValueError("The stock tree must have a recombining triangular shape.")
    if any(len(row) != i + 1 for i, row in enumerate(exercise_nodes)):
        raise ValueError("The exercise tree must have a triangular shape.")
    if not math.isfinite(up_probability) or not 0 <= up_probability <= 1:
        raise ValueError("Risk-neutral up probability must be between zero and one.")

    payoff = make_payoff(payoff_type, payoff_expression)
    down_probability = 1.0 - up_probability
    probabilities = [0.0] * levels
    alive = [1.0]  # Probability mass reaching nodes without prior exercise

    # Record the probability arriving at each node before deciding whether to
    # exercise there, plus the probability that actually exercises there.
    arrival_mass_tree = []
    exercise_mass_tree = []

    for k in range(levels - 1):
        arrival_mass_tree.append(alive[:])
        next_alive = [0.0] * (k + 2)
        exercised_here = [0.0] * (k + 1)
        for j, mass in enumerate(alive):
            if exercise_nodes[k][j]:
                probabilities[k] += mass
                exercised_here[j] = mass
            else:
                next_alive[j] += mass * up_probability
                next_alive[j + 1] += mass * down_probability
        exercise_mass_tree.append(exercised_here)
        alive = next_alive

    # Remaining mass is at maturity: an in-the-money contract is exercised,
    # while a zero-payoff contract expires without exercise.
    arrival_mass_tree.append(alive[:])
    exercised_at_maturity = [0.0] * levels
    expired_unexercised = 0.0
    for j, mass in enumerate(alive):
        if payoff(stock_values[-1][j], strike) > 0:
            probabilities[-1] += mass
            exercised_at_maturity[j] = mass
        else:
            expired_unexercised += mass
    exercise_mass_tree.append(exercised_at_maturity)

    return {
        "exercise_probabilities": probabilities,
        "early_exercise_probability": sum(probabilities[:-1]),
        "maturity_exercise_probability": probabilities[-1],
        "expired_unexercised_probability": expired_unexercised,
        "arrival_mass_tree": arrival_mass_tree,
        "exercise_mass_tree": exercise_mass_tree,
    }


def simulate_delta_hedge(
    stock_values,
    option_values,
    delta_values,
    path: str,
    r: float,
    dt: float,
    *,
    option_style: str = "European",
    exercise_nodes=None,
    multiplier: int = 100,
):
    """Follow a self-financing, long-stock/cash hedge of a short option.

    The stock position is multiplier * tree_delta at every *continuation*
    node. The option premium initially finances the stock position, with any
    deficit borrowed at the risk-free rate. When stock price changes, the cash
    account accrues interest, then the stock position is rebalanced by trading
    only the *difference* in desired holdings.

    The modeled option holder exercises optimally. For American options, stop
    at the first node selected by backward induction; otherwise stop at expiry.
    The returned rows are a per-step ledger for ONE deterministic U/D path.

    This is exact in the frictionless CRR model (up to floating-point error).
    In practice, transaction costs, discrete price jumps, dividends, and
    exercise choices not matching the optimal policy introduce differences.

    T denotes the number of tree levels; the path has T-1 steps. At each
    level, j counts the number of DOWN moves already taken.
    """
    levels = len(stock_values)
    if levels < 2:
        raise ValueError("The stock tree needs at least two levels.")
    for tree, expected_length, name in (
        (stock_values, levels, "Stock"),
        (option_values, levels, "Option"),
        (delta_values, levels - 1, "Delta"),
    ):
        if len(tree) != expected_length or any(len(row) != k + 1 for k, row in enumerate(tree)):
            raise ValueError(f"{name} tree has inconsistent dimensions.")
    if option_style not in ("European", "American"):
        raise ValueError("Exercise style must be European or American.")
    if option_style == "American":
        if exercise_nodes is None or len(exercise_nodes) != levels - 1 or any(
            len(row) != k + 1 for k, row in enumerate(exercise_nodes)
        ):
            raise ValueError("Provide a valid exercise-decision tree for American options.")
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier <= 0:
        raise ValueError("Contract multiplier must be a positive integer.")
    if not math.isfinite(r) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Use finite interest and a positive time increment.")
    if not isinstance(path, str):
        raise ValueError("Path must be a string containing U or D.")
    moves = ''.join(path.upper().split())
    if len(moves) != levels - 1 or any(move not in 'UD' for move in moves):
        raise ValueError(f"Path must contain exactly {levels - 1} U/D moves.")

    interest_growth = math.exp(r * dt)
    premium = multiplier * option_values[0][0]
    rows = []
    node = 0

    # An optimally exercised American contract at t=0 is settled immediately.
    if option_style == "American" and exercise_nodes[0][0]:
        return [{
            "Step": 0,
            "Time (years)": 0.0,
            "Move": "Start",
            "Node j": 0,
            "Stock price": stock_values[0][0],
            "Shares held": 0.0,
            "Shares traded": 0.0,
            "Cash balance": premium,
            "Hedge value": premium,
            "Option liability": premium,
            "Hedging error": 0.0,
            "Action": "Immediate American exercise",
        }]

    shares = multiplier * delta_values[0][0]
    cash = premium - shares * stock_values[0][0]

    def add_row(step, move, traded, action):
        price = stock_values[step][node]
        hedge_value = shares * price + cash
        option_value = multiplier * option_values[step][node]
        rows.append({
            "Step": step,
            "Time (years)": step * dt,
            "Move": move,
            "Node j": node,
            "Stock price": price,
            "Shares held": shares,
            "Shares traded": traded,
            "Cash balance": cash,
            "Hedge value": hedge_value,
            "Option liability": option_value,
            "Hedging error": hedge_value - option_value,
            "Action": action,
        })

    add_row(0, "Start", shares, "Establish initial delta hedge")

    for step, move in enumerate(moves, start=1):
        cash *= interest_growth
        node += int(move == 'D')
        price = stock_values[step][node]

        if step == levels - 1:
            add_row(step, move, 0.0, "Maturity / option settlement")
            break
        if option_style == "American" and exercise_nodes[step][node]:
            add_row(step, move, 0.0, "Optimal early exercise / settlement")
            break

        target_shares = multiplier * delta_values[step][node]
        trade = target_shares - shares
        cash -= trade * price  # Buy shares with cash, or credit sale proceeds.
        shares = target_shares
        add_row(step, move, trade, "Rebalance delta hedge")

    return rows



def calculate_greeks(
    T: int,
    S: float,
    K: float,
    sigma: float,
    r: float,
    dt: float,
    *,
    option_style: str = "European",
    payoff_type: str = "Call",
    payoff_expression: str | None = None,
):
    """Estimate root Greeks for the option selected in the CRR model.

    - delta_hedge: exact one-step replication ratio at the root. It is NOT
      necessarily dV/dS when the lattice is held fixed and spot is bumped.
    - delta_spot: central difference of the initial option value vs spot.
    - gamma: difference of first-step hedge deltas divided by the first-step
      up/down stock-price difference (requires at least 2 transitions).
      This tree-based approximation avoids the near-zero finite-difference
      gamma of piecewise-linear prices at a fixed finite lattice.
    - vega: bump-and-reprice, reported per 1 vol point (sigma + 0.01).
    - rho: bump-and-reprice, reported per 1 basis point (r + 0.0001).
    - theta: negative maturity derivative, reported per calendar day.

    Every bump keeps the exercise type and payoff fixed and reruns backward
    induction. Theta keeps the number of steps fixed and changes dt to vary
    maturity. The model assumes no dividends.

    Sensitivities are per SHARE of option exposure. Multiply by the contract
    multiplier (typically 100) to obtain per-contract dollar sensitivities.
    """
    stock, values, deltas = binomial_pricing(
        T, S, K, sigma, r, dt,
        option_style=option_style,
        payoff_type=payoff_type,
        payoff_expression=payoff_expression,
    )
    base = values[0][0]
    n_steps = T - 1
    maturity = n_steps * dt

    def reprice(**overrides):
        params = {
            "T": T, "S": S, "K": K, "sigma": sigma,
            "r": r, "dt": dt, "option_style": option_style,
            "payoff_type": payoff_type,
            "payoff_expression": payoff_expression,
        }
        params.update(overrides)
        return binomial_pricing(**params)[1][0][0]

    def centered(which, center, bump, transform=None):
        """Return central derivative, or None if the requested bump is invalid."""
        if bump <= 0 or not math.isfinite(bump):
            return None
        transform = transform or (lambda x: x)
        try:
            upper = reprice(**{which: transform(center + bump)})
            lower = reprice(**{which: transform(center - bump)})
            return (upper - lower) / (2.0 * bump)
        except (ValueError, ArithmeticError, OverflowError):
            return None

    # A finite tree is piecewise-linear in S; compute Gamma from its node
    # deltas, not by taking an extremely small second spot difference.
    tree_gamma = None
    if T >= 3:
        stock_gap = stock[1][0] - stock[1][1]
        tree_gamma = (deltas[1][0] - deltas[1][1]) / stock_gap

    # Provide both the exact discrete hedge ratio and the spot-price
    # sensitivity, since these are not identical for a finite lattice.
    spot_bump = min(max(0.01, 0.001 * S), S * 0.4)
    spot_delta = centered("S", S, spot_bump)

    # Under CRR, no-arbitrage requires abs(r)*sqrt(dt) < sigma. Choose
    # symmetric bumps that remain inside this open parameter region.
    sigma_margin = sigma - abs(r) * math.sqrt(dt)
    sigma_bump = min(0.001, 0.25 * sigma_margin, 0.25 * sigma)
    sigma_derivative = centered("sigma", sigma, sigma_bump)

    max_abs_rate = sigma / math.sqrt(dt)
    rate_margin = max_abs_rate - abs(r)
    rate_bump = min(0.0001, 0.25 * rate_margin)
    rate_derivative = centered("r", r, rate_bump)

    # Calendar-time theta is -dV/d(maturity). To vary maturity at a fixed
    # number of steps, adjust dt accordingly (and recalculate u,d,p).
    max_maturity = (
        n_steps * (sigma / abs(r)) ** 2 if r != 0 else math.inf
    )
    maturity_bump = min(
        1 / 365.0,
        0.25 * maturity,
        0.25 * (max_maturity - maturity),
    )
    maturity_derivative = centered(
        "dt", maturity, maturity_bump,
        transform=lambda new_maturity: new_maturity / n_steps,
    )

    return {
        "price": base,
        "delta_hedge": deltas[0][0],
        "delta_spot": spot_delta,
        "gamma": tree_gamma,
        "vega_per_vol_point": (
            0.01 * sigma_derivative if sigma_derivative is not None else None
        ),
        "rho_per_bp": (
            0.0001 * rate_derivative if rate_derivative is not None else None
        ),
        "theta_per_day": (
            -maturity_derivative / 365.0 if maturity_derivative is not None else None
        ),
        "bumps": {
            "spot_dollars": spot_bump,
            "sigma_decimal": sigma_bump,
            "rate_decimal": rate_bump,
            "maturity_years": maturity_bump,
        },
    }
