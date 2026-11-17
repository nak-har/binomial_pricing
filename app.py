"""Streamlit frontend for the binomial option-pricing model."""

import math

import plotly.graph_objects as go
import streamlit as st

from pricing import (binomial_pricing, exercise_time_distribution, make_payoff,
                     simulate_delta_hedge, calculate_greeks)


st.set_page_config(
    page_title="Binomial Pricing Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown("""
<style>
/* Keep the sidebar toggle usable while preventing the top bar from obscuring content. */
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
}
/* Use the available space without hiding the sidebar toggle. */
[data-testid="stAppViewContainer"] .block-container {
    padding-top: .55rem !important;
}
/* A restrained light dashboard with an indigo/teal accent palette. */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 94% 4%, rgba(120, 160, 245, .13), transparent 33%),
        linear-gradient(155deg, #f8fbff 0%, #f1f6fd 60%, #eef8f7 100%);
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #edf2ff 0%, #f2f8fb 100%);
    border-right: 1px solid #d9e4f3;
}
.block-container {
    max-width: 1480px;
    padding-top: .65rem;
    padding-bottom: 1.2rem;
}
.app-eyebrow {
    color: #4f46e5;
    font-size: .71rem;
    letter-spacing: .13em;
    font-weight: 750;
    margin-bottom: .1rem;
}
.app-title {
    color: #192a48;
    font-size: 1.35rem;
    font-weight: 730;
    line-height: 1.25;
    margin: 0 0 .12rem 0;
}
.app-subtitle {
    color: #596b83;
    font-size: .93rem;
    margin-bottom: .35rem;
}
.section-eyebrow {
    margin: 1.2rem 0 .25rem;
    color: #415a79;
    font-size: .84rem;
    font-weight: 700;
    letter-spacing: .025em;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,.94);
    border: 1px solid #dce6f4;
    border-radius: 14px;
    padding: .48rem .78rem;
    box-shadow: 0 5px 22px rgba(39, 64, 112, .045);
}
[data-testid="stMetricLabel"] {
    color: #60738d;
}
[data-testid="stMetricValue"] {
    color: #293f67;
}
[data-testid="stTabs"] {
    background: rgba(255,255,255,.79);
    border: 1px solid #dde7f4;
    padding: .35rem .8rem .5rem;
    border-radius: 16px;
    box-shadow: 0 5px 20px rgba(39,64,112,.045);
}
[data-testid="stSidebar"] h3 {
    font-size: 1.05rem;
    color: #263a5b;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #657891;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    font-weight: 570;
}
@media (max-width: 650px) {
    .app-title { font-size: 1.25rem; }
    .block-container { padding-top: .65rem; }
}

/* Compact Greek cards: Greek symbol above the value, descriptive name below. */
.greek-card {
    min-height: 89px;
    background: rgba(255,255,255,.95);
    border: 1px solid #dce6f4;
    border-radius: 13px;
    padding: .36rem .38rem .38rem;
    text-align: center;
    box-shadow: 0 5px 20px rgba(39,64,112,.045);
}
.greek-symbol {
    color: #4f46e5;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.15rem;
    line-height: 1.05;
    font-weight: 700;
}
.greek-value {
    color: #21395d;
    font-size: clamp(.79rem, 1.15vw, 1.09rem);
    font-weight: 720;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
    margin-top: .16rem;
    line-height: 1.35;
}
.greek-name {
    color: #657891;
    font-size: .71rem;
    font-weight: 600;
    margin-top: .12rem;
}
.reference-title {
    color: #526782;
    font-size: .72rem;
    letter-spacing: .08em;
    font-weight: 740;
    margin-top: .28rem;
}
</style>
""", unsafe_allow_html=True)

def plot_tree(tree, name: str, color: str, show_values: bool = True, exercise_nodes=None):
    """Draw a recombining tree; highlight optimal early exercise if supplied."""
    levels = len(tree)
    edge_x, edge_y = [], []
    node_x, node_y, node_text, node_data, node_colors = [], [], [], [], []

    for k, row in enumerate(tree):
        for j, value in enumerate(row):
            x, y = k, k - 2 * j
            early_exercise = (
                exercise_nodes is not None
                and k < len(exercise_nodes)
                and exercise_nodes[k][j]
            )
            if early_exercise:
                status = "Early exercise"
            elif name == "Option value":
                status = "Terminal payoff" if k == levels - 1 else "Continuation"
            elif name == "Delta":
                status = "Hedge ratio"
            else:
                status = "Stock-price node"
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{value:.2f}")
            node_data.append([k, j, value, status])
            node_colors.append("#f97316" if early_exercise else color)

            if k < levels - 1:
                edge_x.extend([x, k + 1, None, x, k + 1, None])
                edge_y.extend([
                    y, (k + 1) - 2 * j, None,
                    y, (k + 1) - 2 * (j + 1), None,
                ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(109,132,168,0.48)", width=1.5),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text" if show_values else "markers",
        text=node_text if show_values else None,
        textposition="top center",
        textfont=dict(size=11, color="#374966"),
        marker=dict(size=15, color=node_colors, line=dict(color="white", width=1.5)),
        customdata=node_data,
        hovertemplate=(
            "<b>" + name + "</b><br>"
            "Time step: %{customdata[0]}<br>"
            "Position from top: %{customdata[1]}<br>"
            "Value: %{customdata[2]:.4f}<br>"
            "%{customdata[3]}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        height=min(720, max(345, 32 * levels)),
        margin=dict(l=16, r=16, t=12, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        font=dict(color="#425777"),
        xaxis=dict(
            title="Time step", tickmode="array", tickvals=list(range(levels)),
            range=[-0.4, levels - 0.6], showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            visible=False, range=[-levels - 0.5, levels + 0.5],
            fixedrange=False,
        ),
    )
    return fig


def plot_payoff(payoff, K, stock_values):
    """Plot the selected terminal payoff g(S, K), with terminal states marked."""
    terminal_stocks = stock_values[-1]
    low = max(0.001, min(terminal_stocks) * 0.75)
    high = max(terminal_stocks) * 1.25
    sample_stocks = [low + (high - low) * i / 200 for i in range(201)]
    sample_payoffs = []
    for stock in sample_stocks:
        try:
            sample_payoffs.append(payoff(stock, K))
        except ValueError:
            sample_payoffs.append(None)  # Undefined outside the tree's stock states

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_stocks, y=sample_payoffs,
        mode="lines", name="Payoff g(S, K)",
        line=dict(color="#4f46e5", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=terminal_stocks, y=[payoff(stock, K) for stock in terminal_stocks],
        mode="markers", name="Terminal nodes",
        marker=dict(color="#f97316", size=10),
    ))
    fig.update_layout(
        height=350, margin=dict(l=15, r=15, t=12, b=30),
        xaxis_title="Stock price at exercise (S)",
        yaxis_title="Exercise payoff g(S, K)",
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#425777"),
    )
    return fig




def plot_exercise_times(exercise_probabilities, dt, conditional=False):
    """Plot first-exercise probabilities at each discrete exercise time.

    Unconditional probabilities are measured under Q. The optional conditional
    chart normalizes early-exercise mass by P(early exercise), excluding maturity.
    """
    n = len(exercise_probabilities)
    steps = list(range(n - 1)) if conditional else list(range(n))
    total_early = sum(exercise_probabilities[:-1])
    values = [
        100 * exercise_probabilities[k] / total_early if conditional
        else 100 * exercise_probabilities[k]
        for k in steps
    ]
    labels = [f"{k * dt:g}" for k in steps]
    hover = [
        f"Step {k}<br>Time: {k * dt:.4g} years<br>Probability: {values[i]:.4f}%"
        for i, k in enumerate(steps)
    ]
    colors = ["#4f46e5" if k < n - 1 else "#0f766e" for k in steps]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{value:.1f}%" if value >= 0.5 else "" for value in values],
        textposition="outside",
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=15, b=38),
        xaxis_title="First optimal exercise time (years)",
        yaxis_title=("Conditional probability (%)" if conditional
                     else "Risk-neutral probability (%)"),
        yaxis=dict(range=[0, max(10, max(values, default=0) * 1.16)]),
        bargap=0.2,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#425777"),
        yaxis_gridcolor="#e4eaf5",
    )
    return fig


def plot_probability_flow(arrival_mass_tree, exercise_mass_tree):
    """Plot surviving probability mass arriving at each node before exercise."""
    levels = len(arrival_mass_tree)
    edge_x, edge_y = [], []
    node_x, node_y, colors, sizes, labels, hover = [], [], [], [], [], []
    for k, arrivals in enumerate(arrival_mass_tree):
        for j, mass in enumerate(arrivals):
            x, y = k, k - 2 * j
            exercised = exercise_mass_tree[k][j]
            if k < levels - 1 and mass > exercised + 1e-14:
                edge_x.extend([x, k + 1, None, x, k + 1, None])
                edge_y.extend([y, k + 1 - 2 * j, None,
                               y, k + 1 - 2 * (j + 1), None])
            node_x.append(x)
            node_y.append(y)
            if mass < 1e-14:
                colors.append("#cbd5e1")
            elif exercised > 1e-14:
                colors.append("#f97316" if k < levels - 1 else "#0f766e")
            elif k == levels - 1:
                colors.append("#94a3b8")  # reached maturity but expired worthless
            else:
                colors.append("#4f46e5")
            sizes.append(12 + 23 * math.sqrt(max(mass, 0)))
            labels.append(f"{mass:.1%}" if mass >= 0.005 else "")
            status = ("Early exercise" if exercised > 1e-14 and k < levels - 1
                      else "Exercise at maturity" if exercised > 1e-14
                      else "Continue" if mass > 1e-14 and k < levels - 1
                      else "Expire worthless" if mass > 1e-14
                      else "No surviving incoming paths")
            hover.append(
                f"Step {k}, node {j}<br>Surviving mass arriving: {mass:.5%}"
                f"<br>First-exercise mass: {exercised:.5%}<br>{status}"
            )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines", hoverinfo="skip",
        line=dict(color="rgba(130,140,155,0.42)", width=1.5),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=labels,
        textposition="top center", textfont=dict(size=10),
        marker=dict(size=sizes, color=colors, line=dict(color="white", width=1)),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=min(720, max(345, 32 * levels)),
        margin=dict(l=18, r=18, t=12, b=30),
        xaxis=dict(title="Time step", tickmode="array",
                   tickvals=list(range(levels)), showgrid=False, zeroline=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#425777"),
    )
    return fig


def plot_binomial_normal(n, p):
    """Compare exact Bin(n,p) probability masses to a normal density.

    The density is displayed per unit-count spacing, so heights are comparable
    to the discrete probability masses (integer spacing is one).
    """
    mean = n * p
    sd = math.sqrt(n * p * (1 - p))
    k_values = list(range(n + 1))
    ln_p, ln_q = math.log(p), math.log1p(-p)
    masses = [math.exp(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * ln_p + (n - k) * ln_q
    ) for k in k_values]

    # Restrict plotted range to where nearly all probability lies. For very
    # large n this keeps the approximate normal shape visually distinguishable.
    left = max(0, math.floor(mean - 5 * sd))
    right = min(n, math.ceil(mean + 5 * sd))
    shown_k = k_values[left:right + 1]
    shown_masses = masses[left:right + 1]
    fine_x = [left + (right - left) * i / 400 for i in range(401)]
    norm = [math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
            for x in fine_x]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shown_k, y=shown_masses, name="Binomial probability mass",
        marker=dict(color="#4f46e5", opacity=.76),
        hovertemplate="Up moves: %{x}<br>Probability: %{y:.6f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fine_x, y=norm, mode="lines", name="Normal approximation",
        line=dict(color="#0f766e", width=2.6),
        hovertemplate="Up moves: %{x:.1f}<br>Density: %{y:.6f}<extra></extra>",
    ))
    fig.update_layout(
        height=350, margin=dict(l=24, r=18, t=15, b=38),
        barmode="overlay", bargap=.13,
        xaxis=dict(title="Number of upward moves", showgrid=False, zeroline=False),
        yaxis=dict(title="Probability / normal density", gridcolor="#e1e9f2", zeroline=False),
        legend=dict(orientation="h", y=1.11, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#425777"),
    )
    return fig


def plot_hedging_values(records):
    """Compare the value of the long-stock/cash hedge with the short option liability."""
    times = [row["Time (years)"] for row in records]
    hedge = [row["Hedge value"] for row in records]
    liability = [row["Option liability"] for row in records]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=hedge, mode="lines+markers", name="Hedging portfolio",
        line=dict(color="#4f46e5", width=3), marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=times, y=liability, mode="lines+markers", name="Option liability",
        line=dict(color="#0f766e", width=2, dash="dash"),
        marker=dict(size=8, symbol="x"),
    ))
    fig.update_layout(
        height=315, margin=dict(l=18, r=12, t=26, b=28),
        title=dict(text="Replication along selected path", font=dict(size=14)),
        xaxis_title="Time (years)", yaxis_title="Value ($, one contract)",
        legend=dict(orientation="h", y=1.16, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#425777"), yaxis_gridcolor="#e4eaf5",
    )
    return fig


def plot_hedging_error(records):
    times = [row["Time (years)"] for row in records]
    errors = [row["Hedging error"] for row in records]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=errors, mode="lines+markers", name="Hedging error",
        line=dict(color="#ea580c", width=2.5), marker=dict(size=8),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    max_abs = max((abs(error) for error in errors), default=0.0)
    bound = max(1e-10, max_abs * 1.3)
    fig.update_layout(
        height=315, margin=dict(l=18, r=12, t=26, b=28),
        title=dict(text="Hedge minus option", font=dict(size=14)),
        xaxis_title="Time (years)", yaxis_title="Difference ($)",
        yaxis=dict(range=[-bound, bound], gridcolor="#e4eaf5"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#425777"),
    )
    return fig


# Sidebar inputs
st.sidebar.caption("APP VERSION: FORMULAS V4")
st.sidebar.header("Model settings")
option_style = st.sidebar.segmented_control(
    "Exercise style", ["European", "American"],
    default="European", selection_mode="single",
)
payoff_type = st.sidebar.segmented_control(
    "Option payoff", ["Call", "Put", "Custom"],
    default="Call", selection_mode="single",
)
payoff_expression = None
if payoff_type == "Custom":
    payoff_expression = st.sidebar.text_input(
        "Payoff expression g(S, K)",
        value="max(S - K, 0)",
        help="Use S for stock price, K for strike. Supported: +, -, *, /, **, "
             "comparisons, conditional expressions, max, min, abs, sqrt, "
             "exp, log, sin, cos, and constants pi and e.",
    )
    st.sidebar.caption("Examples: `abs(S - K)` or `1 if S > K else 0`.")
    st.sidebar.caption("Custom payoffs depend only on current S and K; path-dependent payoffs aren't supported.")

st.sidebar.divider()
T = st.sidebar.slider("Number of tree levels", min_value=2, max_value=20, value=3)
S = st.sidebar.number_input("Initial stock price (S)", min_value=0.01, value=20.0, step=1.0)
K = st.sidebar.number_input("Strike price (K)", min_value=0.0, value=21.0, step=1.0)
sigma_percent = st.sidebar.number_input(
    "Annualized volatility (sigma, %)",
    min_value=0.01, value=20.0, step=1.0, format="%.2f",
)
r_percent = st.sidebar.number_input("Risk-free rate (% per year)", value=12.0, step=0.5)
dt = st.sidebar.number_input(
    "Time between steps (years)", min_value=0.001, value=0.25,
    step=0.01, format="%.3f",
)
show_values = st.sidebar.checkbox("Show values beside nodes", value=True)

sigma = sigma_percent / 100.0
r = r_percent / 100.0
# The up/down factors are derived rather than entered independently.
u = math.exp(sigma * math.sqrt(dt))
d = math.exp(-sigma * math.sqrt(dt))
st.sidebar.divider()
st.sidebar.write(f"**Up factor (u):** {u:.4f}")
st.sidebar.write(f"**Down factor (d):** {d:.4f}  (d = 1/u)")
st.sidebar.write(f"**Maturity:** {(T - 1) * dt:.3f} years")
st.sidebar.caption("CRR factors: u = exp(sigma * sqrt(dt)), d = exp(-sigma * sqrt(dt)).")
st.sidebar.caption("T counts tree levels; there are T − 1 price transitions.")

st.markdown(
    '<div class="app-eyebrow">INTERACTIVE PRICING MODEL</div>'
    '<div class="app-title">Binomial Pricing Model</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="app-subtitle">{option_style} {payoff_type.lower()} &nbsp;·&nbsp; '
    'Cox–Ross–Rubinstein tree &nbsp;·&nbsp; Risk-neutral valuation</div>',
    unsafe_allow_html=True,
)

try:
    stock_values, option_values, delta_values, exercise_nodes = binomial_pricing(
        T, S, K, sigma, r, dt,
        option_style=option_style,
        payoff_type=payoff_type,
        payoff_expression=payoff_expression,
        return_exercise=True,
    )
    payoff = make_payoff(payoff_type, payoff_expression)
except (ValueError, OverflowError) as exc:
    st.sidebar.error("Please check the parameters or payoff expression")
    st.error(str(exc))
    st.stop()

p = (math.exp(r * dt) - d) / (u - d)
q = 1 - p
st.sidebar.success("No-arbitrage condition satisfied")
st.sidebar.caption(f"Risk-neutral probabilities: up = {p:.4f}, down = {q:.4f}")

# Root Greeks are computed from the currently selected payoff/exercise style.
greeks = calculate_greeks(
    T, S, K, sigma, r, dt,
    option_style=option_style,
    payoff_type=payoff_type,
    payoff_expression=payoff_expression,
)


def greek_fmt(value, digits=4):
    return "—" if value is None else f"{value:,.{digits}f}"


# Keep the root Greeks on-screen, with the symbols over the values.
st.markdown("##### Option price & Greeks")

def greek_card(symbol, value, name, definition):
    """Compact presentation card; the tooltip preserves the exact definition."""
    from html import escape
    st.markdown(
        f'<div class="greek-card" title="{escape(definition, quote=True)}">'
        f'<div class="greek-symbol">{escape(symbol)}</div>'
        f'<div class="greek-value">{escape(value)}</div>'
        f'<div class="greek-name">{escape(name)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

price_col, delta_col, gamma_col, vega_col, theta_col, rho_col = st.columns(6, gap="small")
with price_col:
    greek_card("V₀", f"${option_values[0][0]:.4f}", "Option price", "Price at the root node")
with delta_col:
    greek_card("Δ", greek_fmt(greeks["delta_hedge"]), "Delta", "∂V/∂S")
with gamma_col:
    greek_card("Γ", greek_fmt(greeks["gamma"]), "Gamma", "∂²V/∂S²")
with vega_col:
    greek_card("𝒱", greek_fmt(greeks["vega_per_vol_point"]), "Vega", "∂V/∂σ, per volatility percentage point")
with theta_col:
    greek_card("Θ", greek_fmt(greeks["theta_per_day"], 5), "Theta", "−∂V/∂τ, per calendar day")
with rho_col:
    greek_card("ρ", greek_fmt(greeks["rho_per_bp"], 5), "Rho", "∂V/∂r, per basis point")
st.caption(
    "Δ and Γ use tree-based estimates · Vega is quoted per +1 volatility percentage point "
    "· Θ per calendar day · ρ per +1 basis point (0.01%) of interest rate."
)

st.caption("Select a visualization. Hover over tree nodes to inspect values; use the sidebar to change inputs.")
if option_style == "American":
    option_tab, greeks_tab, stock_tab, delta_tab, payoff_tab, binomial_tab, hedge_tab, exercise_tab = st.tabs(
        ["Option tree", "Greeks · details", "Stock tree", "Delta hedge", "Payoff", "Binomial (n,p)", "Replication", "Exercise times"]
    )
else:
    option_tab, greeks_tab, stock_tab, delta_tab, payoff_tab, binomial_tab, hedge_tab = st.tabs(
        ["Option tree", "Greeks · details", "Stock tree", "Delta hedge", "Payoff", "Binomial (n,p)", "Replication"]
    )

with option_tab:
    st.plotly_chart(
        plot_tree(
            option_values, "Option value", "#4f46e5", show_values,
            exercise_nodes=exercise_nodes if option_style == "American" else None,
        ),
        width="stretch", key="option_tree",
    )
    if option_style == "American":
        st.caption("Orange nodes = optimal early exercise; blue nodes = continuation or terminal payoff.")
    else:
        st.caption("For European exercise, the payoff is used only at maturity; earlier nodes use backward induction.")

    # Formula references stay directly under the option tree; these do not
    # alter valuation and are also useful when the user selects a custom payoff.
    st.markdown('<div class="reference-title">MODEL EQUATIONS · REFERENCE</div>', unsafe_allow_html=True)
    payoff_col, factors_col = st.columns([1, 1], gap="medium")
    with payoff_col:
        st.markdown(
            r"**Call & put payoffs**  "
            "\n" r"$C_T = \max(S_T-K,0)$  "
            "\n" r"$P_T = \max(K-S_T,0)$"
        )
    with factors_col:
        st.markdown(
            r"**CRR up/down factors**  "
            "\n" r"$u=e^{\sigma\sqrt{\Delta t}}$  "
            "\n" r"$d=e^{-\sigma\sqrt{\Delta t}}=1/u$"
        )
    st.markdown(
        r"**Black–Scholes PDE for $V(t,S)$** (non-dividend-paying stock)  "
        "\n" r"$\displaystyle \frac{\partial V}{\partial t} + "
        r"\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2} "
        r"+ rS\frac{\partial V}{\partial S} - rV = 0$"
    )
    st.caption(
        "The PDE applies to a smooth European option value, and to an American option "
        "in its continuation region. The displayed call/put payoffs are reference formulas "
        "when Custom payoff is selected."
    )

with stock_tab:
    st.plotly_chart(
        plot_tree(stock_values, "Stock price", "#0f766e", show_values),
        width="stretch", key="stock_tree",
    )

with delta_tab:
    st.plotly_chart(
        plot_tree(delta_values, "Delta", "#7c3aed", show_values),
        width="stretch", key="delta_tree",
    )
    st.caption("Delta is defined at nonterminal nodes, so this tree has one fewer level.")

with greeks_tab:
    st.markdown("#### Greeks at the initial node")
    st.markdown(r"""
| Greek | Mathematical definition |
|:--|:--|
| Delta (Δ) | $\displaystyle \frac{\partial V}{\partial S}$ |
| Gamma (Γ) | $\displaystyle \frac{\partial^2 V}{\partial S^2}$ |
| Vega (𝒱) | $\displaystyle \frac{\partial V}{\partial \sigma}$ |
| Theta (Θ) | $\displaystyle -\frac{\partial V}{\partial \tau}$ |
| Rho (ρ) | $\displaystyle \frac{\partial V}{\partial r}$ |

Here $\tau$ is **time remaining to maturity**. The displayed numerical values
use the per-unit conventions described above.
""")
    st.info("Delta and Gamma come from tree hedges. Vega, Rho, and Theta are calculated by bumping one input and repricing the entire tree.")
    st.caption(
        "All values are per share of the option, not per 100-share contract. "
        "The hedge ratio and the spot-price finite-difference delta can differ "
        "in a finite binomial tree."
    )
    st.dataframe([
        {"Greek": "Delta (tree hedge)", "Value": greek_fmt(greeks["delta_hedge"], 6),
         "Interpretation": "Shares needed per option share to replicate the next step"},
        {"Greek": "Delta (spot bump)", "Value": greek_fmt(greeks["delta_spot"], 6),
         "Interpretation": "Option-price change per $1 stock-price change"},
        {"Greek": "Gamma (tree)", "Value": greek_fmt(greeks["gamma"], 6),
         "Interpretation": "Change in hedge delta per $1 stock-price change"},
        {"Greek": "Vega", "Value": greek_fmt(greeks["vega_per_vol_point"], 6),
         "Interpretation": "Option-price change per 1 percentage point of volatility"},
        {"Greek": "Rho", "Value": greek_fmt(greeks["rho_per_bp"], 6),
         "Interpretation": "Option-price change per 1 basis point of annual rate"},
        {"Greek": "Theta", "Value": greek_fmt(greeks["theta_per_day"], 6),
         "Interpretation": "Approximate option-price change for one calendar day passing"},
    ], width="stretch", hide_index=True)
    if T < 3:
        st.info("Tree Gamma needs at least three tree levels (two transitions). Increase T to calculate it.")
    st.caption(
        "Vega, Rho, and Theta use central bump-and-reprice differences, "
        "rebuilding the entire CRR tree after each change. "
        "Theta changes remaining maturity while holding the number of steps fixed; "
        "these are numerical approximations, especially near American early-exercise "
        "boundaries and for discontinuous custom payoffs."
    )
    with st.expander("Bump sizes and numerical details"):
        bumps = greeks["bumps"]
        st.write(
            f"Spot: ±${bumps['spot_dollars']:.6g} · "
            f"Volatility: ±{100*bumps['sigma_decimal']:.5g} percentage points · "
            f"Interest rate: ±{10_000*bumps['rate_decimal']:.5g} basis points · "
            f"Maturity: ±{365*bumps['maturity_years']:.5g} days"
        )
        st.caption(
            "With a fixed CRR tree, option value is piecewise linear in S, so a tiny "
            "second spot difference often gives a misleading Gamma of zero. "
            "Tree Gamma is estimated from deltas at the two first-step nodes. "
            "At an immediate American exercise node, use caution interpreting "
            "the continuation hedge delta as a price sensitivity."
        )

with payoff_tab:
    st.plotly_chart(plot_payoff(payoff, K, stock_values), width="stretch", key="payoff_curve")
    st.caption("Orange markers show payoffs at maturity's stock prices. For American options, "
               "the same function also determines exercise value at earlier nodes.")


with binomial_tab:
    st.caption(
        "Compare exact risk-neutral binomial probabilities with a normal approximation. "
        "Maturity and sigma stay fixed as the number of steps changes."
    )
    n_binom = st.slider(
        "Number of binomial steps n (distribution experiment)",
        min_value=2, max_value=400, value=max(40, T - 1), step=1,
        key="n_binomial_experiment",
    )
    maturity = (T - 1) * dt
    dt_binom = maturity / n_binom
    u_binom = math.exp(sigma * math.sqrt(dt_binom))
    d_binom = 1 / u_binom
    growth_binom = math.exp(r * dt_binom)
    if not (d_binom < growth_binom < u_binom):
        st.warning("No-arbitrage fails for these step parameters. Increase n or change the model inputs.")
    else:
        p_binom = (growth_binom - d_binom) / (u_binom - d_binom)
        st.plotly_chart(
            plot_binomial_normal(n_binom, p_binom), width="stretch", key="binomial_normal_chart"
        )
        st.caption(
            f"n = {n_binom}, p = {p_binom:.5f}, "
            f"mean = {n_binom*p_binom:.2f}, "
            f"standard deviation = {math.sqrt(n_binom*p_binom*(1-p_binom)):.2f}. "
            "The normal curve approximates binomial probabilities at integer counts. "
            "With fixed maturity, the CRR log-stock price approaches a normal distribution."
        )


with hedge_tab:
    st.caption(
        "Track a self-financing delta hedge of one short option contract (100 shares). "
        "Choose a stock-price path; the hedge is rebalanced at each continuation node."
    )
    n_steps = T - 1
    choice = st.radio(
        "Path through the tree", ["Alternating", "All up", "All down", "Custom"],
        horizontal=True, key="hedge_path_choice",
    )
    if choice == "Alternating":
        path = ("UD" * n_steps)[:n_steps]
    elif choice == "All up":
        path = "U" * n_steps
    elif choice == "All down":
        path = "D" * n_steps
    else:
        path = st.text_input(
            "Up/down sequence (U or D for each step)",
            value=("UD" * n_steps)[:n_steps],
            help=f"Enter exactly {n_steps} characters. U = upward stock move; D = downward stock move.",
        )
    st.caption(f"Selected path: `{path}`  ·  Each step is {dt:g} years")
    try:
        ledger = simulate_delta_hedge(
            stock_values, option_values, delta_values, path, r, dt,
            option_style=option_style, exercise_nodes=exercise_nodes,
            multiplier=100,
        )
    except ValueError as exc:
        st.warning(str(exc))
    else:
        max_error = max(abs(row["Hedging error"]) for row in ledger)
        a, b, c = st.columns(3)
        a.metric("Initial option premium (×100)", f"${option_values[0][0] * 100:,.2f}")
        b.metric("Final hedge value", f"${ledger[-1]['Hedge value']:,.2f}")
        c.metric("Max absolute hedging error", f"${max_error:.2e}")
        lhs, rhs = st.columns([1.8, 1])
        with lhs:
            st.plotly_chart(plot_hedging_values(ledger), width="stretch", key="hedge_value_plot")
        with rhs:
            st.plotly_chart(plot_hedging_error(ledger), width="stretch", key="hedge_error_plot")
        st.markdown("#### Trading and cash-account ledger")
        st.dataframe(
            ledger, width="stretch", hide_index=True,
            column_order=["Step", "Time (years)", "Move", "Stock price",
                          "Shares held", "Shares traded", "Cash balance",
                          "Hedge value", "Option liability", "Hedging error", "Action"],
        )
        st.caption(
            "A positive share trade is a purchase; a negative trade is a sale. "
            "The cash balance accrues interest between steps and finances share rebalancing. "
            "Values are per 100-share contract. In a frictionless CRR tree, "
            "the hedge matches the modeled option value along every path until settlement."
        )
        if option_style == "American":
            st.caption(
                "American exercise follows the tree's optimal exercise rule; "
                "the hedge stops at the first exercise node. At maturity it settles regardless."
            )

if option_style == "American":
    with exercise_tab:
        distribution = exercise_time_distribution(
            stock_values, exercise_nodes, p, K,
            payoff_type=payoff_type, payoff_expression=payoff_expression,
        )
        st.markdown("#### Distribution of first optimal exercise time")
        st.caption(
            "Exact risk-neutral probabilities computed by forward probability propagation. "
            "At an exercise node, probability mass exits the tree and never exercises again."
        )
        first, second, third = st.columns(3)
        first.metric("Early exercise", f"{distribution['early_exercise_probability']:.2%}")
        second.metric("Exercise at maturity", f"{distribution['maturity_exercise_probability']:.2%}")
        third.metric("Expire without exercise", f"{distribution['expired_unexercised_probability']:.2%}")

        conditional = st.checkbox(
            "Condition on early exercise (exclude maturity)", value=False,
            help="If selected, show P(first exercise at t | exercise strictly before maturity)."
        )
        if conditional and distribution["early_exercise_probability"] <= 1e-14:
            st.info("No paths exercise early for these model parameters, so the conditional distribution is undefined.")
        else:
            st.plotly_chart(
                plot_exercise_times(distribution["exercise_probabilities"], dt, conditional),
                width="stretch", key="exercise_time_distribution",
            )

        st.caption(
            "Blue bars: early exercise; green bar: exercise at maturity (unconditional view). "
            "Expiration without exercise is a separate outcome, not an exercise time. "
            "For the unconditional view, all exercise-time probabilities plus expiration sum to 100%."
        )
        st.markdown("#### Forward probability flow")
        st.caption(
            "Each node label is the probability of arriving there *without prior exercise*. "
            "Orange nodes absorb probability through early exercise; green nodes exercise at maturity; "
            "gray nodes either expire worthless at maturity or have no surviving incoming paths. "
            "Hover for exact probabilities."
        )
        st.plotly_chart(
            plot_probability_flow(
                distribution["arrival_mass_tree"], distribution["exercise_mass_tree"]
            ),
            width="stretch", key="surviving_probability_tree",
        )
        if payoff_type == "Call" and r >= 0:
            st.info(
                "For a standard American call on a non-dividend-paying stock "
                "with a nonnegative interest rate, early exercise is not optimal."
            )
        elif payoff_type == "Custom":
            st.caption(
                "The exercise/no-exercise interpretation assumes a nonnegative custom payoff. "
                "Terminal payoffs at or below zero are treated as expiring unexercised."
            )
