% portfolio.mzn  --  weekly decision model (mixed-integer LINEAR program)
%
% Given the regression forecasts, decide at the start of the week:
%   Gold   : integer number of lots to buy / sell (1 lot = 1 % of the gold price)
%   Spot   : $ to buy / sell of BTC and ETH
%   Margin : open at most one new position per coin (direction + leverage 1..3),
%            and keep or close each already open position
%   Bonds  : $ to lock for one month
% Objective: maximise the expected wealth at the end of the week.
% Risk     : the stressed loss (stop-loss for margin, z*sigma for the rest) must stay below rho * wealth.

% ----- sizes -----
int: NC;                                   % coins: 1 = BTC, 2 = ETH
int: NP;                                   % already open margin positions
int: max_lev;                              % 3
set of int: COIN = 1..NC;
set of int: POS  = 1..NP;
set of int: DIR  = 1..2;                   % 1 = long, 2 = short
set of int: LEV  = 1..max_lev;

% ----- account -----
float: cash;
float: wealth;
float: z;
float: rho;
float: max_bond_frac;

% ----- gold -----
float: gold_price;
int:   gold_lot_pct;
int:   gold_held;                          % lots held
float: gold_ret;                           % forecast weekly return
float: gold_sigma;

% ----- crypto -----
array[COIN] of float: spot_value;          % $ held in spot
array[COIN] of float: c_ret;               % forecast weekly return
array[COIN] of float: c_sigma;
array[COIN, DIR] of float: c_pret;         % expected return per $ notional of a new position (TP/SL applied)
array[COIN] of float: c_sl;                % stop-loss distance (fraction)

% ----- open margin positions -----
array[POS] of int:   p_coin;
array[POS] of int:   p_forced;             % 1 = one month reached, must close now
array[POS] of float: p_notional;
array[POS] of float: p_equity;
array[POS] of float: p_pret;

% ----- bonds -----
float: bond_locked;
float: bond_rate_week;

% ----- derived constants -----
float: lot_value    = gold_price * gold_lot_pct / 100.0;
int:   max_gold_buy = min(100000, floor(cash / lot_value));
float: BIG          = wealth * (max_lev + 2) + 1.0;

% ----- decision variables -----
var 0..max_gold_buy: gold_buy;
var 0..gold_held:    gold_sell;

array[COIN] of var 0.0..BIG: spot_buy;
array[COIN] of var 0.0..BIG: spot_sell;
constraint forall(c in COIN)(spot_sell[c] <= spot_value[c]);

array[COIN, DIR, LEV] of var 0.0..cash: m;     % collateral put into a new margin position
array[COIN, DIR, LEV] of var 0..1: y;          % 1 = this (direction, leverage) is opened
constraint forall(c in COIN, d in DIR, l in LEV)(m[c,d,l] <= cash * y[c,d,l]);
constraint forall(c in COIN)(sum(d in DIR, l in LEV)(y[c,d,l]) <= 1);

array[POS] of var 0..1: keep;                  % 1 = keep the open position another week
constraint forall(i in POS)(keep[i] <= 1 - p_forced[i]);

var 0.0..BIG: bond_buy;
constraint bond_buy <= max(0.0, max_bond_frac * wealth - bond_locked);

% ----- cash balance -----
var float: cash_after =
      cash
    - lot_value * gold_buy + lot_value * gold_sell
    - sum(c in COIN)(spot_buy[c]) + sum(c in COIN)(spot_sell[c])
    - sum(c in COIN, d in DIR, l in LEV)(m[c,d,l])
    + sum(i in POS)((1 - keep[i]) * p_equity[i])
    - bond_buy;
constraint cash_after >= 0.0;

% ----- risk budget -----
var float: gold_after = (gold_held + gold_buy - gold_sell) * lot_value;
array[COIN] of var float: spot_after = [ spot_value[c] + spot_buy[c] - spot_sell[c] | c in COIN ];

var float: stressed_loss =
      gold_after * z * gold_sigma
    + sum(c in COIN)(spot_after[c] * z * c_sigma[c])
    + sum(c in COIN, d in DIR, l in LEV)(l * m[c,d,l] * c_sl[c])
    + sum(i in POS)(keep[i] * p_notional[i] * c_sl[p_coin[i]]);
constraint stressed_loss <= rho * wealth;

% ----- objective: expected wealth at the end of the week -----
var float: exp_wealth =
      cash_after
    + gold_after * (1.0 + gold_ret)
    + sum(c in COIN)(spot_after[c] * (1.0 + c_ret[c]))
    + sum(c in COIN, d in DIR, l in LEV)(m[c,d,l] * (1.0 + l * c_pret[c,d]))
    + sum(i in POS)(keep[i] * (p_equity[i] + p_notional[i] * p_pret[i]))
    + (bond_locked + bond_buy) * (1.0 + bond_rate_week);

solve maximize exp_wealth;
