import os, sys, warnings
import pandas as pd
import numpy as np
import vectorbt as vbt
warnings.filterwarnings("ignore")

df_master = pd.read_csv('Oil_futures_regime_strat/df_master_cached.csv', index_col=0, parse_dates=True)
years = sorted(df_master.index.year.unique())

T_train_grid = [0.10, 0.15, 0.20, 0.25]
W_FILT_grid = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
TP_ATR_grid = [1.5, 2.0, 3.0]
LEVERAGE = 3.0

wf_results = []
oos_returns_list = []

for i in range(4, len(years)):
    test_year = years[i]
    test_start = f"{test_year}-01-01"
    test_end = f"{test_year}-12-31"
    train_end = f"{test_year-1}-12-31"
    
    df_train = df_master.loc[:train_end].copy()
    df_test = df_master.loc[test_start:test_end].copy()
    
    if len(df_test) < 10: continue
    
    best_sharpe = -999
    best_params = {'T': 0.15, 'W_FILT': 0.7, 'TP_ATR': 2.0}
    
    for t_val in T_train_grid:
        for w_filt_val in W_FILT_grid:
            for tp_atr in TP_ATR_grid:
                # 1. Filtered Momentum
                en_1 = (df_train['cdi'] > t_val) & (df_train['brent_close'] > df_train['sma_50']) & (df_train['rsi'] < 70)
                sen_1 = (df_train['cdi'] < -t_val) & (df_train['brent_close'] < df_train['sma_50']) & (df_train['rsi'] > 30)
                
                clean_en_1, clean_sen_1, ex_1, sex_1 = pd.Series(False, index=df_train.index), pd.Series(False, index=df_train.index), pd.Series(False, index=df_train.index), pd.Series(False, index=df_train.index)
                in_trade, in_short = False, False
                for j in range(len(en_1)):
                    current_price = df_train['brent_close'].iloc[j]
                    if en_1.iloc[j] and not in_trade and not in_short:
                        in_trade = True; clean_en_1.iloc[j] = True
                        tp_target = current_price + (tp_atr * df_train['atr'].iloc[j])
                    elif in_trade:
                        if df_train['cdi'].iloc[j] <= 0 or current_price >= tp_target:
                            ex_1.iloc[j] = True; in_trade = False
                            continue
                    if sen_1.iloc[j] and not in_short and not in_trade:
                        in_short = True; clean_sen_1.iloc[j] = True
                        tp_target = current_price - (tp_atr * df_train['atr'].iloc[j])
                    elif in_short:
                        if df_train['cdi'].iloc[j] >= 0 or current_price <= tp_target:
                            sex_1.iloc[j] = True; in_short = False
                            continue
                            
                sz_1 = pd.Series(LEVERAGE, index=df_train.index)
                pf_filt_train = vbt.Portfolio.from_signals(
                    close=df_train['brent_close'], entries=clean_en_1, exits=ex_1,
                    short_entries=clean_sen_1, short_exits=sex_1, sl_trail=0.03,
                    size=sz_1, size_type='percent', fees=0.0002, freq='D', direction='both',
                    upon_opposite_entry='ignore'
                )
                
                # 2. Continuous Trend
                raw_en = (df_train['cdi'] > t_val) & (df_train['brent_close'] > df_train['sma_50'])
                raw_sen = (df_train['cdi'] < -t_val) & (df_train['brent_close'] < df_train['sma_50'])
                safe_en = raw_en & ~raw_sen
                safe_sen = raw_sen & ~raw_en
                
                sz_2 = pd.Series(LEVERAGE, index=df_train.index)
                pf_cont_train = vbt.Portfolio.from_signals(
                    close=df_train['brent_close'], entries=safe_en, short_entries=safe_sen,
                    sl_trail=0.03, tp_stop=0.06, size=sz_2, size_type='percent',
                    upon_opposite_entry='ignore', fees=0.0002, freq='D', direction='both'
                )
                
                w_filt = pd.Series(w_filt_val, index=df_train.index)
                w_cont = 1.0 - w_filt
                
                comb_train_returns = (pf_filt_train.returns() * w_filt) + (pf_cont_train.returns() * w_cont)
                vbt_train = comb_train_returns.vbt.returns(freq='D')
                
                try: sr = vbt_train.sharpe_ratio()
                except: sr = 0
                if pd.isna(sr) or np.isinf(sr): sr = 0
                
                if sr > best_sharpe:
                    best_sharpe = sr
                    best_params = {'T': t_val, 'W_FILT': w_filt_val, 'TP_ATR': tp_atr}

    print(f"[{test_year}] Best Train Params: {best_params} (Train Sharpe: {best_sharpe:.2f})")
    
    # --- EXECUTE ON TEST DATA ---
    t_val = best_params['T']
    w_filt_val = best_params['W_FILT']
    tp_atr = best_params['TP_ATR']
    
    en_1 = (df_test['cdi'] > t_val) & (df_test['brent_close'] > df_test['sma_50']) & (df_test['rsi'] < 70)
    sen_1 = (df_test['cdi'] < -t_val) & (df_test['brent_close'] < df_test['sma_50']) & (df_test['rsi'] > 30)
    
    clean_en_1, clean_sen_1, ex_1, sex_1 = pd.Series(False, index=df_test.index), pd.Series(False, index=df_test.index), pd.Series(False, index=df_test.index), pd.Series(False, index=df_test.index)
    in_trade, in_short = False, False
    for j in range(len(en_1)):
        current_price = df_test['brent_close'].iloc[j]
        if en_1.iloc[j] and not in_trade and not in_short:
            in_trade = True; clean_en_1.iloc[j] = True
            tp_target = current_price + (tp_atr * df_test['atr'].iloc[j])
        elif in_trade:
            if df_test['cdi'].iloc[j] <= 0 or current_price >= tp_target:
                ex_1.iloc[j] = True; in_trade = False
                continue
        if sen_1.iloc[j] and not in_short and not in_trade:
            in_short = True; clean_sen_1.iloc[j] = True
            tp_target = current_price - (tp_atr * df_test['atr'].iloc[j])
        elif in_short:
            if df_test['cdi'].iloc[j] >= 0 or current_price <= tp_target:
                sex_1.iloc[j] = True; in_short = False
                continue
                
    sz_1 = pd.Series(LEVERAGE, index=df_test.index)
    pf_filt_test = vbt.Portfolio.from_signals(
        close=df_test['brent_close'], entries=clean_en_1, exits=ex_1,
        short_entries=clean_sen_1, short_exits=sex_1, sl_trail=0.03,
        size=sz_1, size_type='percent', fees=0.0002, freq='D', direction='both',
        upon_opposite_entry='ignore'
    )
    
    raw_en = (df_test['cdi'] > t_val) & (df_test['brent_close'] > df_test['sma_50'])
    raw_sen = (df_test['cdi'] < -t_val) & (df_test['brent_close'] < df_test['sma_50'])
    safe_en = raw_en & ~raw_sen
    safe_sen = raw_sen & ~raw_en
    
    sz_2 = pd.Series(LEVERAGE, index=df_test.index)
    pf_cont_test = vbt.Portfolio.from_signals(
        close=df_test['brent_close'], entries=safe_en, short_entries=safe_sen,
        sl_trail=0.03, tp_stop=0.06, size=sz_2, size_type='percent',
        upon_opposite_entry='ignore', fees=0.0002, freq='D', direction='both'
    )
    
    w_filt = pd.Series(w_filt_val, index=df_test.index)
    w_cont = 1.0 - w_filt
    
    comb_test_returns = (pf_filt_test.returns() * w_filt) + (pf_cont_test.returns() * w_cont)
    oos_returns_list.append(comb_test_returns)

unified_oos_returns = pd.concat(oos_returns_list)
vbt_oos_unified = unified_oos_returns.vbt.returns(freq='D')
stats = vbt_oos_unified.stats()
print("\n=== FINAL UNIFIED TRUE WFO STATS ===")
print(stats)
