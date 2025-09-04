import os
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["OMP_NUM_THREADS"] = "10"
import polars as pl
import polars.selectors as cs
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from scipy import stats
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from qutils import files
from qdata.core.dt import get_td, shift_n_td
import datetime as dt
warnings.filterwarnings('ignore')

import sys
import logging
from datetime import datetime

expected_columns = [
    'window_type', 'n_snapshots', 'mid_price_mean', 'mid_price_std', 'mid_price_skew', 'mid_price_kurt', 'price_return_mean', 'price_return_std', 'price_return_sum', 'price_range', 'price_range_pct', 'spread_mean', 'spread_std', 'relative_spread_mean', 'total_volume', 'avg_trade_size', 'volume_std', 'n_trades', 'avg_bid_depth', 'avg_ask_depth', 'depth_imbalance', 'bid1_vol_mean', 'ask1_vol_mean', 'level1_vol_imbalance', 'bid2_vol_mean', 'ask2_vol_mean', 'level2_vol_imbalance', 'bid2_price_std', 'ask2_price_std', 'bid3_vol_mean', 'ask3_vol_mean', 'level3_vol_imbalance', 'bid3_price_std', 'ask3_price_std', 'effective_spread_mean', 'price_impact_mean', 'price_impact_std', 'order_flow_imbalance_mean', 'order_flow_imbalance_std', 'order_flow_persistence', 'aggressive_buy_ratio', 'volatility_clustering', 'realized_volatility', 'order_arrival_rate', 'order_arrival_std', 'activity_intensity', 'bid_slope_mean', 'bid_slope_std', 'ask_slope_mean', 'ask_slope_std', 'orderbook_symmetry', 'large_order_bid_ratio', 'large_order_frequency', 'iceberg_signal_count', 'price_discovery_lead', 'max_lead_correlation', 'informed_trading_proxy', 'price_trend_slope', 'price_trend_r2', 'bid1_price_std', 'ask1_price_std'
]
# 设置日志格式
def setup_logging():
    """设置日志记录，同时输出到控制台"""
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    return logger



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-sdate',help='starting date',default=str(dt.datetime.now().date()))
parser.add_argument('-edate',help='ending date',default=str(dt.datetime.now().date()))
parser.add_argument('-ob_orig_path',help='orig l2 path', default='/data/sihang/l2_ob_full_universe_with_info/')
parser.add_argument('-save_path',help='sample save path',default='/data/sihang/l2_ob_agg_5min/')

class OptimizedPolarsOrderBookProcessor:
    """
    优化版本的Polars DataFrame格式的L2订单簿数据处理器
    保留所有原始特征，同时优化性能
    """
    
    def __init__(self, n_levels: int = 10, n_workers: int = None):
        self.n_levels = n_levels
        self.n_workers = n_workers or mp.cpu_count()
        
        # A股交易时间定义
        self.auction_times = [
            (time(9, 15), time(9, 25, 5)),    # 早盘集合竞价
            (time(14, 57), time(15, 0, 5))    # 尾盘集合竞价
        ]
        
        self.continuous_times = [
            (time(9, 30), time(11, 30)),      # 上午连续竞价
            (time(13, 0), time(14, 57))       # 下午连续竞价
        ]
    
    def process_all_symbols(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        并行处理所有股票的数据，生成5分钟特征
        """
        # 确保datetime列是时间类型
        if df.schema['datetime'] == pl.Utf8:
            df = df.with_columns([
                pl.col('datetime').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            ])
        
        # 预处理：添加时间相关列
        df = self._preprocess_time_columns(df)
        
        # 按股票分组
        symbols = df['symbol'].unique().to_list()
        
        # 使用多进程并行处理
        if len(symbols) > 10 and self.n_workers > 1:
            batch_size = max(1, len(symbols) // (self.n_workers * 2))
            symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for batch in symbol_batches:
                    batch_df = df.filter(pl.col('symbol').is_in(batch))
                    future = executor.submit(self._process_symbol_batch, batch_df)
                    futures.append(future)
                
                all_features = []
                for future in futures:
                    result = future.result()
                    if result is not None and len(result) > 0:
                        all_features.append(result)
            
            if all_features:
                return pl.concat(all_features)
            else:
                return pl.DataFrame()
        else:
            return self._process_symbol_batch(df)
    
    def _preprocess_time_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """预处理时间相关列"""
        return df.with_columns([
            pl.col('datetime').dt.time().alias('time_only'),
            pl.col('datetime').dt.date().alias('date'),
            pl.when(
                (pl.col('datetime').dt.time() >= time(9, 15)) & 
                (pl.col('datetime').dt.time() <= time(9, 25, 5))
            ).then(pl.lit("opening_auction"))
            .when(
                (pl.col('datetime').dt.time() >= time(14, 57)) & 
                (pl.col('datetime').dt.time() <= time(15, 0, 5))
            ).then(pl.lit("closing_auction"))
            .when(
                ((pl.col('datetime').dt.time() >= time(9, 30)) & 
                 (pl.col('datetime').dt.time() < time(11, 30))) |
                ((pl.col('datetime').dt.time() >= time(13, 0)) & 
                 (pl.col('datetime').dt.time() < time(14, 57)))
            ).then(pl.lit("continuous"))
            .otherwise(pl.lit("other"))
            .alias('trading_phase')
        ])
    
    def _process_symbol_batch(self, df: pl.DataFrame) -> pl.DataFrame:
        """处理一批股票"""
        all_features = []
        
        for symbol in df['symbol'].unique().to_list():
            symbol_df = df.filter(pl.col('symbol') == symbol)
            symbol_features = self._process_single_symbol(symbol_df, symbol)
            if symbol_features is not None and len(symbol_features) > 0:
                all_features.append(symbol_features)
        
        if all_features:
            return pl.concat(all_features)
        return pl.DataFrame()
    
    def _process_single_symbol(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """处理单个股票的数据"""
        features_list = []
        
        for date in df['date'].unique().to_list():
            daily_df = df.filter(pl.col('date') == date).sort('datetime')
            
            # 1. 早盘集合竞价
            opening_auction = daily_df.filter(
                pl.col('trading_phase') == 'opening_auction'
            )
            
            if len(opening_auction) > 0:
                window_end = datetime.combine(date, time(9, 30))
                features = self._extract_all_features(
                    opening_auction, symbol, window_end, 'auction'
                )
                features_list.append(features)
            
            # 2. 连续竞价时段
            continuous_windows = self._process_continuous_windows(
                daily_df, symbol, date
            )
            features_list.extend(continuous_windows)
            
            # 3. 尾盘时段 (14:55-15:00) - 修复的关键部分
            closing_mixed = daily_df.filter(
                (pl.col('time_only') >= time(14, 55)) & 
                (pl.col('time_only') <= time(15, 0, 5))
            )
            
            if len(closing_mixed) > 0:
                window_end = datetime.combine(date, time(15, 0))
                # 判断窗口类型
                has_continuous = (closing_mixed['trading_phase'] == 'continuous').any()
                has_auction = (closing_mixed['trading_phase'] == 'closing_auction').any()
                
                if has_continuous and has_auction:
                    window_type = 'mixed'
                elif has_auction:
                    window_type = 'closing_auction'
                else:
                    window_type = 'continuous'
                
                features = self._extract_all_features(
                    closing_mixed, symbol, window_end, window_type
                )
                features_list.append(features)
        
        if features_list:
            return pl.DataFrame(features_list)
        return pl.DataFrame()
    
    def _process_continuous_windows(self, df: pl.DataFrame, symbol: str, date) -> List[Dict]:
        """处理连续竞价的5分钟窗口"""
        features_list = []
        
        # 定义窗口
        window_ends = []
        
        # 上午
        current = datetime.combine(date, time(9, 35))
        end_morning = datetime.combine(date, time(11, 30))
        while current <= end_morning:
            window_ends.append(current)
            current += timedelta(minutes=5)
        
        # 下午 (到14:55)
        current = datetime.combine(date, time(13, 5))
        end_afternoon = datetime.combine(date, time(14, 55))
        while current <= end_afternoon:
            window_ends.append(current)
            current += timedelta(minutes=5)
        
        # 处理每个窗口
        for window_end in window_ends:
            window_start = window_end - timedelta(minutes=5)
            
            # 特殊处理开盘后第一个窗口
            if window_end.time() == time(9, 35):
                window_start = datetime.combine(date, time(9, 30))
            elif window_end.time() == time(13, 5):
                window_start = datetime.combine(date, time(13, 0))
            
            window_df = df.filter(
                (pl.col('datetime') >= window_start) &
                (pl.col('datetime') < window_end) &
                (pl.col('trading_phase') == 'continuous')
            )
            
            if len(window_df) > 0:
                features = self._extract_all_features(
                    window_df, symbol, window_end, 'continuous'
                )
                features_list.append(features)
        
        return features_list
    
    def _extract_all_features(self, df: pl.DataFrame, symbol: str, 
                            window_end: datetime, window_type: str) -> Dict:
        """提取所有特征 - 保留原始的完整特征集"""
        features = {
            'symbol': symbol,
            'window_end': window_end,
            'window_type': window_type,
            'n_snapshots': len(df)
        }
        
        try:
            # 转换为numpy以便使用原始的特征提取逻辑
            df_dict = df.to_dict(as_series=False)
            
            # 根据窗口类型决定提取策略
            if window_type in ['auction', 'opening_auction']:
                # 纯集合竞价：只使用最后一个切片的深度信息
                features.update(self._extract_auction_features(df, df_dict))
            elif window_type == 'closing_auction':
                # 收盘集合竞价：只使用最后一个切片
                features.update(self._extract_auction_features(df, df_dict))
            elif window_type == 'mixed':
                # 混合窗口：分别处理连续和集合竞价部分
                features.update(self._extract_mixed_window_features(df, df_dict))
            else:
                # 连续竞价：正常提取所有特征
                features.update(self._extract_continuous_features(df, df_dict))
                
        except Exception as e:
            print(f"Feature extraction error for {symbol}: {str(e)}")
        
        return features
    
    def _safe_float(self, value):
        """安全转换为float"""
        try:
            if value is None:
                return 0.0
            return float(value)
        except:
            return 0.0
    
    def _extract_price_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取价格特征 - 优化版本但保留所有原始特征"""
        features = {}
        
        # 使用向量化计算
        price_df = df.select([
            ((pl.col('a1_p') + pl.col('b1_p')) / 2).alias('mid_price'),
            pl.col('a1_p'),
            pl.col('b1_p')
        ])
        
        mid_prices = np.array(df_dict['a1_p']) + np.array(df_dict['b1_p'])
        mid_prices = mid_prices / 2
        
        # 基础统计
        features['mid_price_mean'] = float(np.mean(mid_prices))
        features['mid_price_std'] = float(np.std(mid_prices))
        
        if len(mid_prices) > 0:
            features['mid_price_skew'] = float(stats.skew(mid_prices))
            features['mid_price_kurt'] = float(stats.kurtosis(mid_prices))
        else:
            features['mid_price_skew'] = 0.0
            features['mid_price_kurt'] = 0.0
        
        # 价格变化
        price_returns = np.diff(mid_prices) / mid_prices[:-1]
        if len(price_returns) > 0:
            features['price_return_mean'] = float(np.mean(price_returns))
            features['price_return_std'] = float(np.std(price_returns))
            features['price_return_sum'] = float(np.sum(price_returns))
        else:
            features['price_return_mean'] = 0.0
            features['price_return_std'] = 0.0
            features['price_return_sum'] = 0.0
        
        # 价格范围
        features['price_range'] = float(np.max(mid_prices) - np.min(mid_prices))
        features['price_range_pct'] = features['price_range'] / features['mid_price_mean'] if features['mid_price_mean'] > 0 else 0
        
        # 趋势特征
        if len(mid_prices) > 1:
            time_index = np.arange(len(mid_prices))
            slope, intercept, r_value, _, _ = stats.linregress(time_index, mid_prices)
            features['price_trend_slope'] = float(slope)
            features['price_trend_r2'] = float(r_value ** 2)
        else:
            features['price_trend_slope'] = 0.0
            features['price_trend_r2'] = 0.0
        
        # 各档位价格特征
        for i in range(1, min(4, self.n_levels + 1)):
            if f'b{i}_p' in df_dict and f'a{i}_p' in df_dict:
                features[f'bid{i}_price_std'] = float(np.std(df_dict[f'b{i}_p']))
                features[f'ask{i}_price_std'] = float(np.std(df_dict[f'a{i}_p']))
        
        return features
    
    def _extract_volume_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取成交量特征"""
        features = {}
        
        volumes = np.array(df_dict['volume'])
        volumes_positive = volumes[volumes > 0]
        
        if len(volumes_positive) > 0:
            features['total_volume'] = float(np.sum(volumes_positive))
            features['avg_trade_size'] = float(np.mean(volumes_positive))
            features['volume_std'] = float(np.std(volumes_positive))
            features['n_trades'] = int(len(volumes_positive))
        else:
            features['total_volume'] = 0.0
            features['avg_trade_size'] = 0.0
            features['volume_std'] = 0.0
            features['n_trades'] = 0
        
        # 订单簿深度
        bid_cols = [f'b{i}_v' for i in range(1, self.n_levels + 1) if f'b{i}_v' in df_dict]
        ask_cols = [f'a{i}_v' for i in range(1, self.n_levels + 1) if f'a{i}_v' in df_dict]
        
        if bid_cols and ask_cols:
            bid_totals = np.zeros(len(df))
            ask_totals = np.zeros(len(df))
            
            for col in bid_cols:
                bid_totals += np.array(df_dict[col])
            for col in ask_cols:
                ask_totals += np.array(df_dict[col])
            
            features['avg_bid_depth'] = float(np.mean(bid_totals))
            features['avg_ask_depth'] = float(np.mean(ask_totals))
            features['depth_imbalance'] = (features['avg_bid_depth'] - features['avg_ask_depth']) / \
                                         (features['avg_bid_depth'] + features['avg_ask_depth'] + 1e-8)
        
        # 各档位量的统计
        for i in range(1, min(4, self.n_levels + 1)):
            if f'b{i}_v' in df_dict and f'a{i}_v' in df_dict:
                bid_mean = float(np.mean(df_dict[f'b{i}_v']))
                ask_mean = float(np.mean(df_dict[f'a{i}_v']))
                
                features[f'bid{i}_vol_mean'] = bid_mean
                features[f'ask{i}_vol_mean'] = ask_mean
                features[f'level{i}_vol_imbalance'] = (bid_mean - ask_mean) / (bid_mean + ask_mean + 1e-8)
        
        return features
    
    def _extract_shape_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取订单簿形状特征"""
        features = {}
        
        bid_slopes = []
        ask_slopes = []
        
        # 批量处理形状计算
        for idx in range(len(df)):
            bid_prices = []
            bid_volumes = []
            ask_prices = []
            ask_volumes = []
            
            for i in range(1, min(6, self.n_levels + 1)):
                if f'b{i}_p' in df_dict:
                    bid_prices.append(df_dict[f'b{i}_p'][idx])
                    bid_volumes.append(df_dict[f'b{i}_v'][idx] if f'b{i}_v' in df_dict else 0)
                if f'a{i}_p' in df_dict:
                    ask_prices.append(df_dict[f'a{i}_p'][idx])
                    ask_volumes.append(df_dict[f'a{i}_v'][idx] if f'a{i}_v' in df_dict else 0)
            
            if len(bid_prices) > 2 and len(bid_volumes) > 2:
                bid_cum_vol = np.cumsum(bid_volumes)
                bid_price_diff = bid_prices[0] - np.array(bid_prices)
                if np.sum(bid_cum_vol) > 0:
                    try:
                        bid_slope, _ = np.polyfit(bid_cum_vol, bid_price_diff, 1)
                        bid_slopes.append(bid_slope)
                    except:
                        pass
            
            if len(ask_prices) > 2 and len(ask_volumes) > 2:
                ask_cum_vol = np.cumsum(ask_volumes)
                ask_price_diff = np.array(ask_prices) - ask_prices[0]
                if np.sum(ask_cum_vol) > 0:
                    try:
                        ask_slope, _ = np.polyfit(ask_cum_vol, ask_price_diff, 1)
                        ask_slopes.append(ask_slope)
                    except:
                        pass
        
        # 形状特征统计
        if bid_slopes:
            features['bid_slope_mean'] = float(np.mean(bid_slopes))
            features['bid_slope_std'] = float(np.std(bid_slopes))
        else:
            features['bid_slope_mean'] = 0.0
            features['bid_slope_std'] = 0.0
        
        if ask_slopes:
            features['ask_slope_mean'] = float(np.mean(ask_slopes))
            features['ask_slope_std'] = float(np.std(ask_slopes))
        else:
            features['ask_slope_mean'] = 0.0
            features['ask_slope_std'] = 0.0
        
        # 订单簿对称性
        symmetry_scores = []
        for idx in range(len(df)):
            bid_vols = []
            ask_vols = []
            
            for i in range(1, min(6, self.n_levels + 1)):
                if f'b{i}_v' in df_dict:
                    bid_vols.append(df_dict[f'b{i}_v'][idx])
                if f'a{i}_v' in df_dict:
                    ask_vols.append(df_dict[f'a{i}_v'][idx])
            
            if bid_vols and ask_vols:
                bid_total = sum(bid_vols)
                ask_total = sum(ask_vols)
                if bid_total > 0 and ask_total > 0:
                    bid_vol_norm = np.array(bid_vols) / bid_total
                    ask_vol_norm = np.array(ask_vols) / ask_total
                    min_len = min(len(bid_vol_norm), len(ask_vol_norm))
                    symmetry = 1 - np.mean(np.abs(bid_vol_norm[:min_len] - ask_vol_norm[:min_len]))
                    symmetry_scores.append(symmetry)
        
        if symmetry_scores:
            features['orderbook_symmetry'] = float(np.mean(symmetry_scores))
        else:
            features['orderbook_symmetry'] = 0.5
        
        return features
    
    def _extract_microstructure_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取市场微观结构特征"""
        features = {}
        
        # 价差计算
        spreads = np.array(df_dict['a1_p']) - np.array(df_dict['b1_p'])
        mid_prices = (np.array(df_dict['a1_p']) + np.array(df_dict['b1_p'])) / 2
        relative_spreads = spreads / mid_prices
        
        features['spread_mean'] = float(np.mean(spreads))
        features['spread_std'] = float(np.std(spreads))
        features['relative_spread_mean'] = float(np.mean(relative_spreads))
        
        # 有效价差
        effective_spreads = []
        
        for idx in range(len(df)):
            std_size = (df_dict['b1_v'][idx] + df_dict['a1_v'][idx]) / 2
            
            if std_size > 0:
                bid_cost = 0
                ask_cost = 0
                bid_qty = 0
                ask_qty = 0
                
                for i in range(1, min(6, self.n_levels + 1)):
                    if f'b{i}_p' in df_dict and f'b{i}_v' in df_dict:
                        if bid_qty < std_size:
                            take = min(std_size - bid_qty, df_dict[f'b{i}_v'][idx])
                            bid_cost += take * df_dict[f'b{i}_p'][idx]
                            bid_qty += take
                    
                    if f'a{i}_p' in df_dict and f'a{i}_v' in df_dict:
                        if ask_qty < std_size:
                            take = min(std_size - ask_qty, df_dict[f'a{i}_v'][idx])
                            ask_cost += take * df_dict[f'a{i}_p'][idx]
                            ask_qty += take
                
                if bid_qty > 0 and ask_qty > 0:
                    eff_spread = (ask_cost / ask_qty) - (bid_cost / bid_qty)
                    effective_spreads.append(eff_spread)
        
        if effective_spreads:
            features['effective_spread_mean'] = float(np.mean(effective_spreads))
        else:
            features['effective_spread_mean'] = features['spread_mean']
        
        # 价格影响
        price_impacts = []
        
        current_prices = np.array(df_dict['current'])
        volumes = np.array(df_dict['volume'])
        
        for i in range(1, len(current_prices)):
            if volumes[i] > 0:
                price_change = current_prices[i] - current_prices[i-1]
                signed_volume = volumes[i] if price_change > 0 else -volumes[i]
                if signed_volume != 0:
                    impact = price_change / signed_volume
                    price_impacts.append(abs(impact))
        
        if price_impacts:
            features['price_impact_mean'] = float(np.mean(price_impacts))
            features['price_impact_std'] = float(np.std(price_impacts))
        else:
            features['price_impact_mean'] = 0.0
            features['price_impact_std'] = 0.0
        
        return features
    
    def _extract_order_flow_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取订单流特征"""
        features = {}
        
        flow_imbalances = []
        
        # 计算总深度
        bid_cols = [f'b{i}_v' for i in range(1, min(6, self.n_levels + 1)) if f'b{i}_v' in df_dict]
        ask_cols = [f'a{i}_v' for i in range(1, min(6, self.n_levels + 1)) if f'a{i}_v' in df_dict]
        
        if bid_cols and ask_cols:
            bid_totals = np.zeros(len(df))
            ask_totals = np.zeros(len(df))
            
            for col in bid_cols:
                bid_totals += np.array(df_dict[col])
            for col in ask_cols:
                ask_totals += np.array(df_dict[col])
            
            # 计算变化
            for i in range(1, len(bid_totals)):
                bid_change = bid_totals[i] - bid_totals[i-1]
                ask_change = ask_totals[i] - ask_totals[i-1]
                
                if abs(bid_change) + abs(ask_change) > 0:
                    imbalance = (bid_change - ask_change) / (abs(bid_change) + abs(ask_change))
                    flow_imbalances.append(imbalance)
        
        if flow_imbalances:
            features['order_flow_imbalance_mean'] = float(np.mean(flow_imbalances))
            features['order_flow_imbalance_std'] = float(np.std(flow_imbalances))
            features['order_flow_persistence'] = self._compute_autocorrelation(flow_imbalances, 1)
        else:
            features['order_flow_imbalance_mean'] = 0.0
            features['order_flow_imbalance_std'] = 0.0
            features['order_flow_persistence'] = 0.0
        
        # 激进订单比例
        aggressive_buy_volume = 0.0
        aggressive_sell_volume = 0.0
        
        current_prices = np.array(df_dict['current'])
        volumes = np.array(df_dict['volume'])
        a1_prices = np.array(df_dict['a1_p'])
        b1_prices = np.array(df_dict['b1_p'])
        
        for i in range(len(df)):
            if volumes[i] > 0:
                if abs(current_prices[i] - a1_prices[i]) < abs(current_prices[i] - b1_prices[i]):
                    aggressive_buy_volume += volumes[i]
                else:
                    aggressive_sell_volume += volumes[i]
        
        total_aggressive = aggressive_buy_volume + aggressive_sell_volume
        if total_aggressive > 0:
            features['aggressive_buy_ratio'] = aggressive_buy_volume / total_aggressive
        else:
            features['aggressive_buy_ratio'] = 0.5
        
        return features
    
    def _extract_temporal_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取时序动态特征"""
        features = {}
        
        # 波动率聚类
        mid_prices = (np.array(df_dict['a1_p']) + np.array(df_dict['b1_p'])) / 2
        returns = np.diff(mid_prices) / mid_prices[:-1]
        
        if len(returns) > 10:
            squared_returns = returns ** 2
            features['volatility_clustering'] = self._compute_autocorrelation(squared_returns.tolist(), 1)
            features['realized_volatility'] = float(np.sqrt(np.sum(squared_returns)))
        else:
            features['volatility_clustering'] = 0.0
            features['realized_volatility'] = 0.0
        
        # 订单到达率
        order_arrivals = []
        
        bid_cols = [f'b{i}_v' for i in range(1, self.n_levels + 1) if f'b{i}_v' in df_dict]
        ask_cols = [f'a{i}_v' for i in range(1, self.n_levels + 1) if f'a{i}_v' in df_dict]
        
        if bid_cols and ask_cols:
            bid_totals = np.zeros(len(df))
            ask_totals = np.zeros(len(df))
            
            for col in bid_cols:
                bid_totals += np.array(df_dict[col])
            for col in ask_cols:
                ask_totals += np.array(df_dict[col])
            
            for i in range(1, len(bid_totals)):
                bid_change = abs(bid_totals[i] - bid_totals[i-1])
                ask_change = abs(ask_totals[i] - ask_totals[i-1])
                total_change = bid_change + ask_change
                order_arrivals.append(total_change)
        
        if order_arrivals:
            features['order_arrival_rate'] = float(np.mean(order_arrivals))
            features['order_arrival_std'] = float(np.std(order_arrivals))
        else:
            features['order_arrival_rate'] = 0.0
            features['order_arrival_std'] = 0.0
        
        features['activity_intensity'] = features.get('n_trades', 0) / len(df) if len(df) > 0 else 0.0
        
        return features
    
    def _extract_large_order_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取大单和异常特征"""
        features = {}
        
        # 大单检测
        all_volumes = []
        
        for i in range(1, self.n_levels + 1):
            if f'b{i}_v' in df_dict:
                all_volumes.extend(df_dict[f'b{i}_v'])
            if f'a{i}_v' in df_dict:
                all_volumes.extend(df_dict[f'a{i}_v'])
        
        if all_volumes:
            volume_threshold = np.percentile(all_volumes, 95)
            
            large_bid_count = 0
            large_ask_count = 0
            
            for i in range(1, min(6, self.n_levels + 1)):
                if f'b{i}_v' in df_dict:
                    large_bid_count += np.sum(np.array(df_dict[f'b{i}_v']) > volume_threshold)
                if f'a{i}_v' in df_dict:
                    large_ask_count += np.sum(np.array(df_dict[f'a{i}_v']) > volume_threshold)
            
            features['large_order_bid_ratio'] = large_bid_count / (large_bid_count + large_ask_count + 1)
            features['large_order_frequency'] = (large_bid_count + large_ask_count) / len(df)
        else:
            features['large_order_bid_ratio'] = 0.5
            features['large_order_frequency'] = 0.0
        
        # 冰山订单检测
        iceberg_signals = 0
        
        for i in range(1, len(df)):
            for level in range(1, min(4, self.n_levels + 1)):
                bid_col = f'b{level}_v'
                ask_col = f'a{level}_v'
                
                if bid_col in df_dict and ask_col in df_dict:
                    curr_bid = df_dict[bid_col][i]
                    prev_bid = df_dict[bid_col][i-1]
                    first_bid = df_dict['b1_v'][i]
                    
                    if prev_bid < first_bid * 0.1 and curr_bid > first_bid * 0.5:
                        iceberg_signals += 1
                    
                    curr_ask = df_dict[ask_col][i]
                    prev_ask = df_dict[ask_col][i-1]
                    first_ask = df_dict['a1_v'][i]
                    
                    if prev_ask < first_ask * 0.1 and curr_ask > first_ask * 0.5:
                        iceberg_signals += 1
        
        features['iceberg_signal_count'] = float(iceberg_signals)
        
        return features
    
    def _extract_price_discovery_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取价格发现相关特征"""
        features = {}
        
        # 信息份额
        mid_prices = (np.array(df_dict['a1_p']) + np.array(df_dict['b1_p'])) / 2
        micro_prices = []
        
        for idx in range(len(df)):
            weights_bid = []
            weights_ask = []
            prices_bid = []
            prices_ask = []
            
            for i in range(1, min(4, self.n_levels + 1)):
                if f'b{i}_v' in df_dict and f'a{i}_v' in df_dict:
                    weights_bid.append(df_dict[f'b{i}_v'][idx])
                    weights_ask.append(df_dict[f'a{i}_v'][idx])
                    prices_bid.append(df_dict[f'b{i}_p'][idx])
                    prices_ask.append(df_dict[f'a{i}_p'][idx])
            
            if weights_bid and weights_ask:
                weights_bid = np.array(weights_bid)
                weights_ask = np.array(weights_ask)
                prices_bid = np.array(prices_bid)
                prices_ask = np.array(prices_ask)
                
                if np.sum(weights_bid) > 0 and np.sum(weights_ask) > 0:
                    weighted_bid = np.sum(prices_bid * weights_bid) / np.sum(weights_bid)
                    weighted_ask = np.sum(prices_ask * weights_ask) / np.sum(weights_ask)
                    micro_prices.append((weighted_bid + weighted_ask) / 2)
                else:
                    micro_prices.append((df_dict['b1_p'][idx] + df_dict['a1_p'][idx]) / 2)
        
        # 价格领先指标
        if len(micro_prices) > 10:
            correlation_lags = []
            
            for lag in range(1, min(6, len(mid_prices) // 2)):
                if lag < len(micro_prices) and lag < len(mid_prices):
                    corr = np.corrcoef(micro_prices[:-lag], mid_prices[lag:])[0, 1]
                    correlation_lags.append(corr)
            
            if correlation_lags:
                features['price_discovery_lead'] = float(np.argmax(correlation_lags) + 1)
                features['max_lead_correlation'] = float(np.max(correlation_lags))
            else:
                features['price_discovery_lead'] = 0.0
                features['max_lead_correlation'] = 0.0
        else:
            features['price_discovery_lead'] = 0.0
            features['max_lead_correlation'] = 0.0
        
        # PIN代理指标
        imbalances = []
        
        bid_cols = [f'b{i}_v' for i in range(1, self.n_levels + 1) if f'b{i}_v' in df_dict]
        ask_cols = [f'a{i}_v' for i in range(1, self.n_levels + 1) if f'a{i}_v' in df_dict]
        
        if bid_cols and ask_cols:
            for idx in range(len(df)):
                total_bid = sum(df_dict[col][idx] for col in bid_cols)
                total_ask = sum(df_dict[col][idx] for col in ask_cols)
                if total_bid + total_ask > 0:
                    imbalance = (total_bid - total_ask) / (total_bid + total_ask)
                    imbalances.append(imbalance)
        
        if len(imbalances) > 1:
            features['informed_trading_proxy'] = self._compute_hurst_exponent(imbalances)
        else:
            features['informed_trading_proxy'] = 0.5
        
        return features
    
    def _extract_auction_specific_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取集合竞价特有的特征"""
        features = {}
        
        prices = np.array(df_dict['current'])
        if len(prices) > 1:
            features['auction_price_start'] = float(prices[0])
            features['auction_price_end'] = float(prices[-1])
            features['auction_price_change'] = float((prices[-1] - prices[0]) / prices[0]) if prices[0] != 0 else 0.0
            features['auction_price_volatility'] = float(np.std(prices))
            
            if len(prices) > 5:
                late_prices = prices[-5:]
                late_mean = np.mean(late_prices)
                late_std = np.std(late_prices)
                features['auction_price_stability'] = 1 / (float(late_std) / float(late_mean) + 1e-8)
            else:
                features['auction_price_stability'] = 1.0
        
        volumes = np.array(df_dict['volume'])
        cum_volumes = np.cumsum(volumes)
        
        if len(cum_volumes) > 0:
            features['auction_volume_final'] = float(cum_volumes[-1])
            if len(cum_volumes) > 1 and cum_volumes[0] > 0:
                features['auction_volume_growth'] = float((cum_volumes[-1] - cum_volumes[0]) / cum_volumes[0])
            else:
                features['auction_volume_growth'] = 0.0
        
        return features
    
    def _extract_mixed_specific_features_optimized(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """提取混合时段特有的特征"""
        features = {}
        
        # 判断交易阶段
        trading_phases = df['trading_phase'].to_list()
        
        auction_indices = [i for i, phase in enumerate(trading_phases) if phase == 'closing_auction']
        continuous_indices = [i for i, phase in enumerate(trading_phases) if phase == 'continuous']
        
        if auction_indices and continuous_indices:
            last_auction_idx = auction_indices[-1]
            auction_final_price = float(df_dict['current'][last_auction_idx])
            features['auction_final_price'] = auction_final_price
            
            first_continuous_idx = continuous_indices[0]
            open_price = float(df_dict['current'][first_continuous_idx])
            
            features['open_vs_auction_return'] = (open_price - auction_final_price) / auction_final_price if auction_final_price != 0 else 0.0
            
            if len(continuous_indices) > 1:
                continuous_prices = [df_dict['current'][i] for i in continuous_indices]
                features['post_auction_volatility'] = float(np.std(continuous_prices)) / auction_final_price if auction_final_price != 0 else 0.0
            else:
                features['post_auction_volatility'] = 0.0
        
        return features
    
    def _compute_autocorrelation(self, series: List[float], lag: int) -> float:
        """计算自相关系数"""
        if len(series) <= lag:
            return 0.0
        
        series = np.array(series)
        mean = np.mean(series)
        c0 = np.sum((series - mean) ** 2) / len(series)
        c_lag = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / len(series)
        
        return float(c_lag / (c0 + 1e-8))
    
    def _compute_hurst_exponent(self, series: List[float]) -> float:
        """计算Hurst指数"""
        if len(series) < 10:
            return 0.5
        
        series = np.array(series)
        lags = range(2, min(20, len(series) // 2))
        tau = []
        
        for lag in lags:
            std_values = []
            for i in range(0, len(series) - lag, lag):
                subseries = series[i:i+lag]
                if len(subseries) > 1:
                    mean = np.mean(subseries)
                    cumsum = np.cumsum(subseries - mean)
                    R = np.max(cumsum) - np.min(cumsum)
                    S = np.std(subseries)
                    if S > 0:
                        std_values.append(R / S)
            
            if std_values:
                tau.append(np.mean(std_values))
        
        if len(tau) > 2:
            log_lags = np.log(list(lags)[:len(tau)])
            log_tau = np.log(tau)
            H, _ = np.polyfit(log_lags, log_tau, 1)
            return float(H)
        
        return 0.5
    
    def _extract_auction_features(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """集合竞价特征提取 - 只使用最后一个切片的深度信息"""
        features = {}
        
        # 使用最后一个切片计算深度相关特征
        last_idx = len(df) - 1
        
        # 价格特征 - 使用统一的命名
        mid_prices = (np.array(df_dict['a1_p']) + np.array(df_dict['b1_p'])) / 2
        features['mid_price_mean'] = float(np.mean(mid_prices))
        features['mid_price_std'] = float(np.std(mid_prices))
        
        # 对于集合竞价，偏度和峰度可能意义不大，但保持一致性
        if len(mid_prices) > 2:
            features['mid_price_skew'] = float(stats.skew(mid_prices))
            features['mid_price_kurt'] = float(stats.kurtosis(mid_prices))
        else:
            features['mid_price_skew'] = 0.0
            features['mid_price_kurt'] = 0.0
        
        # 价格变化
        if len(mid_prices) > 1:
            price_returns = np.diff(mid_prices) / mid_prices[:-1]
            features['price_return_mean'] = float(np.mean(price_returns))
            features['price_return_std'] = float(np.std(price_returns))
            features['price_return_sum'] = float(np.sum(price_returns))
        else:
            features['price_return_mean'] = 0.0
            features['price_return_std'] = 0.0
            features['price_return_sum'] = 0.0
        
        features['price_range'] = float(np.max(mid_prices) - np.min(mid_prices))
        features['price_range_pct'] = features['price_range'] / features['mid_price_mean'] if features['mid_price_mean'] > 0 else 0
        
        # 价差特征
        features['spread_mean'] = float(df_dict['a1_p'][last_idx] - df_dict['b1_p'][last_idx])
        features['spread_std'] = 0.0  # 集合竞价期间价差不变
        features['relative_spread_mean'] = features['spread_mean'] / features['mid_price_mean'] if features['mid_price_mean'] > 0 else 0
        
        # 成交量特征
        volumes = np.array(df_dict['volume'])
        volumes_positive = volumes[volumes > 0]
        
        if len(volumes_positive) > 0:
            features['total_volume'] = float(np.sum(volumes_positive))
            features['avg_trade_size'] = float(np.mean(volumes_positive))
            features['volume_std'] = float(np.std(volumes_positive))
            features['n_trades'] = int(len(volumes_positive))
        else:
            features['total_volume'] = 0.0
            features['avg_trade_size'] = 0.0
            features['volume_std'] = 0.0
            features['n_trades'] = 0
        
        # 订单簿深度 - 只有第一档有意义
        features['avg_bid_depth'] = float(df_dict['b1_v'][last_idx])
        features['avg_ask_depth'] = float(df_dict['a1_v'][last_idx])
        features['depth_imbalance'] = (features['avg_bid_depth'] - features['avg_ask_depth']) / \
                                     (features['avg_bid_depth'] + features['avg_ask_depth'] + 1e-8)
        
        # 各档位特征
        features['bid1_vol_mean'] = float(df_dict['b1_v'][last_idx])
        features['ask1_vol_mean'] = float(df_dict['a1_v'][last_idx])
        features['level1_vol_imbalance'] = features['depth_imbalance']
        
        # 其他档位设为0
        for i in range(2, min(4, self.n_levels + 1)):
            features[f'bid{i}_vol_mean'] = 0.0
            features[f'ask{i}_vol_mean'] = 0.0
            features[f'level{i}_vol_imbalance'] = 0.0
            features[f'bid{i}_price_std'] = 0.0
            features[f'ask{i}_price_std'] = 0.0
        
        # 微观结构特征
        features['effective_spread_mean'] = features['spread_mean']
        features['price_impact_mean'] = 0.0
        features['price_impact_std'] = 0.0
        
        # 订单流特征
        features['order_flow_imbalance_mean'] = features['depth_imbalance']
        features['order_flow_imbalance_std'] = 0.0
        features['order_flow_persistence'] = 0.0
        features['aggressive_buy_ratio'] = 0.5
        
        # 时序特征
        features['volatility_clustering'] = 0.0
        features['realized_volatility'] = 0.0
        features['order_arrival_rate'] = 0.0
        features['order_arrival_std'] = 0.0
        features['activity_intensity'] = features['n_trades'] / len(df) if len(df) > 0 else 0.0
        
        # 形状特征 - 集合竞价无意义
        features['bid_slope_mean'] = 0.0
        features['bid_slope_std'] = 0.0
        features['ask_slope_mean'] = 0.0
        features['ask_slope_std'] = 0.0
        features['orderbook_symmetry'] = 0.5
        
        # 大单特征
        features['large_order_bid_ratio'] = 0.5
        features['large_order_frequency'] = 0.0
        features['iceberg_signal_count'] = 0.0
        
        # 价格发现特征
        features['price_discovery_lead'] = 0.0
        features['max_lead_correlation'] = 0.0
        features['informed_trading_proxy'] = 0.5
        
        # 趋势特征
        if len(mid_prices) > 1:
            time_index = np.arange(len(mid_prices))
            slope, intercept, r_value, _, _ = stats.linregress(time_index, mid_prices)
            features['price_trend_slope'] = float(slope)
            features['price_trend_r2'] = float(r_value ** 2)
        else:
            features['price_trend_slope'] = 0.0
            features['price_trend_r2'] = 0.0
        
        return features
    
    def _extract_continuous_features(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """连续竞价特征提取 - 使用所有原始特征"""
        features = {}
        
        # 使用所有原始的特征提取函数
        features.update(self._extract_price_features_optimized(df, df_dict))
        features.update(self._extract_volume_features_optimized(df, df_dict))
        features.update(self._extract_shape_features_optimized(df, df_dict))
        features.update(self._extract_microstructure_features_optimized(df, df_dict))
        features.update(self._extract_order_flow_features_optimized(df, df_dict))
        features.update(self._extract_temporal_features_optimized(df, df_dict))
        features.update(self._extract_large_order_features_optimized(df, df_dict))
        features.update(self._extract_price_discovery_features_optimized(df, df_dict))
        
        return features
    
    def _extract_mixed_window_features(self, df: pl.DataFrame, df_dict: dict) -> Dict[str, float]:
        """混合窗口特征提取 - 分别处理连续和集合竞价部分，但使用相同的特征名"""
        # 分离连续竞价和集合竞价
        trading_phases = df['trading_phase'].to_list()
        
        continuous_indices = [i for i, phase in enumerate(trading_phases) if phase == 'continuous']
        auction_indices = [i for i, phase in enumerate(trading_phases) if phase == 'closing_auction']
        
        if continuous_indices and not auction_indices:
            # 只有连续竞价
            continuous_df = df.filter(pl.col('trading_phase') == 'continuous')
            continuous_dict = continuous_df.to_dict(as_series=False)
            return self._extract_continuous_features(continuous_df, continuous_dict)
        
        elif auction_indices and not continuous_indices:
            # 只有集合竞价
            auction_df = df.filter(pl.col('trading_phase') == 'closing_auction')
            auction_dict = auction_df.to_dict(as_series=False)
            return self._extract_auction_features(auction_df, auction_dict)
        
        else:
            # 混合情况：以连续竞价为主，但价格统计包含全部
            features = {}
            
            # 1. 价格特征 - 使用整个窗口
            features.update(self._extract_price_features_optimized(df, df_dict))
            
            # 2. 对于深度相关特征，使用连续竞价部分
            continuous_df = df.filter(pl.col('trading_phase') == 'continuous')
            continuous_dict = continuous_df.to_dict(as_series=False)
            
            if len(continuous_df) > 0:
                # 成交量特征
                volume_features = self._extract_volume_features_optimized(continuous_df, continuous_dict)
                features.update(volume_features)
                
                # 订单簿形状特征
                shape_features = self._extract_shape_features_optimized(continuous_df, continuous_dict)
                features.update(shape_features)
                
                # 微观结构特征
                micro_features = self._extract_microstructure_features_optimized(continuous_df, continuous_dict)
                features.update(micro_features)
                
                # 订单流特征
                flow_features = self._extract_order_flow_features_optimized(continuous_df, continuous_dict)
                features.update(flow_features)
                
                # 时序特征
                temporal_features = self._extract_temporal_features_optimized(continuous_df, continuous_dict)
                features.update(temporal_features)
                
                # 大单特征
                large_order_features = self._extract_large_order_features_optimized(continuous_df, continuous_dict)
                features.update(large_order_features)
                
                # 价格发现特征
                discovery_features = self._extract_price_discovery_features_optimized(continuous_df, continuous_dict)
                features.update(discovery_features)
            else:
                # 如果没有连续竞价数据，使用集合竞价数据
                auction_df = df.filter(pl.col('trading_phase') == 'closing_auction')
                auction_dict = auction_df.to_dict(as_series=False)
                features.update(self._extract_auction_features(auction_df, auction_dict))
            
            # 3. 添加过渡信息（但不创建新的特征列）
            if continuous_indices and auction_indices:
                # 记录连续到集合的价格变化，但作为已有特征的一部分
                last_continuous_idx = continuous_indices[-1]
                first_auction_idx = auction_indices[0]
                
                continuous_final_price = float(df_dict['current'][last_continuous_idx])
                auction_start_price = float(df_dict['current'][first_auction_idx])
                
                # 这个信息已经包含在price_return等特征中了
            
            return features

if __name__ == "__main__":
    logger = setup_logging()
    args = parser.parse_args()
    sdate = args.sdate
    edate = args.edate
    ob_orig_path = args.ob_orig_path
    save_path = args.save_path
    logger.info(f"Starting order book processing")
    logger.info(f"Date range: {sdate} to {edate}")
    logger.info(f"Source path: {ob_orig_path}")
    logger.info(f"Save path: {save_path}")
    tdates = get_td(sdate,edate).to_numpy().flatten().tolist()
    processor = OptimizedPolarsOrderBookProcessor(n_levels=10, n_workers=1)
    for tdt in tdates:
        logger.info(f"Processing date: {tdt}")
        start_time = datetime.now()
        
        try:
            t = files.read_df(dir_path=ob_orig_path, start=tdt, end=tdt, dt_format='date',sort_cols=['datetime','symbol'])
            t = t.filter(((pl.col('datetime').dt.time()>=dt.time(9,15,0)) & (pl.col('datetime').dt.time()<=dt.time(9,25,5))) | ((pl.col('datetime').dt.time()>=dt.time(9,30,0)) & (pl.col('datetime').dt.time()<=dt.time(11,30,0))) | ((pl.col('datetime').dt.time()>=dt.time(13,0,0)) & (pl.col('datetime').dt.time()<=dt.time(15,0,5))))
            t = t.with_columns(
                pl.col('volume').diff().over('symbol').fill_null(0.0).alias('volume'),
                pl.col('money').diff().over('symbol').fill_null(0.0).alias('money'),
            )
            
            symbols = t['symbol'].unique().to_list()
            total_symbols = len(symbols)
            logger.info(f"Found {total_symbols} symbols to process")
            
            res = []
            for i, symbol in enumerate(symbols):
                if i % 10 == 0:  # 每10个symbol记录一次
                    logger.info(f"Progress: {i}/{total_symbols} symbols processed ({i/total_symbols*100:.1f}%)")
                
                symbol_df = t.filter(pl.col('symbol') == symbol)
                features = processor.process_all_symbols(symbol_df)
                features = features.with_columns(
                    cs.numeric().cast(pl.Float64)
                )
                if len([i for i in expected_columns if i not in features.columns])>0:
                    logger.warning(f"Missing expected columns for {symbol}: {set(expected_columns) - set(features.columns)}")
                    continue
                res.append(features)
            
            logger.info(f"All symbols processed, concatenating results...")
            df=pl.concat(res)
            df = df.rename({'symbol':'code','window_end':'datetime'})
            
            logger.info(f"Saving data for {tdt}...")
            files.save_table(table=df.with_columns(pl.col('datetime').dt.date().alias('date')),
                           save_dir=save_path, partition_by='date', fformat='parquet', 
                           mode='cover', mkdir=True, with_tm=False, shrink_mem=False, index=None)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Successfully completed {tdt} in {elapsed:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing {tdt}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    logger.info("All dates processed successfully")